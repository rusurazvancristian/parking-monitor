#!/usr/bin/env python3
"""
Parking spot monitor + security sentinel
Raspberry Pi 5 + Camera Module 3 (IMX708) + Hailo-8 (26 TOPS) + YOLOv8s

Controls:
  Q / ESC  - quit
  F        - full reset (parking states + night mode, keep geometry)
  P        - add a new parking spot  (click 4 corners)
  X        - define security zone for last parking spot  (click 4 corners)
  BkSp     - remove the last parking spot
  V        - toggle night mode (long exposure + high gain)
  H        - toggle heatmap overlay
  S        - send snapshot to Telegram
  C        - recalibrate camera stabilizer (set new reference frame)
"""

import sys
import cv2
import json
import time
import signal
import logging
import subprocess
import threading
import numpy as np
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)
from scipy.optimize import linear_sum_assignment

# ─── Paths ────────────────────────────────────────────────────────────────────

HEF_PATH    = "/usr/local/hailo/resources/models/hailo8/yolov8m.hef"
CONFIG_PATH = Path.home() / ".traffic_counter.json"
LOG_DIR     = Path.home() / "traffic_logs"
INPUT_SIZE  = 640

# ─── COCO classes ─────────────────────────────────────────────────────────────

TRACKED = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PARK_CLASSES = {1, 2, 3, 5, 7}   # vehicles only (not pedestrians)

COLORS = {
    "pedestrian": (0,   220, 220),
    "bicycle":    (255, 140,   0),
    "car":        (0,   230,   0),
    "motorcycle": (0,   165, 255),
    "bus":        (60,   20, 220),
    "truck":      (180,   0, 255),
}

FONT    = cv2.FONT_HERSHEY_SIMPLEX
STATS_W = 400
WIN_Y   = 40

# ─── Auto night mode ──────────────────────────────────────────────────────────

LUMA_TO_NIGHT         = 45    # day-mode EMA luma below this  → start night timer
LUMA_TO_NIGHT_CANCEL  = 70    # day-mode EMA luma above this  → cancel night timer (hysteresis)
LUMA_TO_DAY_NIGHT_EXP = 160   # night-mode EMA luma above this → start day timer
LUMA_TO_DAY_CANCEL    = 130   # night-mode EMA luma below this → cancel day timer (hysteresis)
LUMA_EMA_ALPHA        = 0.03  # EMA weight for new frame (~33-frame window ≈ 1.5 s at 25 fps)
AUTO_NIGHT_DELAY_S    = 10.0  # seconds EMA must stay past threshold before switching

# ─── Parking state machine ────────────────────────────────────────────────────

PARK_CONFIRM_S      = 1.0    # seconds vehicle must be present before → OCCUPIED
PARK_RELEASE_S      = 7.5    # seconds of absence before → FREE (time-based, mode-independent)
TG_STATE_COOLDOWN_S = 60     # min seconds between Telegram messages for the same spot
SPOT_OCCUPY_THRESH  = 0.30   # confidence-weighted overlap threshold (overlap × conf ≥ this)
LOITER_THRESHOLD_S  = 4.0    # pedestrian must stay in zone this long before alert triggers
CONF_DAY            = 0.38   # detection confidence threshold in day mode
CONF_NIGHT          = 0.22   # lower threshold at night (model less confident on dark frames)
MIN_BBOX_AREA       = 2000   # minimum bbox area in camera px² — rejects tiny noise detections
TRACK_AGE_MIN       = 3      # track must be this many frames old before used for parking

# ─── Bbox smoothing / heatmap ─────────────────────────────────────────────────

SMOOTH_ALPHA  = 0.4
HEATMAP_DECAY = 0.9997

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "cam_w": 1920, "cam_h": 1080,
    "conf": 0.38,
    "disp_w": 1280, "disp_h": 720,
    "parking_spots": [],
    "telegram_token":   "",
    "telegram_chat_id": "",
}

def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg

def save_cfg(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ─── Telegram ─────────────────────────────────────────────────────────────────

def _tg_worker(token: str, chat_id: str, text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=8,
        )
    except Exception as exc:
        logging.warning("Telegram send failed: %s", exc)


def send_telegram(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    threading.Thread(target=_tg_worker, args=(token, chat_id, text),
                     daemon=True).start()


def _tg_photo_worker(token: str, chat_id: str, jpeg_bytes: bytes, caption: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("snap.jpg", jpeg_bytes, "image/jpeg")},
            timeout=20,
        )
    except Exception as exc:
        logging.warning("Telegram photo send failed: %s", exc)


def send_telegram_photo(token: str, chat_id: str, frame_bgr: np.ndarray, caption: str = ""):
    if not token or not chat_id:
        return
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return
    threading.Thread(target=_tg_photo_worker,
                     args=(token, chat_id, buf.tobytes(), caption),
                     daemon=True).start()


# ─── Telegram video — H.264 transcode so Telegram plays inline ────────────────

def _tg_video_worker(token: str, chat_id: str, path: Path, caption: str):
    h264_path = path.with_name(path.stem + "_h264.mp4")
    send_path = path
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "28", "-an",
             str(h264_path)],
            check=True, capture_output=True, timeout=120,
        )
        path.unlink(missing_ok=True)
        send_path = h264_path
    except Exception as exc:
        logging.warning("ffmpeg transcode failed: %s — sending original", exc)
    try:
        with open(send_path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={"chat_id": chat_id, "caption": caption},
                files={"video": ("alert.mp4", f, "video/mp4")},
                timeout=120,
            )
        logging.info("Alert video sent to Telegram")
    except Exception as exc:
        logging.warning("Telegram video send failed: %s", exc)
    finally:
        try:
            send_path.unlink()
        except Exception:
            pass


def send_telegram_video(token: str, chat_id: str, path: Path, caption: str = ""):
    if not token or not chat_id:
        path.unlink(missing_ok=True)
        return
    threading.Thread(target=_tg_video_worker, args=(token, chat_id, path, caption),
                     daemon=True).start()


# ─── Shared frame store ────────────────────────────────────────────────────────

class FrameStore:
    def __init__(self):
        self._lock    = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._caption = ""

    def put(self, frame: np.ndarray, caption: str):
        with self._lock:
            self._frame   = frame.copy()
            self._caption = caption

    def get(self) -> Tuple[Optional[np.ndarray], str]:
        with self._lock:
            return self._frame, self._caption


def _poll_worker(token: str, chat_id: str, store: FrameStore):
    offset: Optional[int] = None
    logging.info("Telegram poller ready — send /snap to your bot for a live snapshot")
    while True:
        try:
            params: dict = {"timeout": 30, "allowed_updates": ["message"]}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params=params, timeout=40,
            )
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg     = upd.get("message", {})
                text    = msg.get("text", "").strip()
                from_id = str(msg.get("chat", {}).get("id", ""))
                if from_id != chat_id:
                    continue
                cmd = text.split()[0].lower() if text else ""
                if cmd in ("/snap", "/snapshot", "/foto"):
                    frame, caption = store.get()
                    if frame is not None:
                        send_telegram_photo(token, chat_id, frame,
                                            f"📸 On-demand snapshot\n{caption}")
                        logging.info("Snapshot sent (Telegram command '%s')", cmd)
                    else:
                        _tg_worker(token, chat_id,
                                   "⚠️ No frame ready yet — try again in a moment.")
        except Exception as exc:
            logging.warning("Telegram poller: %s — retrying in 5 s", exc)
            time.sleep(5)


def start_telegram_poller(token: str, chat_id: str, store: FrameStore):
    if not token or not chat_id:
        return
    threading.Thread(target=_poll_worker, args=(token, chat_id, store),
                     daemon=True, name="tg-poller").start()

# ─── Image utils ──────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas, s, px, py


def unbox(y1n, x1n, y2n, x2n, s, px, py, W, H) -> Tuple[int, int, int, int]:
    x1 = np.clip((x1n * INPUT_SIZE - px) / s, 0, W)
    y1 = np.clip((y1n * INPUT_SIZE - py) / s, 0, H)
    x2 = np.clip((x2n * INPUT_SIZE - px) / s, 0, W)
    y2 = np.clip((y2n * INPUT_SIZE - py) / s, 0, H)
    return int(x1), int(y1), int(x2), int(y2)

# ─── Click state ──────────────────────────────────────────────────────────────

_click_pts: List[Tuple[int, int]] = []
_click_max = 4

def _on_click(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN and len(_click_pts) < _click_max:
        _click_pts.append((x, y))

# ─── Parking spot selection ───────────────────────────────────────────────────

def pick_parking_spot(frame_bgr: np.ndarray, dw: int, dh: int,
                      name: str) -> Optional[np.ndarray]:
    global _click_pts, _click_max
    _click_max = 4
    _click_pts.clear()
    sx = frame_bgr.shape[1] / dw
    sy = frame_bgr.shape[0] / dh

    WIN = f"Define '{name}'  |  Click 4 corners  |  ENTER=confirm  R=redo  ESC=cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)
    cv2.moveWindow(WIN, 0, WIN_Y)
    cv2.setMouseCallback(WIN, _on_click)

    while True:
        disp = cv2.resize(frame_bgr, (dw, dh))
        n = len(_click_pts)
        msg = (f"Click corner {n+1}/4 for '{name}'" if n < 4
               else "ENTER = confirm  |  R = redo")
        cv2.putText(disp, msg, (20, 44), FONT, 1.1, (50, 220, 50), 2)
        for i, pt in enumerate(_click_pts):
            cv2.circle(disp, pt, 10, (50, 220, 50), -1)
            cv2.putText(disp, str(i + 1), (pt[0] + 12, pt[1] - 6), FONT, 0.7, (255, 255, 255), 2)
        if n >= 2:
            for i in range(n - 1):
                cv2.line(disp, _click_pts[i], _click_pts[i + 1], (50, 220, 50), 2)
        if n == 4:
            cv2.line(disp, _click_pts[3], _click_pts[0], (50, 220, 50), 2)
            pts_disp = np.array(_click_pts, np.int32).reshape(-1, 1, 2)
            overlay = disp.copy()
            cv2.fillPoly(overlay, [pts_disp], (50, 220, 50))
            cv2.addWeighted(overlay, 0.25, disp, 0.75, 0, disp)
        cv2.imshow(WIN, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and n == 4:
            break
        if k == ord('r'):
            _click_pts.clear()
        if k == 27:
            cv2.destroyWindow(WIN)
            return None

    cv2.destroyWindow(WIN)
    cam_pts = [(int(p[0] * sx), int(p[1] * sy)) for p in _click_pts]
    return np.array(cam_pts, np.int32).reshape(-1, 1, 2)


def pick_spot_name(default: str, dw: int, dh: int) -> str:
    """On-screen text input to name a new parking spot."""
    text = default
    WIN  = "Name this spot  —  ENTER confirm  |  ESC default  |  BkSp delete"
    W, H = min(dw, 680), 110
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.moveWindow(WIN, 0, WIN_Y)
    while True:
        canvas = np.full((H, W, 3), 20, np.uint8)
        cv2.putText(canvas, "Enter spot name:", (16, 30), FONT,
                    0.6, (140, 200, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (10, 42), (W - 10, H - 8), (38, 38, 38), -1)
        cv2.putText(canvas, text + "|", (20, 85), FONT,
                    0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)
        k = cv2.waitKey(40) & 0xFF
        if k == 13:                   # Enter — confirm
            break
        if k == 27:                   # ESC — revert to default
            text = default
            break
        if k in (8, 127) and text:    # Backspace
            text = text[:-1]
        elif 32 <= k <= 126:           # Printable ASCII
            text += chr(k)
    cv2.destroyWindow(WIN)
    return text.strip() or default


# ─── Parking spot dataclass ───────────────────────────────────────────────────

@dataclass
class ParkingSpot:
    name: str
    poly: np.ndarray
    state: str = "FREE"
    first_seen_t:   Optional[float] = None   # when vehicle first continuously appeared
    last_seen_t:    Optional[float] = None   # when vehicle was last seen
    occupied_since: Optional[float] = None
    alert_until:    float = 0.0
    alert_zone: Optional[np.ndarray] = None
    alert_cooldown_until: float           = field(default=0.0,  repr=False)
    recording:            bool            = field(default=False, repr=False)
    record_writer:        Optional[object] = field(default=None, repr=False)
    record_path:          Optional[Path]  = field(default=None,  repr=False)
    record_end:           float           = field(default=0.0,  repr=False)
    zone_entry_time:      Optional[float] = field(default=None,  repr=False)
    last_tg_occupied:     float           = field(default=0.0,   repr=False)
    last_tg_free:         float           = field(default=0.0,   repr=False)
    canonical_poly:       Optional[np.ndarray] = field(default=None, repr=False)
    canonical_alert_zone: Optional[np.ndarray] = field(default=None, repr=False)

    def bbox_overlap(self, x1: int, y1: int, x2: int, y2: int) -> float:
        spot_area = float(cv2.contourArea(self.poly))
        if spot_area < 1:
            return 0.0
        bbox_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        spot_pts = self.poly.reshape(-1, 2).astype(np.float32)
        retval, inter = cv2.intersectConvexConvex(
            cv2.convexHull(bbox_pts), cv2.convexHull(spot_pts))
        if retval == 0 or inter is None or len(inter) == 0:
            return 0.0
        return float(cv2.contourArea(inter)) / spot_area

    def to_cfg(self) -> dict:
        poly_to_save = (self.canonical_poly if self.canonical_poly is not None
                        else self.poly)
        d: dict = {"name": self.name, "poly": poly_to_save.reshape(-1, 2).tolist()}
        if self.alert_zone is not None:
            az = (self.canonical_alert_zone if self.canonical_alert_zone is not None
                  else self.alert_zone)
            d["alert_zone"] = az.reshape(-1, 2).tolist()
        return d

    @staticmethod
    def from_cfg(d: dict) -> "ParkingSpot":
        poly = np.array(d["poly"], np.int32).reshape(-1, 1, 2)
        spot = ParkingSpot(name=d["name"], poly=poly.copy(),
                           canonical_poly=poly.copy())
        if "alert_zone" in d:
            az = np.array(d["alert_zone"], np.int32).reshape(-1, 1, 2)
            spot.alert_zone            = az.copy()
            spot.canonical_alert_zone  = az.copy()
        return spot


def vehicle_in_spot(spot: ParkingSpot, tracks: Dict) -> bool:
    for t in tracks.values():
        if t.miss > 5 or t.cls not in PARK_CLASSES or t.age < TRACK_AGE_MIN:
            continue
        x1, y1, x2, y2 = t.bbox
        if spot.bbox_overlap(x1, y1, x2, y2) * t.conf >= SPOT_OCCUPY_THRESH:
            return True
    return False


def threat_in_zone(spot: ParkingSpot, tracks: Dict) -> bool:
    """True if any pedestrian overlaps the alert zone but is NOT inside the parking polygon."""
    if spot.alert_zone is None:
        return False
    for t in tracks.values():
        if t.miss > 0 or t.cls != 0:   # pedestrians only
            continue
        x1, y1, x2, y2 = t.bbox
        candidates = [
            (float(t.cx), float(t.cy)),
            (float(x1), float(y1)), (float(x2), float(y1)),
            (float(x1), float(y2)), (float(x2), float(y2)),
        ]
        for pt in candidates:
            if cv2.pointPolygonTest(spot.alert_zone, pt, False) >= 0:
                centroid = (float(t.cx), float(t.cy))
                if cv2.pointPolygonTest(spot.poly, centroid, False) < 0:
                    return True
                break
    return False

# ─── Parking event logger ─────────────────────────────────────────────────────

class ParkingLogger:
    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / f"parking_{datetime.now():%Y%m%d}.csv"
        if not self.path.exists():
            self.path.write_text("timestamp,spot,event,duration_secs\n")
        logging.info("Parking log: %s", self.path)

    def log(self, spot_name: str, event: str, duration: Optional[float] = None):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dur = f"{duration:.0f}" if duration is not None else ""
        with open(self.path, "a") as f:
            f.write(f"{ts},{spot_name},{event},{dur}\n")
        logging.info("PARKING %s → %s%s", spot_name, event,
                     f" (occupied {duration:.0f}s)" if duration else "")

# ─── Tracker ──────────────────────────────────────────────────────────────────

class Track:
    __slots__ = ("tid", "cls", "bbox", "smooth_bbox", "cx", "cy", "pcx", "pcy",
                 "age", "miss", "cls_votes", "conf")

    def __init__(self, tid: int, cls: int, bbox, cx: float, cy: float, conf: float = 1.0):
        self.tid = tid; self.cls = cls; self.bbox = bbox
        self.smooth_bbox = bbox
        self.cx, self.cy = cx, cy
        self.pcx = self.pcy = None
        self.age = self.miss = 0
        self.cls_votes: dict = {cls: 1}
        self.conf: float = conf


def _bbox_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


class Tracker:
    def __init__(self, max_dist: float = 200, max_miss: int = 15):
        self.tracks: Dict[int, Track] = {}
        self._nid = 0
        self.max_dist = max_dist
        self.max_miss = max_miss

    def update(self, dets: List) -> Dict[int, Track]:
        for t in self.tracks.values():
            t.miss += 1
        if dets:
            new_cx = np.array([(d[1] + d[3]) / 2.0 for d in dets])
            new_cy = np.array([(d[2] + d[4]) / 2.0 for d in dets])
            active = {k: v for k, v in self.tracks.items() if v.miss <= self.max_miss}
            if active:
                tids   = list(active.keys())
                # Predict next position via constant velocity
                pred_cx = np.array([
                    active[i].cx * 2 - active[i].pcx
                    if active[i].pcx is not None else active[i].cx
                    for i in tids
                ])
                pred_cy = np.array([
                    active[i].cy * 2 - active[i].pcy
                    if active[i].pcy is not None else active[i].cy
                    for i in tids
                ])
                # Centroid distance cost
                dx   = pred_cx[:, None] - new_cx[None, :]
                dy   = pred_cy[:, None] - new_cy[None, :]
                dist_cost = np.sqrt(dx ** 2 + dy ** 2)
                # IOU cost — predict each track's bbox displaced by velocity
                iou_mat = np.zeros_like(dist_cost)
                for ri, tid in enumerate(tids):
                    tr = active[tid]
                    bx1, by1, bx2, by2 = tr.bbox
                    dvx = (tr.cx - tr.pcx) if tr.pcx is not None else 0.0
                    dvy = (tr.cy - tr.pcy) if tr.pcy is not None else 0.0
                    pred_box = (bx1 + dvx, by1 + dvy, bx2 + dvx, by2 + dvy)
                    for ci, det in enumerate(dets):
                        det_box = (det[1], det[2], det[3], det[4])
                        iou_mat[ri, ci] = _bbox_iou(pred_box, det_box)
                # Combined cost: distance reduced when IOU is high
                cost = dist_cost * (1.0 - 0.5 * iou_mat)
                ri_all, ci_all = linear_sum_assignment(cost)
                matched_d = set()
                for ri, ci in zip(ri_all, ci_all):
                    if dist_cost[ri, ci] <= self.max_dist:
                        t = active[tids[ri]]
                        cls, x1, y1, x2, y2, conf = dets[ci]
                        t.pcx, t.pcy = t.cx, t.cy
                        t.cx, t.cy   = float(new_cx[ci]), float(new_cy[ci])
                        t.bbox = (x1, y1, x2, y2)
                        a = SMOOTH_ALPHA
                        sx1, sy1, sx2, sy2 = t.smooth_bbox
                        t.smooth_bbox = (sx1*(1-a)+x1*a, sy1*(1-a)+y1*a,
                                         sx2*(1-a)+x2*a, sy2*(1-a)+y2*a)
                        t.cls_votes[cls] = t.cls_votes.get(cls, 0) + 1
                        t.cls  = max(t.cls_votes, key=t.cls_votes.get)
                        t.conf = t.conf * 0.7 + conf * 0.3   # EMA-smooth confidence
                        t.miss = 0; t.age += 1
                        matched_d.add(ci)
                for j, det in enumerate(dets):
                    if j not in matched_d:
                        self._new(det, new_cx[j], new_cy[j])
            else:
                for j, det in enumerate(dets):
                    self._new(det, new_cx[j], new_cy[j])
        self.tracks = {k: v for k, v in self.tracks.items() if v.miss <= self.max_miss}
        return self.tracks

    def _new(self, det, cx: float, cy: float):
        cls, x1, y1, x2, y2, conf = det
        self.tracks[self._nid] = Track(self._nid, cls, (x1, y1, x2, y2), cx, cy, conf)
        self._nid += 1

# ─── Camera stabilizer ────────────────────────────────────────────────────────

class CameraStabilizer:
    """
    ORB + RANSAC homography — tracks camera drift and warps parking polygons
    so they stay aligned even if the camera is nudged.

    Reference frame is set once at startup (or on C key).
    Every ~3 s, update() estimates H (current → ref) via ORB feature matching.
    transform_poly()         maps reference-frame coords → current camera frame.
    inverse_transform_pts()  maps current camera frame  → reference frame.
    """

    STATUS_OK    = "STAB OK"
    STATUS_DRIFT = "DRIFT"
    STATUS_LOST  = "LOST"

    _MIN_INLIERS  = 12
    _GOOD_INLIERS = 28

    def __init__(self, max_features: int = 1500):
        self._orb      = cv2.ORB_create(nfeatures=max_features)
        self._bf       = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # CLAHE pulls out local contrast on overexposed/low-texture surfaces
        self._clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._ref_kp   = None
        self._ref_desc = None
        self._H        = np.eye(3, dtype=np.float64)   # current → ref
        self._H_inv    = np.eye(3, dtype=np.float64)   # ref → current
        self.status    = self.STATUS_OK
        self._has_ref  = False

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale + CLAHE — makes features visible on bright flat surfaces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        return self._clahe.apply(gray)

    def set_reference(self, frame: np.ndarray):
        gray = self._enhance(frame)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is not None and len(kp) >= self._MIN_INLIERS:
            self._ref_kp   = kp
            self._ref_desc = desc
            self._H        = np.eye(3, dtype=np.float64)
            self._H_inv    = np.eye(3, dtype=np.float64)
            self._has_ref  = True
            self.status    = self.STATUS_OK
            logging.info("Stabilizer reference set (%d features)", len(kp))
        else:
            logging.warning("Stabilizer: too few features in reference frame (%d)",
                            len(kp) if kp else 0)

    def update(self, frame: np.ndarray) -> bool:
        """Recompute homography from current frame to reference. Returns True if stable."""
        if not self._has_ref:
            return False
        gray = self._enhance(frame)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < self._MIN_INLIERS:
            self.status = self.STATUS_LOST
            return False
        try:
            matches = self._bf.match(desc, self._ref_desc)
        except cv2.error:
            self.status = self.STATUS_LOST
            return False
        matches = sorted(matches, key=lambda m: m.distance)[:150]
        if len(matches) < self._MIN_INLIERS:
            self.status = self.STATUS_LOST
            return False
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            self.status = self.STATUS_LOST
            return False
        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < self._MIN_INLIERS:
            self.status = self.STATUS_LOST
            return False
        self._H     = H
        self._H_inv = np.linalg.inv(H)
        self.status = (self.STATUS_OK if n_inliers >= self._GOOD_INLIERS
                       else self.STATUS_DRIFT)
        return True

    def transform_poly(self, canonical_pts: np.ndarray) -> np.ndarray:
        """Map polygon from reference frame → current camera frame."""
        if not self._has_ref:
            return canonical_pts
        pts = canonical_pts.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self._H_inv)
        return out.reshape(-1, 1, 2).astype(np.int32)

    def inverse_transform_pts(self, current_pts: np.ndarray) -> np.ndarray:
        """Map polygon from current camera frame → reference frame."""
        if not self._has_ref:
            return current_pts
        pts = current_pts.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self._H)
        return out.reshape(-1, 1, 2).astype(np.int32)


# ─── Renderer helpers ─────────────────────────────────────────────────────────

def _draw_dashed_poly(img: np.ndarray, pts: np.ndarray, color,
                      thickness: int = 2, dash: int = 14):
    flat = pts.reshape(-1, 2).astype(float)
    n    = len(flat)
    for i in range(n):
        p1, p2 = flat[i], flat[(i + 1) % n]
        seg    = p2 - p1
        dist   = float(np.hypot(*seg))
        if dist < 1:
            continue
        steps = max(1, int(dist / (dash * 2)))
        for s in range(steps):
            t1 = min(1.0, s * 2 * dash / dist)
            t2 = min(1.0, (s * 2 * dash + dash) / dist)
            cv2.line(img,
                     tuple((p1 + t1 * seg).astype(int)),
                     tuple((p1 + t2 * seg).astype(int)),
                     color, thickness, cv2.LINE_AA)


# ─── Renderer ─────────────────────────────────────────────────────────────────

def render_frame(bgr: np.ndarray, tracks: Dict, fps: float,
                 dw: int, dh: int, spots: List[ParkingSpot]) -> np.ndarray:

    cam_h, cam_w = bgr.shape[:2]
    sx, sy = dw / cam_w, dh / cam_h
    out = cv2.resize(bgr, (dw, dh))

    # Alert zone overlays (dashed orange)
    for spot in spots:
        if spot.alert_zone is None:
            continue
        d_az = (spot.alert_zone.reshape(-1, 2) * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
        _draw_dashed_poly(out, d_az, (0, 140, 255), thickness=2, dash=12)

    # Parking spot overlays
    now_mono = time.monotonic()
    for spot in spots:
        d_pts = (spot.poly.reshape(-1, 2) * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
        occupied   = spot.state == "OCCUPIED"
        fill_color = (0,  30, 220) if occupied else (0, 180,  40)
        line_color = (0,  60, 255) if occupied else (0, 230,  60)
        overlay = out.copy()
        cv2.fillPoly(overlay, [d_pts], fill_color)
        cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
        cv2.polylines(out, [d_pts], True, line_color, 3 if occupied else 2)
        cx_d = int(d_pts[:, 0, 0].mean())
        cy_d = int(d_pts[:, 0, 1].mean())
        status_txt = "OCCUPIED" if occupied else "FREE"
        status_col = (80, 80, 255) if occupied else (80, 230, 80)
        (tw, _), _ = cv2.getTextSize(spot.name, FONT, 0.65, 2)
        cv2.putText(out, spot.name, (cx_d - tw // 2, cy_d - 6), FONT, 0.65, (255, 255, 255), 2)
        (sw, sh), _ = cv2.getTextSize(status_txt, FONT, 0.55, 2)
        cv2.putText(out, status_txt, (cx_d - sw // 2, cy_d + sh + 4), FONT, 0.55, status_col, 2)
        if occupied and spot.occupied_since:
            dur = int(time.time() - spot.occupied_since)
            dur_txt = (f"{dur // 3600}h {(dur % 3600) // 60}m"
                       if dur >= 3600 else f"{dur // 60}m {dur % 60:02d}s")
            (dtw, _), _ = cv2.getTextSize(dur_txt, FONT, 0.46, 1)
            cv2.putText(out, dur_txt, (cx_d - dtw // 2, cy_d + sh + 22),
                        FONT, 0.46, (190, 190, 220), 1, cv2.LINE_AA)

    # Alert banner
    alert_names = [sp.name for sp in spots
                   if sp.state == "OCCUPIED" and now_mono < sp.alert_until]
    if alert_names:
        b_col = (0, 0, 200) if int(now_mono * 4) % 2 == 0 else (0, 0, 160)
        cv2.rectangle(out, (0, 0), (dw, 52), b_col, -1)
        cv2.putText(out, f"  ALERT: {', '.join(alert_names)} OCCUPIED!",
                    (10, 36), FONT, 1.1, (255, 255, 255), 2)

    # Bounding boxes
    for t in tracks.values():
        if t.miss > 5:
            continue
        name  = TRACKED.get(t.cls, "?")
        color = COLORS.get(name, (180, 180, 180))
        if t.miss > 0:
            color = tuple(max(0, c // 3) for c in color)
        x1, y1, x2, y2 = t.smooth_bbox
        bx1, by1 = int(x1 * sx), int(y1 * sy)
        bx2, by2 = int(x2 * sx), int(y2 * sy)
        thickness = 2 if t.miss == 0 else 1
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thickness)
        label = f"{name} #{t.tid}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), color, -1)
        cv2.putText(out, label, (bx1 + 2, by1 - 3), FONT, 0.5, (0, 0, 0), 1)

    # Corner HUD
    cv2.putText(out, f"FPS {fps:.1f}",
                (dw - 110, 28), FONT, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(out, datetime.now().strftime("%H:%M:%S"),
                (dw - 166, 54), FONT, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
    return out

# ─── Stats panel ──────────────────────────────────────────────────────────────

def render_stats(fps: float, spots: List[ParkingSpot], now_t: float,
                 panel_w: int = STATS_W, panel_h: int = 720) -> np.ndarray:

    BG      = (18,  18,  18)
    BG2     = (28,  28,  28)
    SEP     = (48,  48,  48)
    C_TITLE = (60, 210, 255)
    C_HEAD  = (190, 190, 190)
    C_WHITE = (235, 235, 235)
    C_DIM   = (100, 100, 100)
    C_OCC   = (60,  60, 220)
    C_FREE  = (40, 185,  40)
    C_BAR_BG   = (40,  40,  40)
    C_BAR_IN   = (40, 180,  40)   # green — confirming occupancy
    C_BAR_OUT  = (55,  55, 195)   # blue  — counting down to FREE
    C_BAR_FULL = (35, 130,  35)   # full green — vehicle solid in spot

    out = np.full((panel_h, panel_w, 3), BG, np.uint8)

    def sep_line(y):
        cv2.line(out, (12, y), (panel_w - 12, y), SEP, 1)

    def section(y, title):
        cv2.putText(out, title, (14, y), FONT, 0.55, C_HEAD, 1, cv2.LINE_AA)
        sep_line(y + 6)
        return y + 22

    cv2.rectangle(out, (0, 0), (panel_w, 50), BG2, -1)
    cv2.putText(out, "PARKING MONITOR", (14, 34), FONT, 0.9, C_TITLE, 2, cv2.LINE_AA)
    sep_line(50)
    y = 68

    now = datetime.now()
    cv2.putText(out, now.strftime("%d/%m/%Y"), (14, y),
                FONT, 0.58, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(out, now.strftime("%H:%M:%S"), (panel_w - 140, y),
                FONT, 0.58, C_WHITE, 1, cv2.LINE_AA)
    y += 22
    cv2.putText(out, f"FPS  {fps:.1f}", (14, y), FONT, 0.55, C_DIM, 1, cv2.LINE_AA)

    # Summary pill (e.g. "2 FREE  /  1 OCC")
    if spots:
        n_occ  = sum(1 for sp in spots if sp.state == "OCCUPIED")
        n_free = len(spots) - n_occ
        s_col  = C_OCC if n_occ == len(spots) else C_FREE if n_occ == 0 else C_WHITE
        summary = f"{n_free} FREE  /  {n_occ} OCC"
        (smw, _), _ = cv2.getTextSize(summary, FONT, 0.52, 1)
        cv2.putText(out, summary, (panel_w - smw - 12, y),
                    FONT, 0.52, s_col, 1, cv2.LINE_AA)
    y += 18
    sep_line(y); y += 18

    y = section(y, "PARKING SPOTS")
    bar_x1 = 14;  bar_x2 = panel_w - 14;  bar_bw = bar_x2 - bar_x1

    for spot in spots:
        if y + 50 > panel_h - 80:
            break
        occupied = spot.state == "OCCUPIED"
        dot_col  = C_OCC if occupied else C_FREE

        # Dot + name
        cv2.circle(out, (22, y - 4), 8, dot_col, -1, cv2.LINE_AA)
        cv2.putText(out, spot.name, (38, y), FONT, 0.56, C_WHITE, 1, cv2.LINE_AA)

        # State (right-aligned)
        (stw, _), _ = cv2.getTextSize(spot.state, FONT, 0.56, 2)
        cv2.putText(out, spot.state, (panel_w - stw - 12, y),
                    FONT, 0.56, dot_col, 2, cv2.LINE_AA)

        # Duration (below name, shown when occupied)
        if occupied and spot.occupied_since:
            dur = int(now_t - spot.occupied_since)
            dur_txt = (f"{dur // 3600}h {(dur % 3600) // 60}m"
                       if dur >= 3600 else f"{dur // 60}m {dur % 60:02d}s")
            cv2.putText(out, dur_txt, (38, y + 17), FONT, 0.47, C_DIM, 1, cv2.LINE_AA)

        # Progress bar (8 px tall, spanning full width)
        bar_y = y + 28
        cv2.rectangle(out, (bar_x1, bar_y), (bar_x2, bar_y + 6), C_BAR_BG, -1)

        if not occupied and spot.first_seen_t is not None:
            # Vehicle entering — bar fills green toward PARK_CONFIRM_S
            frac = min(1.0, (now_t - spot.first_seen_t) / PARK_CONFIRM_S)
            cv2.rectangle(out, (bar_x1, bar_y),
                          (bar_x1 + int(bar_bw * frac), bar_y + 6), C_BAR_IN, -1)
        elif occupied:
            absence = now_t - spot.last_seen_t if spot.last_seen_t else 0.0
            if absence <= 1.0:
                # Vehicle actively present — solid full bar
                cv2.rectangle(out, (bar_x1, bar_y), (bar_x2, bar_y + 6), C_BAR_FULL, -1)
            else:
                # Vehicle gone — bar fills blue toward PARK_RELEASE_S
                frac = min(1.0, absence / PARK_RELEASE_S)
                cv2.rectangle(out, (bar_x1, bar_y),
                              (bar_x1 + int(bar_bw * frac), bar_y + 6), C_BAR_OUT, -1)

        y += 50
    sep_line(y); y += 14

    hints = [
        ("Q / ESC", "quit"),
        ("F",       "full reset"),
        ("P",       "add parking spot"),
        ("X",       "set security zone for last spot"),
        ("BkSp",    "remove last spot"),
        ("V",       "toggle night mode"),
        ("H",       "toggle heatmap"),
        ("S",       "send snapshot via Telegram"),
        ("C",       "recalibrate stabilizer"),
    ]
    sep_line(y); y += 14
    cv2.putText(out, "KEYS", (14, y), FONT, 0.5, C_DIM, 1, cv2.LINE_AA)
    y += 18
    for key, desc in hints:
        if y + 16 > panel_h - 4:
            break
        cv2.putText(out, key, (14, y), FONT, 0.46, C_HEAD, 1, cv2.LINE_AA)
        cv2.putText(out, desc, (84, y), FONT, 0.46, C_DIM, 1, cv2.LINE_AA)
        y += 17

    return out


# ─── Camera ───────────────────────────────────────────────────────────────────

def start_camera(cam_w: int, cam_h: int) -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (cam_w, cam_h), "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    try:
        cam.set_controls({"AfMode": 2, "AfRange": 2})
    except Exception:
        pass
    cam.start()
    logging.info("Camera started (%dx%d)", cam_w, cam_h)
    return cam

NIGHT_EXPOSURE_US = 700_000

def set_night_mode(cam: "Picamera2", enabled: bool, cam_w: int, cam_h: int):
    cam.stop()
    cfg = cam.create_video_configuration(
        main={"size": (cam_w, cam_h), "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    cam.start()
    if enabled:
        try:
            cam.set_controls({
                "AeEnable":     False,
                "ExposureTime": NIGHT_EXPOSURE_US,
                "AnalogueGain": 8.0,
                "AwbEnable":    False,
                "ColourGains":  (1.8, 1.4),
            })
        except Exception as exc:
            logging.warning("Night controls failed: %s", exc)
    else:
        try:
            cam.set_controls({
                "AeEnable":  True,
                "AwbEnable": True,
                "AfMode": 2, "AfRange": 2,
            })
        except Exception as exc:
            logging.warning("Day controls failed: %s", exc)
    time.sleep(0.3)
    logging.info("Night mode %s", "ON" if enabled else "OFF")


# ─── Web dashboard ────────────────────────────────────────────────────────────

WEB_PORT = 8080

def start_web_server(store: FrameStore, spots_ref: list):
    """Serve a live MJPEG dashboard on http://<pi-ip>:8080"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # suppress per-request console noise

        def do_GET(self):
            if self.path == "/frame.jpg":
                frame, _ = store.get()
                if frame is None:
                    self.send_response(503); self.end_headers(); return
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    self.send_response(503); self.end_headers(); return
                data = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            elif self.path == "/status.json":
                payload = json.dumps({
                    "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                    "spots": [{"name": sp.name, "state": sp.state,
                               "occupied_s": int(time.time() - sp.occupied_since)
                                             if sp.occupied_since else 0}
                              for sp in spots_ref]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            elif self.path == "/":
                html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Parking Monitor</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; margin:0; padding:16px; }
    h1   { color:#3cd2ff; margin:0 0 12px; font-size:1.3em; }
    img  { max-width:100%; border:2px solid #333; border-radius:6px; display:block; }
    #spots { margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }
    .spot { padding:8px 16px; border-radius:6px; font-weight:bold; font-size:1em; }
    .FREE     { background:#1a4a1a; color:#4de04d; border:1px solid #4de04d; }
    .OCCUPIED { background:#3a0a0a; color:#ff5555; border:1px solid #ff5555; }
    #ts { color:#666; font-size:0.8em; margin-top:10px; }
  </style>
</head>
<body>
  <h1>Parking Monitor</h1>
  <img id="feed" src="/frame.jpg">
  <div id="spots"></div>
  <div id="ts"></div>
  <script>
    function refresh() {
      document.getElementById('feed').src = '/frame.jpg?t=' + Date.now();
      fetch('/status.json').then(r=>r.json()).then(d=>{
        document.getElementById('ts').textContent = d.time;
        document.getElementById('spots').innerHTML = d.spots.map(s => {
          let dur = s.state==='OCCUPIED' ? ' — ' + Math.floor(s.occupied_s/60) + 'm ' + (s.occupied_s%60) + 's' : '';
          return '<div class="spot ' + s.state + '">' + s.name + ': ' + s.state + dur + '</div>';
        }).join('');
      });
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>""".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            else:
                self.send_response(404); self.end_headers()

    server = HTTPServer(("0.0.0.0", WEB_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="web").start()
    logging.info("Web dashboard: http://<this-pi-ip>:%d", WEB_PORT)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Delete saved config and start fresh")
    args = ap.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / f"app_{datetime.now():%Y%m%d}.log"),
        ],
    )

    def _log_exception(exc_type, exc_value, exc_tb):
        logging.critical("Unhandled exception — script will exit",
                         exc_info=(exc_type, exc_value, exc_tb))
    sys.excepthook = _log_exception

    if args.reset and CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
        logging.info("Config cleared — starting fresh")

    cfg = load_cfg()
    cam_w, cam_h = cfg["cam_w"], cfg["cam_h"]
    dw,    dh    = cfg["disp_w"], cfg["disp_h"]

    cam = start_camera(cam_w, cam_h)
    time.sleep(0.5)

    spots: List[ParkingSpot] = [ParkingSpot.from_cfg(d)
                                 for d in cfg.get("parking_spots", [])]
    logging.info("Loaded %d parking spot(s)", len(spots))

    logging.info("Loading HEF: %s", HEF_PATH)
    hef     = HEF(HEF_PATH)
    in_name = hef.get_input_vstream_infos()[0].name

    with VDevice() as dev:
        cp  = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        ng  = dev.configure(hef, cp)[0]
        ivp = InputVStreamParams.make(ng, quantized=False, format_type=FormatType.UINT8)
        ovp = OutputVStreamParams.make(ng, quantized=False, format_type=FormatType.FLOAT32)
        ng_params = ng.create_params()
        logging.info("Hailo YOLOv8m ready")

        tracker  = Tracker()
        park_log = ParkingLogger()

        tg_token   = cfg.get("telegram_token",   "")
        tg_chat_id = cfg.get("telegram_chat_id", "")
        logging.info("Telegram %s", "enabled" if tg_token and tg_chat_id else "disabled")

        import socket
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = "<?>"
        logging.info("Web dashboard will be at http://%s:%d", local_ip, WEB_PORT)

        frame_store = FrameStore()
        start_telegram_poller(tg_token, tg_chat_id, frame_store)
        start_web_server(frame_store, spots)

        running            = True
        night_mode         = False
        auto_switch_t: float = 0.0
        luma_ema: float    = -1.0   # -1 = uninitialized; resets when camera mode changes
        reset_banner_until: float = 0.0
        frame_num          = 0
        heatmap            = np.zeros((cam_h, cam_w), dtype=np.float32)
        show_heatmap       = False
        fps_buf            = deque(maxlen=30)
        spot_seq           = len(spots)

        stabilizer    = CameraStabilizer()
        stab_ref_set  = False
        stab_update_t = 0.0

        WIN       = "Parking Monitor"
        WIN_STATS = "Parking Stats"

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, dw, dh)
        cv2.moveWindow(WIN, 0, WIN_Y)
        cv2.namedWindow(WIN_STATS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_STATS, STATS_W, dh)
        cv2.moveWindow(WIN_STATS, dw + 4, WIN_Y)

        def on_stop(*_):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT,  on_stop)
        signal.signal(signal.SIGTERM, on_stop)

        with InferVStreams(ng, ivp, ovp) as pipeline:
            with ng.activate(ng_params):
                logging.info("Live. Q=quit F=reset P=+spot X=+zone BkSp=-spot V=night H=heatmap S=snap")

                while running:
                    t0    = time.perf_counter()
                    now_t = time.time()

                    try:
                        frame = cam.capture_array()
                    except Exception as e:
                        logging.warning("Camera error: %s — restarting", e)
                        try: cam.stop(); cam.close()
                        except Exception: pass
                        time.sleep(2)
                        cam = start_camera(cam_w, cam_h)
                        continue

                    H, W = frame.shape[:2]

                    # ── Luma: grayscale mean of the centre 50 % of the frame.
                    # Centre ROI avoids sky (top) and edge headlights skewing the reading.
                    # EMA smoothing (α=0.03 ≈ 33-frame window) prevents single bright/dark
                    # frames from resetting the switch timer.
                    roi = frame[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]
                    raw_luma = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)))
                    if luma_ema < 0:
                        luma_ema = raw_luma          # bootstrap on first frame
                    else:
                        luma_ema = luma_ema * (1 - LUMA_EMA_ALPHA) + raw_luma * LUMA_EMA_ALPHA

                    now_mono = time.monotonic()

                    # Set stabilizer reference on first frame
                    if not stab_ref_set:
                        stabilizer.set_reference(frame)
                        stab_ref_set  = True
                        stab_update_t = now_mono

                    # ── Auto night mode switching
                    # Timer only cancels when EMA crosses back past a WIDER band (hysteresis),
                    # so brief luma oscillations at dusk/dawn no longer keep resetting it.
                    if not night_mode:
                        if luma_ema < LUMA_TO_NIGHT:
                            if auto_switch_t == 0.0:
                                auto_switch_t = now_mono
                                logging.info("Night timer started (luma_ema %.1f)", luma_ema)
                            elif now_mono - auto_switch_t >= AUTO_NIGHT_DELAY_S:
                                night_mode    = True
                                auto_switch_t = 0.0
                                luma_ema      = -1.0   # reset EMA — different camera exposure
                                set_night_mode(cam, True, cam_w, cam_h)
                                logging.info("Auto night ON  (luma_ema %.1f)", luma_ema)
                        elif luma_ema > LUMA_TO_NIGHT_CANCEL:   # hysteresis band
                            if auto_switch_t != 0.0:
                                logging.info("Night timer cancelled (luma_ema %.1f)", luma_ema)
                            auto_switch_t = 0.0
                    else:
                        if luma_ema > LUMA_TO_DAY_NIGHT_EXP:
                            if auto_switch_t == 0.0:
                                auto_switch_t = now_mono
                                logging.info("Day timer started (luma_ema %.1f)", luma_ema)
                            elif now_mono - auto_switch_t >= AUTO_NIGHT_DELAY_S:
                                night_mode    = False
                                auto_switch_t = 0.0
                                luma_ema      = -1.0   # reset EMA — different camera exposure
                                set_night_mode(cam, False, cam_w, cam_h)
                                logging.info("Auto night OFF (luma_ema %.1f)", luma_ema)
                        elif luma_ema < LUMA_TO_DAY_CANCEL:     # hysteresis band
                            if auto_switch_t != 0.0:
                                logging.info("Day timer cancelled (luma_ema %.1f)", luma_ema)
                            auto_switch_t = 0.0

                    # Camera stabilization — update every 3 s, warp spot polygons
                    if stab_ref_set and now_mono - stab_update_t >= 3.0:
                        stab_update_t = now_mono
                        if stabilizer.update(frame):
                            for sp in spots:
                                if sp.canonical_poly is not None:
                                    sp.poly = stabilizer.transform_poly(sp.canonical_poly)
                                if sp.canonical_alert_zone is not None:
                                    sp.alert_zone = stabilizer.transform_poly(
                                        sp.canonical_alert_zone)

                    # Infer
                    lb, s, px, py = letterbox(frame)
                    raw = pipeline.infer({in_name: lb[np.newaxis]})
                    classes_list = raw[next(iter(raw))][0]

                    conf_thresh = CONF_NIGHT if night_mode else CONF_DAY
                    dets = []
                    for cls_id, det_arr in enumerate(classes_list):
                        if cls_id not in TRACKED or len(det_arr) == 0:
                            continue
                        for d in det_arr:
                            y1n, x1n, y2n, x2n, conf = d
                            if conf < conf_thresh:
                                continue
                            x1, y1, x2, y2 = unbox(y1n, x1n, y2n, x2n, s, px, py, W, H)
                            if (x2 > x1 and y2 > y1
                                    and (x2 - x1) * (y2 - y1) >= MIN_BBOX_AREA):
                                dets.append((cls_id, x1, y1, x2, y2, float(conf)))

                    tracks = tracker.update(dets)

                    # Parking occupancy (time-based — consistent across day and night fps)
                    pending_snaps: List[str] = []
                    for spot in spots:
                        occ = vehicle_in_spot(spot, tracks)
                        if occ:
                            if spot.first_seen_t is None:
                                spot.first_seen_t = now_t
                            spot.last_seen_t = now_t
                        else:
                            spot.first_seen_t = None

                        if spot.state == "FREE":
                            if (spot.first_seen_t is not None
                                    and now_t - spot.first_seen_t >= PARK_CONFIRM_S):
                                spot.state = "OCCUPIED"
                                spot.occupied_since = now_t
                                spot.alert_until    = time.monotonic() + 5.0
                                park_log.log(spot.name, "OCCUPIED")
                                if now_t - spot.last_tg_occupied >= TG_STATE_COOLDOWN_S:
                                    ts  = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
                                    msg = f"🔴 {spot.name} OCCUPIED\nTime: {ts}"
                                    send_telegram(tg_token, tg_chat_id, msg)
                                    pending_snaps.append(msg)
                                    spot.last_tg_occupied = now_t
                        elif spot.state == "OCCUPIED":
                            absence = now_t - spot.last_seen_t if spot.last_seen_t else PARK_RELEASE_S
                            if absence >= PARK_RELEASE_S:
                                dur = now_t - (spot.occupied_since or now_t)
                                spot.state = "FREE"
                                spot.occupied_since = None
                                spot.last_seen_t    = None
                                park_log.log(spot.name, "FREE", dur)
                                if now_t - spot.last_tg_free >= TG_STATE_COOLDOWN_S:
                                    ts  = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
                                    msg = (f"🟢 {spot.name} FREE\n"
                                           f"Was occupied for {int(dur//60)}m {int(dur%60)}s\n"
                                           f"Time: {ts}")
                                    send_telegram(tg_token, tg_chat_id, msg)
                                    pending_snaps.append(msg)
                                    spot.last_tg_free = now_t

                    # FPS
                    fps_buf.append(time.perf_counter() - t0)
                    fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

                    # Heatmap accumulation
                    for t in tracks.values():
                        if t.miss > 0:
                            continue
                        x1, y1, x2, y2 = t.bbox
                        heatmap[max(0, y1):min(cam_h, y2),
                                max(0, x1):min(cam_w, x2)] += 1.0
                    heatmap *= HEATMAP_DECAY

                    # Render
                    display   = render_frame(frame, tracks, fps, dw, dh, spots)
                    stats_img = render_stats(fps, spots, now_t, STATS_W, dh)

                    # Heatmap overlay
                    if show_heatmap:
                        hm_max = heatmap.max()
                        if hm_max > 1.0:
                            hm_small = cv2.resize(heatmap, (dw, dh))
                            hm_u8    = np.clip(hm_small / hm_max * 255, 0, 255).astype(np.uint8)
                            hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
                            mask     = (hm_u8 > 12)[:, :, np.newaxis].astype(np.float32)
                            display  = (display.astype(np.float32) * (1 - mask * 0.6)
                                        + hm_color.astype(np.float32) * mask * 0.6
                                        ).astype(np.uint8)
                        cv2.putText(display, "HEATMAP ON",
                                    (10, dh - 12), FONT, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

                    # Parking badge
                    if spots:
                        n_free = sum(1 for sp in spots if sp.state == "FREE")
                        n_occ  = len(spots) - n_free
                        badge  = f"P  {n_free} FREE  {n_occ} OCC"
                        col    = (80, 230, 80) if n_free > 0 else (80, 80, 255)
                        (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.65, 2)
                        bx = dw - bw - 14
                        by = dh - 14
                        cv2.rectangle(display, (bx - 8, by - bh - 8), (dw - 6, by + 6),
                                      (20, 20, 20), -1)
                        cv2.putText(display, badge, (bx, by), FONT, 0.65, col, 2, cv2.LINE_AA)

                    # Start new alert recordings (pedestrian loitering ≥ 4 s)
                    for spot in spots:
                        if spot.alert_zone is not None and threat_in_zone(spot, tracks):
                            if spot.zone_entry_time is None:
                                spot.zone_entry_time = now_t
                        else:
                            spot.zone_entry_time = None

                    for spot in spots:
                        loitering = (spot.zone_entry_time is not None
                                     and now_t - spot.zone_entry_time >= LOITER_THRESHOLD_S)
                        if (spot.state == "OCCUPIED"
                                and spot.alert_zone is not None
                                and not spot.recording
                                and now_t > spot.alert_cooldown_until
                                and loitering):
                            ts_rec   = datetime.now().strftime("%Y%m%d_%H%M%S")
                            rec_path = Path(
                                f"/tmp/alert_{spot.name.replace(' ', '_')}_{ts_rec}.mp4")
                            rec_fps  = float(max(10, min(30, int(fps) or 20)))
                            writer   = cv2.VideoWriter(
                                str(rec_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                rec_fps,
                                (display.shape[1], display.shape[0]),
                            )
                            spot.recording       = True
                            spot.record_writer   = writer
                            spot.record_path     = rec_path
                            spot.record_end      = now_t + 10.0
                            spot.zone_entry_time = None
                            logging.info("Alert recording started: %s", rec_path)
                            send_telegram(tg_token, tg_chat_id,
                                          f"🎥 {spot.name}: pedestrian near occupied spot — recording 10 s…")

                    # REC badge
                    if any(sp.recording for sp in spots):
                        blink   = int(time.monotonic() * 2) % 2 == 0
                        rec_col = (0, 0, 220) if blink else (0, 0, 150)
                        cv2.circle(display, (18, 18), 10, rec_col, -1, cv2.LINE_AA)
                        cv2.putText(display, "REC", (32, 24), FONT, 0.65, rec_col, 2)

                    # Night / luma badge
                    if night_mode:
                        cv2.rectangle(display, (8, 34), (160, 58), (30, 20, 0), -1)
                        cv2.putText(display, f"NIGHT  L:{luma_ema:.0f}", (12, 52),
                                    FONT, 0.52, (0, 180, 255), 1, cv2.LINE_AA)
                    elif auto_switch_t > 0.0:
                        remaining = max(0.0, AUTO_NIGHT_DELAY_S - (now_mono - auto_switch_t))
                        cv2.putText(display, f"->NIGHT {remaining:.0f}s  L:{luma_ema:.0f}",
                                    (8, 52), FONT, 0.50, (0, 120, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(display, f"DAY  L:{luma_ema:.0f}",
                                    (8, 52), FONT, 0.50, (160, 200, 100), 1, cv2.LINE_AA)

                    # Stabilizer status badge (bottom-left, above heatmap label)
                    if stab_ref_set:
                        s_col = ((0, 200,  80) if stabilizer.status == CameraStabilizer.STATUS_OK
                                 else (0, 165, 255) if stabilizer.status == CameraStabilizer.STATUS_DRIFT
                                 else (0,  60, 220))
                        cv2.putText(display, stabilizer.status,
                                    (8, dh - 30), FONT, 0.46, s_col, 1, cv2.LINE_AA)

                    # Full-reset flash
                    if time.monotonic() < reset_banner_until:
                        bh2, bw2 = display.shape[:2]
                        cv2.rectangle(display, (0, bh2//2 - 36), (bw2, bh2//2 + 36), (0, 0, 0), -1)
                        cv2.putText(display, "FULL RESET", (bw2//2 - 130, bh2//2 + 14),
                                    FONT, 1.6, (0, 255, 128), 3, cv2.LINE_AA)

                    cv2.imshow(WIN, display)
                    cv2.imshow(WIN_STATS, stats_img)

                    # Write frames + finalize recordings
                    for spot in spots:
                        if not spot.recording:
                            continue
                        spot.record_writer.write(display)
                        if time.time() >= spot.record_end:
                            spot.record_writer.release()
                            path = spot.record_path
                            spot.recording            = False
                            spot.record_writer        = None
                            spot.record_path          = None
                            spot.alert_cooldown_until = time.time() + 60.0
                            caption = (
                                f"🚨 Alert: {spot.name} — pedestrian near occupied spot\n"
                                f"Time: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
                            )
                            send_telegram_video(tg_token, tg_chat_id, path, caption)

                    # Update frame store for /snap
                    snap_caption = f"{datetime.now().strftime('%H:%M:%S %d/%m/%Y')}  FPS:{fps:.1f}"
                    if spots:
                        snap_caption += "\n" + "  ".join(
                            f"{sp.name}:{sp.state}" for sp in spots)
                    frame_store.put(display, snap_caption)

                    # Auto-snap on state change
                    for caption in pending_snaps:
                        send_telegram_photo(tg_token, tg_chat_id, display, caption)

                    # Key handling
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('q'), 27):
                        running = False

                    elif key == ord('h'):
                        show_heatmap = not show_heatmap
                        logging.info("Heatmap %s", "ON" if show_heatmap else "OFF")

                    elif key == ord('v'):
                        night_mode    = not night_mode
                        auto_switch_t = 0.0
                        luma_ema      = -1.0   # reset EMA — exposure changes
                        set_night_mode(cam, night_mode, cam_w, cam_h)

                    elif key == ord('f'):
                        for sp in spots:
                            if sp.recording and sp.record_writer:
                                sp.record_writer.release()
                        spots.clear()
                        tracker  = Tracker()
                        spot_seq = 0
                        cfg["parking_spots"] = []
                        save_cfg(cfg)
                        if night_mode:
                            night_mode = False
                            set_night_mode(cam, False, cam_w, cam_h)
                        auto_switch_t      = 0.0
                        heatmap[:] = 0.0
                        show_heatmap       = False
                        reset_banner_until = time.monotonic() + 2.0
                        logging.info("Full reset")

                    elif key == ord('p'):
                        spot_seq += 1
                        default_name = f"Spot {spot_seq}"
                        name = pick_spot_name(default_name, dw, dh)
                        cur  = cam.capture_array()
                        poly = pick_parking_spot(cur, dw, dh, name)
                        cv2.waitKey(1)
                        if poly is not None:
                            canonical = stabilizer.inverse_transform_pts(poly)
                            spots.append(ParkingSpot(name=name, poly=poly.copy(),
                                                     canonical_poly=canonical))
                            cfg["parking_spots"] = [sp.to_cfg() for sp in spots]
                            save_cfg(cfg)
                            logging.info("Added parking spot '%s'", name)

                    elif key == ord('x') and spots:
                        try:
                            cur  = cam.capture_array()
                            zone = pick_parking_spot(
                                cur, dw, dh, f"Alert zone - {spots[-1].name}")
                            if zone is not None:
                                spots[-1].alert_zone           = zone.copy()
                                spots[-1].canonical_alert_zone = stabilizer.inverse_transform_pts(zone)
                                cfg["parking_spots"] = [sp.to_cfg() for sp in spots]
                                save_cfg(cfg)
                                logging.info("Alert zone set for '%s'", spots[-1].name)
                        except Exception as exc:
                            logging.error("Alert zone definition failed: %s", exc)
                        cv2.waitKey(1)

                    elif key in (8, 127) and spots:
                        removed = spots.pop()
                        cfg["parking_spots"] = [sp.to_cfg() for sp in spots]
                        save_cfg(cfg)
                        logging.info("Removed parking spot '%s'", removed.name)

                    elif key == ord('s'):
                        ts          = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        spot_lines  = [f"  {sp.name}: {sp.state}" for sp in spots]
                        caption     = f"📸 Snapshot — {ts}\n"
                        if spot_lines:
                            caption += "Parking:\n" + "\n".join(spot_lines)
                        send_telegram_photo(tg_token, tg_chat_id, display, caption)
                        logging.info("Snapshot sent to Telegram")

                    elif key == ord('c'):
                        for sp in spots:
                            sp.canonical_poly = sp.poly.copy()
                            sp.canonical_alert_zone = (sp.alert_zone.copy()
                                                       if sp.alert_zone is not None else None)
                        stabilizer.set_reference(frame)
                        stab_update_t = now_mono
                        logging.info("Stabilizer recalibrated — new reference frame set")

                    del frame
                    frame_num += 1

        # Exit — log any still-occupied spots
        for spot in spots:
            if spot.state == "OCCUPIED" and spot.occupied_since:
                park_log.log(spot.name, "FREE", time.time() - spot.occupied_since)

        logging.info("─── Session summary ────────────────────────────")
        for spot in spots:
            logging.info("  Parking %-12s  state=%s", spot.name, spot.state)
        logging.info("────────────────────────────────────────────────")

        cam.stop(); cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
