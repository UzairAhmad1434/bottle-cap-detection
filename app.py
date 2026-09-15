"""
=====================================================================
  BOTTLE CAP DETECTION & LINE-CROSSING COUNTER
  ---------------------------------------------------------------
  - Detects "Capped bottle" (GREEN box) and "UnCapped bottle" (RED box)
  - Draws a vertical counting LINE on screen
  - A bottle is counted ONLY when its center crosses that line
    (not just because it's visible) -> accurate real-world counting
  - Bounding boxes are SMOOTHED (EMA) per tracked object -> no jitter
  - Clean, glowing, modern dashboard UI (no FPS clutter)
  - Saves the final annotated video to disk

  Just edit the CONFIG section below and run:
      python bottle_cap_detector.py
=====================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ============================== CONFIG ==============================
MODEL_PATH   = r"U:\bottle\best.pt"
VIDEO_PATH   = r"U:\bottle\bottle.mp4"
OUTPUT_PATH  = r"U:\bottle\bottle_output.mp4"

CONF_THRESH  = 0.4
IMG_SIZE     = 640
SHOW_LIVE    = True

# Position of the vertical counting line as a FRACTION of frame width
# 0.5 = exact middle. Change to 0.6, 0.4 etc. depending on your video.
LINE_X_RATIO = 0.5

# Smoothing factor for bounding boxes (0 = no smoothing, 1 = frozen).
# 0.55-0.7 gives a nice stable-but-responsive box.
SMOOTHING_ALPHA = 0.6

CAPPED_KEYWORDS   = ["capped"]
UNCAPPED_KEYWORDS = ["uncapped"]

# Colors (BGR)
COLOR_GREEN      = (80, 220, 100)
COLOR_RED        = (60, 60, 235)
COLOR_LINE       = (0, 225, 255)
COLOR_PANEL_BG   = (18, 18, 18)
COLOR_PANEL_EDGE = (0, 200, 255)
COLOR_TEXT       = (240, 240, 240)
# ======================================================================


def classify_label(class_name: str) -> str:
    name = class_name.lower()
    if any(k in name for k in CAPPED_KEYWORDS) and "uncapped" not in name:
        return "capped"
    if any(k in name for k in UNCAPPED_KEYWORDS):
        return "uncapped"
    return "unknown"


def draw_glow_line(frame, x, height, pulse_phase):
    """Vertical counting line with a soft animated glow."""
    glow_strength = 0.5 + 0.5 * np.sin(pulse_phase)  # 0..1 pulsing
    overlay = frame.copy()
    glow_thickness = int(10 + 6 * glow_strength)
    cv2.line(overlay, (x, 0), (x, height), COLOR_LINE, glow_thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.line(frame, (x, 0), (x, height), COLOR_LINE, 2, cv2.LINE_AA)

    # small triangular markers at top and bottom of the line
    for y in (18, height - 18):
        pts = np.array([[x - 8, y - 8], [x + 8, y - 8], [x, y + 6]], np.int32)
        cv2.fillPoly(frame, [pts], COLOR_LINE, cv2.LINE_AA)


def draw_fancy_box(frame, x1, y1, x2, y2, color, label, conf, track_id):
    thickness = 3
    corner_len = max(10, min(25, (x2 - x1) // 4, (y2 - y1) // 4))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness, cv2.LINE_AA)

    tag = f"#{track_id} {label.upper()} {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    tag_y1 = max(0, y1 - th - 12)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, y1), color, -1, cv2.LINE_AA)
    cv2.putText(frame, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)


def draw_rounded_panel(frame, x, y, w, h, radius=15, bg=COLOR_PANEL_BG, edge=COLOR_PANEL_EDGE):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), bg, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), bg, -1)
    for cx, cy in [(x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, bg, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), edge, 2, cv2.LINE_AA)


def draw_dashboard(frame, capped_count, uncapped_count):
    panel_w, panel_h = 320, 130
    draw_rounded_panel(frame, 15, 15, panel_w, panel_h)

    cv2.putText(frame, "BOTTLE CAP INSPECTOR", (30, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.circle(frame, (40, 78), 9, COLOR_GREEN, -1)
    cv2.putText(frame, f"Capped:   {capped_count}", (58, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2, cv2.LINE_AA)

    cv2.circle(frame, (40, 108), 9, COLOR_RED, -1)
    cv2.putText(frame, f"Uncapped: {uncapped_count}", (58, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2, cv2.LINE_AA)

    total = capped_count + uncapped_count
    cv2.putText(frame, f"Total Crossed: {total}", (185, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


class Smoother:
    """Keeps an exponentially-smoothed bounding box per track ID
    so boxes stop jittering frame-to-frame."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = {}  # track_id -> np.array([x1,y1,x2,y2])

    def update(self, track_id, box):
        box = np.array(box, dtype=float)
        if track_id not in self.state:
            self.state[track_id] = box
        else:
            self.state[track_id] = (self.alpha * self.state[track_id] +
                                     (1 - self.alpha) * box)
        return self.state[track_id].astype(int)

    def cleanup(self, active_ids):
        for tid in list(self.state.keys()):
            if tid not in active_ids:
                del self.state[tid]


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    line_x = int(width * LINE_X_RATIO)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))

    smoother = Smoother(alpha=SMOOTHING_ALPHA)

    # last known side of the line for each track_id: "left" / "right"
    last_side = {}
    # track IDs already counted (crossed once) so we never double count
    already_counted = set()

    capped_count = 0
    uncapped_count = 0

    frame_idx = 0
    pulse_phase = 0.0

    print("Processing video... press 'q' in the preview window to stop early.")

    results_gen = model.track(
        source=VIDEO_PATH,
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    for result in results_gen:
        frame = result.orig_img.copy()
        frame_idx += 1
        pulse_phase += 0.15

        active_ids = set()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy  = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            ids   = boxes.id.cpu().numpy().astype(int)

            for box, conf, cls_id, track_id in zip(xyxy, confs, clses, ids):
                active_ids.add(track_id)

                # Smooth the box for stability
                x1, y1, x2, y2 = smoother.update(track_id, box)
                cx = (x1 + x2) // 2  # center x, used for line-crossing check

                class_name = model.names[cls_id]
                category = classify_label(class_name)
                color = COLOR_GREEN if category == "capped" else \
                        COLOR_RED if category == "uncapped" else (200, 200, 0)

                # ---- Line-crossing counting logic ----
                side = "left" if cx < line_x else "right"
                if track_id in last_side and last_side[track_id] != side:
                    if track_id not in already_counted:
                        already_counted.add(track_id)
                        if category == "capped":
                            capped_count += 1
                        elif category == "uncapped":
                            uncapped_count += 1
                last_side[track_id] = side

                draw_fancy_box(frame, x1, y1, x2, y2, color, category, conf, track_id)

        smoother.cleanup(active_ids)

        draw_glow_line(frame, line_x, height, pulse_phase)
        draw_dashboard(frame, capped_count, uncapped_count)

        writer.write(frame)

        if SHOW_LIVE:
            cv2.imshow("Bottle Cap Inspector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    writer.release()
    cv2.destroyAllWindows()

    print("\n================ FINAL REPORT ================")
    print(f"Total frames processed : {frame_idx}")
    print(f"CAPPED bottles crossed  : {capped_count}")
    print(f"UNCAPPED bottles crossed: {uncapped_count}")
    print(f"Total bottles crossed   : {capped_count + uncapped_count}")
    print(f"Output saved to         : {OUTPUT_PATH}")
    print("================================================")


if __name__ == "__main__":
    main()