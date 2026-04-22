"""
CAULIFLOWER INFERENCE — Single Class
====================================
Uses ONLY the trained YOLOv8-seg model output for 1 class (cauliflower).
  class 0 = cauliflower

What this script gives you:
  ✅ Coloured segmentation mask per plant (drawn ON the image)
  ✅ Centroid (cx, cy) marked directly ON each plant
  ✅ Coordinates printed as text ON the plant
  ✅ Z coordinate is fixed (you set it below)
  ✅ One JSON file with clean robot-ready coordinates
  ✅ False positive removal: tiny blob filter

USAGE:
    # Single image — shows window + saves result
    python detect_cauliflower.py --model best.pt --source image.jpg
    
    # Folder of images
    python detect_cauliflower.py --model best.pt --source ./test_images/
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# ─── SET THESE ─────────────────────────────────────────────────────────────────
Z_MM         = 500     # ← your camera height above ground in mm
CONF         = 0.15    # ← confidence threshold (0.15 = sensitive, 0.40 = strict)
IMGSZ        = 640     # ← MUST match your training imgsz (you trained at 640)
IOU          = 0.45    # NMS IoU threshold

# False positive filters
MIN_AREA_PX  = 400     # ignore masks smaller than 400px² (drip tubes, stones, noise)

# Colours
COL_CROP     = (0, 220, 0)     # green  — cauliflower
COL_TEXT     = (255, 255, 255) # white  — text
# ───────────────────────────────────────────────────────────────────────────────

def run(model, frame):
    """YOLOv8 inference → list of detections."""
    h, w = frame.shape[:2]
    # task='segment' ensures ONNX uses the seg head if applicable
    res   = model(frame, conf=CONF, iou=IOU, imgsz=IMGSZ,
                  task="segment", verbose=False)[0]

    dets = []
    if res.masks is None or res.boxes is None:
        return dets

    masks  = res.masks.data.cpu().numpy()
    boxes  = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs  = res.boxes.conf.cpu().numpy()

    for mask_raw, box, cls_id, conf in zip(masks, boxes, cls_ids, confs):

        # Full-resolution binary mask
        m    = cv2.resize(mask_raw, (w, h))
        m    = (m > 0.5).astype(np.uint8) * 255

        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue

        # Centroid from mask pixels (accurate centre of plant, not bbox centre)
        cx   = int(xs.mean())
        cy   = int(ys.mean())
        area = int(len(xs))

        x1, y1, x2, y2 = map(int, box)

        dets.append({
            "cls":      int(cls_id),          # 0=cauliflower
            "name":     "cauliflower",
            "conf":     round(float(conf), 3),
            "cx":       cx,
            "cy":       cy,
            "z":        Z_MM,
            "cx_norm":  round(cx / w, 4),
            "cy_norm":  round(cy / h, 4),
            "bbox":     [x1, y1, x2, y2],
            "area":     area,
            "_m":       m,                    # mask (stripped before JSON)
        })

    return dets


def filter_fp(dets):
    """
    Filter 1: Remove tiny masks (noise, stones, etc)
    """
    crops = [d for d in dets if d["area"] >= MIN_AREA_PX]
    return crops


def draw(frame, crops):
    """
    Draw segmentation masks, centroids, and coordinate labels
    directly ON the image.
    """
    out     = frame.copy()
    overlay = out.copy()
    h, w    = out.shape[:2]

    # ── Segmentation fills ────────────────────────────────────────────────────
    for c in crops:
        overlay[c["_m"] > 0] = np.clip(
            overlay[c["_m"] > 0] * 0.3 + np.array([20, 200, 20]) * 0.7, 0, 255
        ).astype(np.uint8)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # ── Draw segmentation CONTOURS (outline per plant) ────────────────────────
    for c in crops:
        cnts, _ = cv2.findContours(c["_m"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_CROP, 2)

    # ── Crop centroid + coordinates ON the plant ──────────────────────────────
    for c in crops:
        cx, cy = c["cx"], c["cy"]

        # Filled circle at centroid
        cv2.circle(out, (cx, cy), 10, COL_CROP, -1)
        cv2.circle(out, (cx, cy), 10, (0, 0, 0), 1)  # thin black ring

        # Coordinate text directly on the plant
        label = f"CAULIFLOWER  x={cx}  y={cy}  z={Z_MM}mm"
        conf_label = f"conf={c['conf']:.2f}"

        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (cx - 4, cy + 12), (cx + tw + 8, cy + 14 + th + 4),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_CROP, 1)
        cv2.putText(out, conf_label, (cx, cy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

    # ── Summary banner ────────────────────────────────────────────────────────
    action = f"{len(crops)} cauliflowers detected"

    cv2.rectangle(out, (0, 0), (w, 50), (10, 10, 10), -1)
    cv2.putText(out, f"cauliflowers={len(crops)}  |  {action}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # ── Legend (bottom) ───────────────────────────────────────────────────────
    lx, ly = 10, h - 30
    cv2.rectangle(out, (lx, ly), (lx + 18, ly + 14), COL_CROP, -1)
    cv2.putText(out, "cauliflower (detected)",
                (lx + 22, ly + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_CROP, 1)

    return out


def process_one(model, img_path, out_dir, show):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read: {img_path}")
        return None

    h, w = frame.shape[:2]

    print(f"\n{'═' * 65}")
    print(f"  {img_path.name}  ({w}×{h})")
    print(f"  conf={CONF}  imgsz={IMGSZ}  z={Z_MM}mm")

    # ── Detect ────────────────────────────────────────────────────────────────
    dets = run(model, frame)
    print(f"  Model output: {len(dets)} detections (before filter)")

    # ── Filter ────────────────────────────────────────────────────────────────
    crops = filter_fp(dets)
    print(f"  After filter: {len(crops)} cauliflowers")

    if not crops:
        print(f"\n  ⚠️  Nothing detected at conf={CONF}")
        print(f"  → Try: python detect_cauliflower.py --model best.pt --source {img_path.name} --conf 0.10")

    # ── Draw ─────────────────────────────────────────────────────────────────
    annotated = draw(frame, crops)

    # ── Print per-plant results ───────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  CAULIFLOWERS DETECTED:")
    for i, c in enumerate(crops):
        print(f"    #{i}  centroid=({c['cx']}, {c['cy']})  z={Z_MM}mm"
              f"  conf={c['conf']}  area={c['area']}px²")

    # ── Output logic ──────────────────────────────────────────────────────────
    robot = {
        "n_cauliflowers": len(crops),
        "cauliflowers": [{"x": c["cx"], "y": c["cy"], "z": Z_MM,
                       "conf": c["conf"]} for c in crops],
    }

    print(f"  {'─'*55}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = img_path.stem
    out_img  = out_dir / f"{stem}_result.jpg"
    out_json = out_dir / f"{stem}_result.json"

    cv2.imwrite(str(out_img), annotated)

    def _clean(lst):
        return [{k: v for k, v in d.items() if k != "_m"} for d in lst]

    result = {
        "image":      img_path.name,
        "size":       {"w": w, "h": h},
        "z_mm":       Z_MM,
        "conf_used":  CONF,
        "cauliflowers": _clean(crops),
        "robot":      robot,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → {out_img.name}")
    print(f"  Saved → {out_json.name}")

    if show:
        scale = min(1.0, 900 / max(w, h))
        disp  = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        cv2.imshow(f"{img_path.name}", disp)
        print(f"  (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def main():
    global CONF, IMGSZ, Z_MM, MIN_AREA_PX

    p = argparse.ArgumentParser(
        description="YOLOv8 cauliflower inference — single class"
    )
    p.add_argument("--model",   required=True, help="best.pt or best.onnx")
    p.add_argument("--source",  required=True, help="image file or folder")
    p.add_argument("--output",  default="./results")
    p.add_argument("--conf",    type=float, default=CONF,
                   help=f"confidence (default {CONF}). Lower=more detections")
    p.add_argument("--imgsz",   type=int,   default=IMGSZ,
                   help=f"inference size (default {IMGSZ}, match training)")
    p.add_argument("--z",       type=int,   default=Z_MM,
                   help=f"fixed Z in mm (default {Z_MM})")
    p.add_argument("--min_px",  type=int,   default=MIN_AREA_PX,
                   help=f"min mask area px (default {MIN_AREA_PX})")
    p.add_argument("--no_show", action="store_true")
    args = p.parse_args()

    CONF        = args.conf
    IMGSZ       = args.imgsz
    Z_MM        = args.z
    MIN_AREA_PX = args.min_px

    from ultralytics import YOLO
    mp = Path(args.model)
    if not mp.exists():
        print(f"❌ Model not found: {mp}")
        return

    print(f"\n{'='*65}")
    print(f"  CAULIFLOWER MODEL INFERENCE")
    print(f"{'='*65}")
    print(f"  Model  : {mp.name}")
    print(f"  conf   : {CONF}  |  imgsz : {IMGSZ}  |  Z : {Z_MM}mm")
    print(f"  min_px : {MIN_AREA_PX}")
    if mp.suffix.lower() == ".onnx":
        print(f"  ⚠️  ONNX mode — task=segment forced (fixes 0-detection bug)")
    print(f"{'='*65}")

    if mp.suffix.lower() == ".onnx":
        model = YOLO(str(mp), task="segment")
    else:
        model = YOLO(str(mp))

    src  = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in exts
    )

    if not imgs:
        print(f"❌ No images found: {args.source}")
        return

    out_dir = Path(args.output)
    results = []
    for img_path in imgs:
        r = process_one(model, img_path, out_dir, show=not args.no_show)
        if r:
            results.append(r)

    if len(results) > 1:
        tc = sum(r["robot"]["n_cauliflowers"] for r in results)
        print(f"\n{'='*65}")
        print(f"  DONE — {len(results)} images")
        print(f"  Total cauliflowers detected : {tc}")
        print(f"  Results in                  : {out_dir.resolve()}")
        print(f"{'='*65}")
    else:
        print(f"\n  Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
