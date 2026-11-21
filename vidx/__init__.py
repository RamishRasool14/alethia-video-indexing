"""Video indexing: per-second object presence and people counts with YOLO11l."""

# The five classes we report, as (COCO class id, output name). Order matters:
# per-frame count arrays use this column order everywhere.
CLASSES = [
    (0, "person"),
    (2, "car"),
    (16, "dog"),
    (63, "laptop"),
    (67, "cell phone"),
]
CLASS_IDS = [c for c, _ in CLASSES]
CLASS_NAMES = [n for _, n in CLASSES]
PERSON_COL = 0

IMGSZ = 640
CONF = 0.4
IOU = 0.7
