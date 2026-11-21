"""GPU post-processing equivalent to Ultralytics' non_max_suppression for our use.

Ultralytics (multi_label=False) gives each candidate box its single best class
over all 80 classes, keeps it if that score > conf, then runs class-aware NMS
(IoU 0.7). Restricting to our five classes after the arg-max (as its
`classes=` argument does) does not change the five classes' results, because
NMS never compares boxes of different classes.

We only need the number of surviving boxes per class per image.
"""
import torch
import torchvision

from . import CLASS_IDS, CONF, IOU

MAX_NMS = 30000  # Ultralytics defaults
MAX_DET = 300


_CLS = {}


def count_classes(pred, conf=CONF, iou=IOU):
    """pred: (B, 4 + 80, N) raw YOLO output -> int32 tensor (B, 5) on GPU."""
    B = pred.shape[0]
    device = pred.device
    cls_ids = _CLS.get(device)
    if cls_ids is None:
        cls_ids = _CLS[device] = torch.tensor(CLASS_IDS, device=device)
    p = pred.transpose(1, 2)  # (B, N, 84)
    scores, labels = p[..., 4:].max(-1)  # best class per box
    keep = (scores > conf) & torch.isin(labels, cls_ids)
    b_idx, a_idx = keep.nonzero(as_tuple=True)
    out = torch.zeros((B, len(CLASS_IDS)), dtype=torch.int32, device=device)
    if b_idx.numel() == 0:
        return out
    boxes = p[b_idx, a_idx, :4].float()
    xy, wh = boxes[:, :2], boxes[:, 2:] / 2
    boxes = torch.cat((xy - wh, xy + wh), 1)  # xywh -> xyxy
    sc = scores[b_idx, a_idx].float()
    lab = labels[b_idx, a_idx]
    # Ultralytics keeps at most MAX_NMS candidates per image (by score); with
    # conf 0.4 and five classes this never binds, so we do not emulate it.
    group = b_idx * 80 + lab
    kept = torchvision.ops.batched_nms(boxes, sc, group, iou)
    kb, kl = b_idx[kept], lab[kept]
    # MAX_DET per image: also never binds in practice for 5 classes.
    col = torch.searchsorted(cls_ids, kl)
    out.index_put_((kb, col), torch.ones_like(kb, dtype=torch.int32), accumulate=True)
    return out
