"""Standard monocular-depth and segmentation metrics."""
import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.ravel(), gt.ravel()
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "delta_1.25": float(np.mean(ratio < 1.25)),
    }


def seg_metrics(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict:
    ious = []
    for c in range(n_classes):
        inter = np.sum((pred == c) & (gt == c))
        union = np.sum((pred == c) | (gt == c))
        if union:
            ious.append(inter / union)
    return {"pixel_acc": float(np.mean(pred == gt)), "miou": float(np.mean(ious)),
            "per_class_iou": [float(i) for i in ious]}
