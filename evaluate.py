"""Re-evaluate a trained model without retraining: per-class depth error and a
fair side-by-side 3D plot (shared axes).

    python evaluate.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from depth3d.geometry import backproject
from depth3d.model import DepthSegNet
from depth3d.scene import CLASSES, Camera, make_dataset
from train import PALETTE, predict

cam = Camera()
model = DepthSegNet(n_classes=len(CLASSES))
model.load_state_dict(torch.load("results/model.pt", map_location="cpu"))
rgb, gt_d, gt_s = make_dataset(300, seed=2, cam=cam)  # same held-out scenes as train.py
pr_d, pr_s = predict(model, rgb)

per_class = {}
for c, name in enumerate(CLASSES):
    m = gt_s == c
    rel = np.abs(pr_d[m] - gt_d[m]) / gt_d[m]
    bias = np.mean((pr_d[m] - gt_d[m]) / gt_d[m])
    per_class[name] = {"abs_rel": round(float(rel.mean()), 4), "signed_bias": round(float(bias), 4)}
print(json.dumps(per_class, indent=2))
res = json.load(open("results/metrics.json"))
res["model"]["depth_by_class"] = per_class
json.dump(res, open("results/metrics.json", "w"), indent=2)

i = 0
pts = [backproject(gt_d[i], cam.K), backproject(pr_d[i], cam.K)]
allp = np.concatenate(pts)
lo, hi = allp.min(0), allp.max(0)
fig = plt.figure(figsize=(10, 4.5))
for k, (p, s, title) in enumerate([(pts[0], gt_s[i], "true 3D scene"),
                                   (pts[1], pr_s[i], "reconstructed from one image")]):
    a = fig.add_subplot(1, 2, k + 1, projection="3d")
    a.scatter(p[:, 0], p[:, 2], p[:, 1], c=PALETTE[s.ravel()], s=2)
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[2], hi[2]); a.set_zlim(lo[1], hi[1])
    a.set_xlabel("x"); a.set_ylabel("depth z"); a.set_zlabel("y")
    a.view_init(elev=22, azim=-60)
    a.set_title(title, fontsize=10)
plt.tight_layout()
plt.savefig("results/pointcloud.png", dpi=120)
print("updated results/pointcloud.png and results/metrics.json")
