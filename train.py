"""Train the multi-task network, evaluate in 2D and 3D, and save results.

    python train.py              # full run (CPU, a few minutes)
    python train.py --quick      # smoke test
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth3d.geometry import backproject, chamfer, write_ply
from depth3d.metrics import depth_metrics, seg_metrics
from depth3d.model import DepthSegNet
from depth3d.scene import CLASSES, Camera, make_dataset

PALETTE = np.array([[0.75, 0.75, 0.75], [0.55, 0.40, 0.25], [0.15, 0.55, 0.95], [0.95, 0.45, 0.10]])


def to_tensors(rgb, depth, label):
    return (torch.from_numpy(rgb).permute(0, 3, 1, 2), torch.from_numpy(np.log(depth)),
            torch.from_numpy(label))


def predict(model, rgb, batch=128):
    model.eval()
    ds, ss = [], []
    with torch.no_grad():
        for i in range(0, len(rgb), batch):
            x = torch.from_numpy(rgb[i:i + batch]).permute(0, 3, 1, 2)
            d, s = model(x)
            ds.append(torch.exp(d).numpy())
            ss.append(s.argmax(1).numpy())
    return np.concatenate(ds), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.train = 1, 64

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = Camera()
    os.makedirs("results", exist_ok=True)

    t0 = time.time()
    tr = make_dataset(args.train, seed=1, cam=cam)
    te = make_dataset(64 if args.quick else 300, seed=2, cam=cam)  # held-out scenes
    print(f"generated {len(tr[0])} train / {len(te[0])} test scenes in {time.time() - t0:.1f}s")

    x, logd, y = to_tensors(*tr)
    model = DepthSegNet(n_classes=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(x) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            xb, db, yb = x[idx].to(device), logd[idx].to(device), y[idx].to(device)
            if torch.rand(1) < 0.5:  # horizontal flip: the scenes are left-right symmetric
                xb, db, yb = xb.flip(-1), db.flip(-1), yb.flip(-1)
            pd, ps = model(xb)
            loss = F.l1_loss(pd, db) + F.cross_entropy(ps, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {tot / len(x):.4f}  ({time.time() - t0:.0f}s)")
    train_time = time.time() - t0
    torch.save(model.state_dict(), "results/model.pt")

    # ---------- evaluation on held-out scenes ----------
    rgb, gt_d, gt_s = te
    pr_d, pr_s = predict(model.cpu(), rgb)

    # Baseline: predict the average training depth map, and per-pixel majority class.
    mean_d = np.exp(np.log(tr[1]).mean(0))
    maj_s = np.apply_along_axis(lambda v: np.bincount(v, minlength=4).argmax(), 0, tr[2])
    base_d = np.broadcast_to(mean_d, gt_d.shape)
    base_s = np.broadcast_to(maj_s, gt_s.shape)

    # 3D: back-project predicted and true depth, compare the point clouds.
    rng = np.random.default_rng(0)
    cd_model, cd_base, sem3d = [], [], []
    for i in range(len(rgb)):
        gt_p = backproject(gt_d[i], cam.K)
        idx = rng.choice(len(gt_p), 1024, replace=False)
        cd_model.append(chamfer(backproject(pr_d[i], cam.K)[idx], gt_p[idx]))
        cd_base.append(chamfer(backproject(base_d[i], cam.K)[idx], gt_p[idx]))
        sem3d.append(np.mean(pr_s[i].ravel() == gt_s[i].ravel()))

    results = {
        "setup": {"train_scenes": len(tr[0]), "test_scenes": len(rgb), "image": f"{cam.width}x{cam.height}",
                  "epochs": args.epochs, "params": sum(p.numel() for p in model.parameters()),
                  "device": device, "train_seconds": round(train_time, 1)},
        "model": {**depth_metrics(pr_d, gt_d), **seg_metrics(pr_s, gt_s, 4),
                  "chamfer_3d": float(np.mean(cd_model))},
        "baseline_mean_depth_majority_class": {**depth_metrics(base_d, gt_d),
                                               **seg_metrics(base_s, gt_s, 4),
                                               "chamfer_3d": float(np.mean(cd_base))},
        "classes": list(CLASSES),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("model", "baseline_mean_depth_majority_class")}, indent=2))

    # ---------- figures ----------
    n = 4
    fig, ax = plt.subplots(n, 5, figsize=(11, 2.3 * n))
    vmax = float(np.percentile(gt_d[:n], 99))
    for r in range(n):
        for c, (img, title) in enumerate([
            (rgb[r], "input RGB"), (gt_d[r], "true depth"), (pr_d[r], "predicted depth"),
            (PALETTE[gt_s[r]], "true labels"), (PALETTE[pr_s[r]], "predicted labels")]):
            ax[r, c].imshow(img, cmap="viridis" if img.ndim == 2 else None, vmin=0, vmax=vmax if img.ndim == 2 else None)
            ax[r, c].set_axis_off()
            if r == 0:
                ax[r, c].set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/examples.png", dpi=120)
    plt.close()

    # One reconstructed scene: true vs predicted 3D, coloured by predicted class.
    i = 0
    gt_p = backproject(gt_d[i], cam.K)
    pr_p = backproject(pr_d[i], cam.K)
    write_ply("results/scene_true.ply", gt_p, PALETTE[gt_s[i].ravel()])
    write_ply("results/scene_predicted.ply", pr_p, PALETTE[pr_s[i].ravel()])
    fig = plt.figure(figsize=(10, 4.5))
    for k, (pts, cols, title) in enumerate([(gt_p, PALETTE[gt_s[i].ravel()], "true 3D scene"),
                                            (pr_p, PALETTE[pr_s[i].ravel()], "reconstructed from one image")]):
        a = fig.add_subplot(1, 2, k + 1, projection="3d")
        a.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=2)
        a.set_xlabel("x"); a.set_ylabel("depth z"); a.set_zlabel("y")
        a.view_init(elev=22, azim=-60)
        a.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/pointcloud.png", dpi=120)
    plt.close()
    print("saved results/")


if __name__ == "__main__":
    main()
