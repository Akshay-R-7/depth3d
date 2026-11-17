"""3D geometry: back-projection, Chamfer distance, PLY export."""
from __future__ import annotations

import numpy as np


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Turn a z-depth map (H,W) into camera-space points (H*W, 3).

    Inverse of the pinhole projection u = fx*x/z + cx, v = cy - fy*y/z.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    return np.stack([x, y, z], -1).reshape(-1, 3)


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * points[:, 0] / points[:, 2] + cx
    v = cy - fy * points[:, 1] / points[:, 2]
    return np.stack([u, v], -1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(1)).mean() + np.sqrt(d2.min(0)).mean()) / 2


def write_ply(path: str, points: np.ndarray, colours: np.ndarray) -> None:
    """ASCII PLY, readable by MeshLab, CloudCompare, Open3D, Blender."""
    c = (np.clip(colours, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, col in zip(points, c):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {col[0]} {col[1]} {col[2]}\n")


def read_ply_points(path: str) -> np.ndarray:
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("end_header") + 1
    return np.array([[float(x) for x in ln.split()[:3]] for ln in lines[start:]])
