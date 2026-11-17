import os
import tempfile

import numpy as np
import torch

from depth3d.geometry import backproject, chamfer, project, read_ply_points, write_ply
from depth3d.metrics import depth_metrics, seg_metrics
from depth3d.model import DepthSegNet
from depth3d.scene import BOX, FLOOR, SPHERE, WALL, Camera, _hit_box, _hit_sphere, render

CAM = Camera()


def centre(arr):
    return arr[CAM.height // 2, CAM.width // 2]


def test_sphere_hits_lie_exactly_on_its_surface():
    d = CAM.rays()
    c, r = np.array([0.0, 0.0, 5.0]), 1.0
    t, _ = _hit_sphere(d, c, r)
    hit = np.isfinite(t)
    assert hit.sum() > 100
    pts = t[hit][:, None] * d[hit]
    assert np.allclose(np.linalg.norm(pts - c, axis=1), r, atol=1e-9)
    # The near side faces the camera: depth is close to 5 - 1 near the centre.
    assert abs(centre(t) - 4.0) < 0.01


def test_box_front_face_depth_and_normal():
    t, n = _hit_box(CAM.rays(), np.array([-1.0, -1.0, 3.0]), np.array([1.0, 1.0, 4.0]))
    assert abs(centre(t) - 3.0) < 1e-6
    assert np.allclose(centre(n), [0, 0, -1])


def test_render_shapes_and_labels():
    rgb, depth, label = render(np.random.default_rng(0), CAM)
    assert rgb.shape == (64, 64, 3) and depth.shape == (64, 64) and label.shape == (64, 64)
    assert np.isfinite(depth).all() and (depth > 0).all()
    assert set(np.unique(label)) <= {WALL, FLOOR, SPHERE, BOX}
    # Bottom rows see the floor, top rows the wall (unless an object is in front).
    assert (label[-1] == FLOOR).mean() > 0.5


def test_backproject_then_project_returns_pixel_grid():
    depth = np.random.default_rng(1).uniform(1, 9, (64, 64))
    uv = project(backproject(depth, CAM.K), CAM.K)
    u, v = np.meshgrid(np.arange(64), np.arange(64))
    assert np.allclose(uv, np.stack([u, v], -1).reshape(-1, 2), atol=1e-6)


def test_backprojected_floor_is_flat():
    rng = np.random.default_rng(3)
    _, depth, label = render(rng, CAM, max_objects=1)
    pts = backproject(depth, CAM.K)[label.ravel() == FLOOR]
    assert pts[:, 1].std() < 1e-4  # every floor point has the same height


def test_chamfer():
    a = np.random.default_rng(2).normal(size=(50, 3))
    assert chamfer(a, a) == 0
    assert abs(chamfer(a, a + [0, 0, 1.0]) - 1.0) < 1.0  # bounded by the shift


def test_ply_roundtrip():
    pts = np.random.default_rng(4).normal(size=(10, 3))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.ply")
        write_ply(p, pts, np.zeros((10, 3)))
        assert np.allclose(read_ply_points(p), pts, atol=1e-4)


def test_model_output_shapes():
    depth, seg = DepthSegNet(n_classes=4, width=8)(torch.zeros(2, 3, 64, 64))
    assert depth.shape == (2, 64, 64) and seg.shape == (2, 4, 64, 64)


def test_metrics_perfect_prediction():
    gt = np.random.default_rng(5).uniform(1, 9, (8, 8))
    m = depth_metrics(gt, gt)
    assert m["abs_rel"] == 0 and m["rmse"] == 0 and m["delta_1.25"] == 1
    lab = np.random.default_rng(6).integers(0, 4, (8, 8))
    assert seg_metrics(lab, lab, 4)["miou"] == 1
