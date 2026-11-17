"""Synthetic RGB-D scene generator: a small vectorised ray caster in NumPy.

Each scene is a room (back wall + floor) with random spheres and boxes resting on
the floor. For every pixel we get an RGB colour, the exact depth (z-distance from
the camera) and a semantic label. Exact ground truth is what makes the 3D
evaluation possible.

Camera: pinhole at the origin, looking along +z, y up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ("wall", "floor", "sphere", "box")
WALL, FLOOR, SPHERE, BOX = range(4)


@dataclass(frozen=True)
class Camera:
    width: int = 64
    height: int = 64
    focal: float = 56.0  # pixels; ~60 degree field of view at 64 px

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal, 0, (self.width - 1) / 2],
                         [0, self.focal, (self.height - 1) / 2],
                         [0, 0, 1.0]])

    def rays(self) -> np.ndarray:
        """Ray direction per pixel, scaled so the z component is 1.

        With z = 1, the ray parameter t at a hit point equals the z-depth.
        """
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        x = (u - cx) / self.focal
        y = -(v - cy) / self.focal  # image rows go down, world y goes up
        return np.stack([x, y, np.ones_like(x)], axis=-1)


def _hit_sphere(d, center, radius):
    # |t d - c|^2 = r^2  ->  a t^2 + b t + c = 0
    a = (d * d).sum(-1)
    b = -2 * (d @ center)
    c = center @ center - radius ** 2
    disc = b * b - 4 * a * c
    t = np.full(a.shape, np.inf)
    ok = disc >= 0
    t_near = (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a)
    t[ok & (t_near > 0)] = t_near[ok & (t_near > 0)]
    hit = t[..., None] * d
    normal = (hit - center) / radius
    return t, normal


def _hit_box(d, lo, hi):
    # Slab method for an axis-aligned box.
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - 0) / d
        t2 = (hi - 0) / d
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    t_enter = tmin.max(-1)
    t_exit = tmax.min(-1)
    ok = (t_enter <= t_exit) & (t_enter > 0)
    t = np.where(ok, t_enter, np.inf)
    # The face we entered through is the axis with the largest tmin.
    axis = tmin.argmax(-1)
    normal = np.zeros(d.shape)
    sign = -np.sign(np.take_along_axis(d, axis[..., None], -1))[..., 0]
    np.put_along_axis(normal, axis[..., None], sign[..., None], -1)
    return t, normal


def render(rng: np.random.Generator, cam: Camera = Camera(), max_objects: int = 4):
    """Render one random scene. Returns rgb (H,W,3) float32 in [0,1],
    depth (H,W) float32 in scene units, labels (H,W) int64."""
    d = cam.rays()
    H, W = cam.height, cam.width
    floor_y = -rng.uniform(0.8, 1.4)
    wall_z = rng.uniform(6.0, 9.0)

    depth = np.full((H, W), np.inf)
    label = np.zeros((H, W), dtype=np.int64)
    normal = np.zeros((H, W, 3))
    base = np.zeros((H, W, 3))

    def take(t, n, cls, colour):
        closer = t < depth
        depth[closer] = t[closer]
        label[closer] = cls
        normal[closer] = n[closer] if n.ndim == 3 else n
        base[closer] = colour

    # Back wall (plane z = wall_z) and floor (plane y = floor_y).
    take(np.full((H, W), wall_z), np.array([0, 0, -1.0]), WALL, rng.uniform(0.55, 0.85) * np.ones(3))
    with np.errstate(divide="ignore"):
        t_floor = np.where(d[..., 1] < 0, floor_y / d[..., 1], np.inf)
    take(t_floor, np.array([0, 1.0, 0]), FLOOR, rng.uniform(0.25, 0.5) * np.array([1.0, 0.9, 0.75]))

    for _ in range(rng.integers(1, max_objects + 1)):
        colour = rng.uniform(0.2, 1.0, 3)
        z = rng.uniform(2.5, wall_z - 1.0)
        x = rng.uniform(-0.45, 0.45) * z  # keep roughly inside the view
        if rng.random() < 0.5:
            r = rng.uniform(0.3, 0.8)
            t, n = _hit_sphere(d, np.array([x, floor_y + r, z]), r)
            take(t, n, SPHERE, colour)
        else:
            s = rng.uniform(0.4, 1.2, 3)
            lo = np.array([x - s[0] / 2, floor_y, z - s[2] / 2])
            t, n = _hit_box(d, lo, lo + s)
            take(t, n, BOX, colour)

    # Lambertian shading with a random light, plus sensor noise.
    light = rng.normal(size=3) + np.array([0.3, 1.5, -1.0])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normal @ light, 0, 1)
    rgb = base * shade[..., None] + rng.normal(0, 0.02, (H, W, 3))
    return (np.clip(rgb, 0, 1).astype(np.float32),
            depth.astype(np.float32),
            label)


def make_dataset(n: int, seed: int, cam: Camera = Camera()):
    rng = np.random.default_rng(seed)
    rgbs, depths, labels = zip(*(render(rng, cam) for _ in range(n)))
    return np.stack(rgbs), np.stack(depths), np.stack(labels)
