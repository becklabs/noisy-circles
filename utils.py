from typing import NamedTuple, Optional, Tuple, Generator

import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int


class CircleParamsScaler:
    """
    Normalizes CircleParams based on data generator properties.
    """

    def __init__(self, img_size: int, min_radius: int, max_radius: int) -> None:
        # Radius sampled from Uniform(min_radius, max_radius - 1)
        self.radius_mean = (min_radius + max_radius - 1) / 2
        self.radius_std = np.sqrt(((max_radius - 1 - min_radius) ** 2) / 12)

        # Row, col sampled from Uniform(0, img_size - 1)
        self.center_mean = (img_size - 1) / 2
        self.center_std = np.sqrt(((img_size - 1) ** 2) / 12)

    def transform(self, circle: CircleParams) -> CircleParams:
        """
        Transforms the given CircleParams to the standard scale.
        """
        radius = (circle.radius - self.radius_mean) / self.radius_std
        col = (circle.col - self.center_mean) / self.center_std
        row = (circle.row - self.center_mean) / self.center_std
        return CircleParams(row, col, radius)

    def inverse_transform(self, circle: CircleParams) -> CircleParams:
        """
        Transforms the given CircleParams back to its original scale.
        """
        radius = circle.radius * self.radius_std + self.radius_mean
        col = circle.col * self.center_std + self.center_mean
        row = circle.row * self.center_std + self.center_mean
        return CircleParams(row, col, radius)


def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img


def noisy_circle(
    img_size: int, min_radius: float, max_radius: float, noise_level: float
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """

    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def show_circle(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title("Circle")
    plt.show()


def generate_examples(
    noise_level: float = 0.5,
    img_size: int = 100,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None,
    dataset_path: str = "ds",
) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    assert max_radius > min_radius, "max_radius must be greater than min_radius"
    assert img_size > max_radius, "size should be greater than max_radius"
    assert noise_level >= 0, "noise should be non-negative"

    params = (
        f"{noise_level=}, {img_size=}, {min_radius=}, {max_radius=}, {dataset_path=}"
    )
    # print(f"Using parameters: {params}")
    while True:
        img, params = noisy_circle(
            img_size=img_size,
            min_radius=min_radius,
            max_radius=max_radius,
            noise_level=noise_level,
        )
        yield img, params


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        # If the distance between the centers is greater than the sum of the radii, then the circles don't intersect
        return 0.0
    if d <= abs(r1 - r2):
        # If the distance between the centers is less than the absolute difference of the radii, then one circle is
        # inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        return smaller_r**2 / larger_r**2
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union

