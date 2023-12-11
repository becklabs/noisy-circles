from typing import List

import numpy as np

from utils import CircleParams, iou


def thresholded_iou(
    circles: List[CircleParams], pred_circles: List[CircleParams], thresh: float = 0.75
) -> float:
    """
    Returns the % of circles who's IOU is greater than the threshold
    """
    ious = np.array([iou(pred, actual) for pred, actual in zip(pred_circles, circles)])
    return np.sum(np.where(ious > thresh, 1, 0)) / len(ious)
