"""A training-free local texture and tone anomaly baseline."""

from __future__ import annotations

import cv2
import numpy as np


def _robust_positive_zscore(values: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Positive robust z-score independently along each image row."""

    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, floor)
    return np.maximum((values - median) / scale, 0.0)


def anomaly_score(
    grayscale: np.ndarray, gaussian_sigma: float = 9.0, local_sigma: float = 2.0
) -> np.ndarray:
    """Score local departures from each row's dominant woven texture."""

    if grayscale.ndim != 2:
        raise ValueError("The baseline expects a two-dimensional grayscale image")
    image = grayscale.astype(np.float32) / 255.0
    broad = cv2.GaussianBlur(image, (0, 0), gaussian_sigma, borderType=cv2.BORDER_REFLECT)
    local = cv2.GaussianBlur(image, (0, 0), local_sigma, borderType=cv2.BORDER_REFLECT)

    tone_deviation = np.abs(local - broad)
    high_frequency = image - local
    texture_energy = np.sqrt(
        cv2.GaussianBlur(
            high_frequency * high_frequency,
            (0, 0),
            local_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    )
    score = np.maximum(
        _robust_positive_zscore(tone_deviation),
        _robust_positive_zscore(texture_energy),
    )
    border = max(2, int(round(gaussian_sigma)))
    score[:, :border] = 0
    score[:, -border:] = 0
    score[:2, :] = 0
    score[-2:, :] = 0
    return np.clip(score, 0.0, 20.0).astype(np.float32)


def postprocess_mask(
    score: np.ndarray,
    threshold: float,
    morphology_kernel: int = 5,
    min_component_area: int = 24,
) -> np.ndarray:
    """Threshold, close small gaps, and remove isolated responses."""

    mask = (score >= threshold).astype(np.uint8)
    if morphology_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if min_component_area > 1:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label in range(1, component_count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                filtered[labels == label] = 1
        mask = filtered
    return mask.astype(bool)
