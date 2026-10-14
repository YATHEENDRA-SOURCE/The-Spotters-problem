# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt
from task1 import *
from task2 import *

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


TEMPLATE_KEYPOINTS = 80    #TODO: Make your own Choice  -> ~80 "parts" of Waldo
SCENE_KEYPOINTS = 3000     #TODO: Make your own chouce  -> ~15-20 per person in a 200-person scene


def sample_bilinear(image: np.ndarray, ys: np.ndarray,
                    xs: np.ndarray) -> np.ndarray:
    """Read the image at fractional coordinates, clamping outside the border."""
    h, w = image.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = ys - y0, xs - x0
    return ((1 - fy) * (1 - fx) * image[y0, x0]
            + (1 - fy) * fx * image[y0, x1]
            + fy * (1 - fx) * image[y1, x0]
            + fy * fx * image[y1, x1])


def orientation_histogram(magnitude: np.ndarray, orientation: np.ndarray,
                          n_bins: int, weights: np.ndarray = None) -> np.ndarray:
    """Histogram of gradient directions over [0, 360), weighted by magnitude.

    Each sample goes entirely into one bin (hard assignment). Bin b covers
    [b * 360 / n_bins, (b + 1) * 360 / n_bins).
    """
    mag = np.asarray(magnitude).ravel()
    ori = np.asarray(orientation).ravel()
    w = mag if weights is None else mag * np.asarray(weights).ravel()
    index = np.clip(np.floor(ori / (360.0 / n_bins)).astype(int), 0, n_bins - 1)
    return np.bincount(index, weights=w, minlength=n_bins).astype(np.float64)


# ================================================================
# %%
# Task 3
# ======
#

def harris_response(image: np.ndarray, sigma_d: float = 1.0,
                    sigma_i: float = 2.0, k: float = 0.04) -> np.ndarray:
    """Harris cornerness: how much the image changes in EVERY direction at once.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma_d : float
        Blur applied before differentiating
    sigma_i : float
        Blur applied to the gradient products
    k : float
        Harris constant, usually 0.04 to 0.06

    Returns
    -------
    np.ndarray
        Cornerness map of shape (H, W)
    """
    # return np.zeros_like(image)  # comment this line and write your code for the function
    smoothed = gaussian_blur(image, sigma_d)
    gx, gy, _mag, _ori = sobel_edges(smoothed)      # Sobel from Task 1
    # structure tensor entries, each averaged over a Gaussian window
    sxx = gaussian_blur(gx * gx, sigma_i)
    syy = gaussian_blur(gy * gy, sigma_i)
    sxy = gaussian_blur(gx * gy, sigma_i)
    det = sxx * syy - sxy ** 2
    trace = sxx + syy
    return det - k * trace ** 2


def detect_corners(image: np.ndarray, threshold_rel: float = 0.02,
                   min_distance: int = 3, max_corners: int = 120,
                   border: int = 4) -> np.ndarray:
    """Turn the Harris map into a list of corner positions.


    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    threshold_rel : float
        Fraction of the maximum response a corner must reach
    min_distance : int
        Minimum separation between corners
    max_corners : int
        Keep at most this many
    border : int
        Width of the ignored frame

    Returns
    -------
    np.ndarray
        Shape (N, 2), columns (y, x), strongest first. (0, 2) if none.
    """
    # return np.zeros((0, 2))  # comment this line and write your code for the function
    response = harris_response(image)
    if border > 0:                     # ignore a frame around the edge
        response = response.copy()
        response[:border, :] = 0.0
        response[-border:, :] = 0.0
        response[:, :border] = 0.0
        response[:, -border:] = 0.0
    top = response.max()
    if top <= 0:
        return np.zeros((0, 2))
    # keep only true local maxima: a point must be the largest value in its
    # own (2*min_distance+1)^2 neighbourhood, otherwise the "shoulders" next
    # to a strong corner also pass the threshold
    r = max(int(min_distance), 1)
    padded = np.pad(response, r, mode="constant", constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(
        padded, (2 * r + 1, 2 * r + 1)).max(axis=(-2, -1))
    response = np.where(response >= local_max, response, 0.0)
    # same greedy, well-separated peak picking as Task 2
    peaks = find_peaks(response, threshold_rel * top,
                       min_distance=min_distance, max_peaks=max_corners)
    return peaks[:, :2].astype(np.float64)


def dog_keypoints(image: np.ndarray, sigma0: float = 0.8, n_scales: int = 8,
                  k: float = 1.25, threshold: float = 0.005,
                  border: int = 4, max_keypoints: int = 800) -> np.ndarray:
    """Find blob-like keypoints, and the SIZE each one lives at.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma0, n_scales, k : float, int, float
        The scale ladder
    threshold : float
        Minimum absolute response
    border : int
        Ignored frame width
    max_keypoints : int
        Keep at most this many

    Returns
    -------
    np.ndarray
        Shape (N, 3), columns (y, x, sigma), strongest first. (0, 3) if none.
    """
    # return np.zeros((0, 3))  # comment this line and write your code for the function
    image = np.asarray(image, dtype=np.float64)
    sigmas = sigma0 * k ** np.arange(n_scales)
    blurred = [gaussian_blur(image, s) for s in sigmas]
    # difference of adjacent levels: dog[l] lives at scale sigmas[l]
    dog = np.stack([blurred[i + 1] - blurred[i] for i in range(n_scales - 1)])
    if dog.shape[0] < 3:
        return np.zeros((0, 3))

    # A keypoint must be STRICTLY above (or below) all 26 neighbours in the
    # 3x3x3 space-scale cube. Using ">=" lets a perfectly straight edge in
    # these synthetic scenes produce a whole line of tied "extrema".
    L, H, W = dog.shape
    padded = np.pad(dog, 1, mode="edge")
    centre = dog
    is_max = dog >= threshold
    is_min = dog <= -threshold
    for dl in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dl == 0 and dy == 0 and dx == 0:
                    continue
                nb = padded[1 + dl:1 + dl + L, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                is_max &= centre > nb
                is_min &= centre < nb
    is_peak = is_max | is_min

    # the top and bottom DoG levels have no neighbour above/below them
    is_peak[0] = False
    is_peak[-1] = False
    # and ignore a frame around the image
    b = max(int(border), 1)
    is_peak[:, :b, :] = False
    is_peak[:, -b:, :] = False
    is_peak[:, :, :b] = False
    is_peak[:, :, -b:] = False

    lev, ys, xs = np.nonzero(is_peak)
    if len(ys) == 0:
        return np.zeros((0, 3))
    strength = np.abs(dog[lev, ys, xs])
    order = np.argsort(-strength, kind="stable")[:max_keypoints]
    return np.column_stack([ys[order], xs[order],
                            sigmas[lev[order]]]).astype(np.float64)


def dominant_orientation(image: np.ndarray, y: float, x: float,
                         sigma: float = 1.6, n_bins: int = 36) -> float:
    """The direction a keypoint's neighbourhood mostly points in.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    y, x : float
        Keypoint position
    sigma : float
        Scale the keypoint was found at
    n_bins : int
        Number of orientation bins

    Returns
    -------
    float
        Angle in degrees in [0, 360), or 0.0 if the window has no gradient
    """
    # return 0.0  # comment this line and write your code for the function
    h, w = image.shape
    weight_sigma = 1.5 * sigma                 # window scales with the keypoint
    radius = int(np.ceil(3.0 * weight_sigma))
    margin = radius + 2                        # a little extra so Sobel sees real pixels
    yi, xi = int(round(y)), int(round(x))
    y0, y1 = max(0, yi - margin), min(h, yi + margin + 1)
    x0, x1 = max(0, xi - margin), min(w, xi + margin + 1)
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 0.0

    _gx, _gy, mag, ori = sobel_edges(gaussian_blur(patch, sigma))
    py, px = np.mgrid[y0:y1, x0:x1]
    d2 = (py - y) ** 2 + (px - x) ** 2
    weights = np.exp(-d2 / (2.0 * weight_sigma ** 2)) * (d2 <= radius ** 2)

    hist = orientation_histogram(mag, ori, n_bins, weights)
    if hist.max() <= 0:
        return 0.0
    peak = int(np.argmax(hist))
    return float(((peak + 0.5) * 360.0 / n_bins) % 360.0)   # centre of the peak bin


def describe_keypoints(image: np.ndarray, keypoints: np.ndarray,
                       orientations: np.ndarray = None) -> np.ndarray:
    """Describe each keypoint by a grid of gradient-orientation histograms.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    keypoints : np.ndarray
        Shape (N, 3), columns (y, x, sigma)
    orientations : np.ndarray, optional
        Shape (N,) in degrees; None means upright descriptors

    Returns
    -------
    np.ndarray
        Shape (N, 128), one row per keypoint
    """
    # return np.zeros((len(keypoints), 128))  # comment this line and write your code for the function
    image = np.asarray(image, dtype=np.float64)
    kps = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    n = len(kps)
    if n == 0:
        return np.zeros((0, 128))
    angles = (np.zeros(n) if orientations is None
              else np.asarray(orientations, dtype=np.float64).ravel())

    # 16 x 16 sample grid in the keypoint's own frame, in units of sigma.
    # 4 x 4 cells, each cell 3 sigma wide -> the window is 12 sigma across.
    cell = 3.0
    step = 4 * cell / 16.0
    offs = (np.arange(16) - 7.5) * step
    v, u = np.meshgrid(offs, offs, indexing="ij")    # v down, u across
    cell_index = (np.arange(16) // 4)
    cell_y, cell_x = np.meshgrid(cell_index, cell_index, indexing="ij")
    spatial_w = np.exp(-(u ** 2 + v ** 2) / (2.0 * (2 * cell) ** 2))

    desc = np.zeros((n, 4, 4, 8))
    # gradients at the keypoint's scale; only a handful of distinct sigmas
    for s in np.unique(kps[:, 2]):
        _gx, _gy, _m, _o = sobel_edges(gaussian_blur(image, s))
        gx, gy = _gx, _gy
        for i in np.nonzero(kps[:, 2] == s)[0]:
            y, x, sig = kps[i]
            th = np.radians(angles[i])
            c, sn = np.cos(th), np.sin(th)
            # rotate + scale the grid into the image
            xs = x + sig * (c * u - sn * v)
            ys = y + sig * (sn * u + c * v)
            px = sample_bilinear(gx, ys, xs)
            py = sample_bilinear(gy, ys, xs)
            mag = np.hypot(px, py) * spatial_w
            # gradient direction measured relative to the keypoint's angle
            rel = (np.degrees(np.arctan2(py, px)) - angles[i]) % 360.0
            b = np.clip((rel // 45.0).astype(int), 0, 7)
            np.add.at(desc[i], (cell_y, cell_x, b), mag)

    desc = desc.reshape(n, 128)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)        # all-zero rows stay zero
    return desc / safe


if __name__ == "__main__":

    template = load_image_as_grayscale("imgs/waldo_template.png")
    scene = load_image_as_grayscale("imgs/scene_easy.png")

    # #############################################
    # Comment these lines and write your code here
    # response = np.zeros_like(template)
    # corners = np.zeros((0, 2))
    # keypoints = np.zeros((0, 3))
    # angles = np.zeros(0)
    # descriptors = np.zeros((0, 128))
    response = harris_response(template)
    corners = detect_corners(template)
    keypoints = dog_keypoints(template, max_keypoints=TEMPLATE_KEYPOINTS)
    angles = np.array([dominant_orientation(template, y, x, s)
                       for y, x, s in keypoints])
    descriptors = describe_keypoints(template, keypoints, angles)
    # #############################################

    print("Harris corners on the template :", len(corners))
    print("DoG keypoints on the template  :", len(keypoints))
    print("sigmas found                   :",
          np.round(np.unique(np.round(keypoints[:, 2], 2)), 2))
    norms = np.linalg.norm(descriptors, axis=1) if len(descriptors) else np.zeros(1)
    print("descriptors                    :", descriptors.shape,
          "row norms in [{:.3f}, {:.3f}]".format(norms.min(), norms.max()))

    scene_keypoints = dog_keypoints(scene, max_keypoints=SCENE_KEYPOINTS)
    print("DoG keypoints on scene_easy    :", len(scene_keypoints))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(template, cmap="gray")
    axes[0].set_title("Template")
    axes[1].imshow(response, cmap="gray")
    axes[1].set_title("Harris response R")
    axes[2].imshow(template, cmap="gray")
    if len(corners):
        axes[2].scatter(corners[:, 1], corners[:, 0], s=18,
                        facecolors="none", edgecolors="lime")
    axes[2].set_title("Harris corners")
    axes[3].imshow(template, cmap="gray")
    if len(keypoints):
        axes[3].scatter(keypoints[:, 1], keypoints[:, 0], s=keypoints[:, 2] * 12,
                        facecolors="none", edgecolors="red")
    axes[3].set_title("DoG keypoints (size = sigma)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task3_results.png", metadata={"Author": getpass.getuser()})
    print("\nwrote plots/task3_results.png")
