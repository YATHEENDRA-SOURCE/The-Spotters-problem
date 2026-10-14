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
from task3 import *
# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


def with_unit_scale(keypoints: np.ndarray) -> np.ndarray:
    """Give scale-less keypoints a sigma of 1, so (N, 2) and (N, 3) both work.

    Harris corners are (y, x): a corner has no size of its own. Padding them
    to (y, x, 1) lets the same matching code take either detector -- at the
    cost of the scale ratio then carrying no information at all.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    kp = kp.reshape(-1, kp.shape[-1]) if kp.size else kp.reshape(0, 3)
    if kp.shape[1] >= 3:
        return kp
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1)


def box_from_centre(cx: float, cy: float, scale: float,
                    template_shape: tuple) -> tuple:
    """A box the shape of the template, scaled and centred on (cx, cy)."""
    th, tw = template_shape
    w = max(1, int(round(tw * scale)))
    h = max(1, int(round(th * scale)))
    return (int(np.floor(cx - w / 2.0)), int(np.floor(cy - h / 2.0)), w, h)


def draw_matches(template: np.ndarray, scene_rgb: np.ndarray,
                 kp_t: np.ndarray, kp_s: np.ndarray,
                 matches: np.ndarray, votes: np.ndarray, path: str) -> None:
    """Template and scene side by side, with a line for every match."""
    zoom = 4
    left = np.repeat(np.repeat(np.dstack([template] * 3), zoom, 0), zoom, 1)
    height = max(left.shape[0], scene_rgb.shape[0])
    canvas = np.ones((height, left.shape[1] + scene_rgb.shape[1], 3))
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:scene_rgb.shape[0], left.shape[1]:] = scene_rgb

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)
    for (a, b), voted in zip(matches, votes):
        plt.plot([kp_t[a, 1] * zoom, kp_s[b, 1] + left.shape[1]],
                 [kp_t[a, 0] * zoom, kp_s[b, 0]],
                 color="lime" if voted else "red", linewidth=0.7, alpha=0.85)
    plt.title("green = agreed with the consensus, red = outvoted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, metadata={"Author": getpass.getuser()})
    plt.close()


# ================================================================
# %%
# Task 4
# ======


def descriptor_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Distance from every descriptor in A to every descriptor in B.

    Parameters
    ----------
    desc_a : np.ndarray
        Shape (Na, D)
    desc_b : np.ndarray
        Shape (Nb, D)

    Returns
    -------
    np.ndarray
        Shape (Na, Nb) of Euclidean distances
    """
    # return np.zeros((len(desc_a), len(desc_b)))  # comment this line and write your code for the function
    a = np.asarray(desc_a, dtype=np.float64)
    b = np.asarray(desc_b, dtype=np.float64)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, all pairs from one matrix product
    sq = (np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :]
          - 2.0 * a @ b.T)
    return np.sqrt(np.maximum(sq, 0.0))     # rounding can dip just below zero


def match_ratio_test(desc_scene: np.ndarray, desc_template: np.ndarray,
                     ratio: float = 0.8) -> np.ndarray:
    """Ask every SCENE keypoint which template part it looks like.

    Parameters
    ----------
    desc_scene : np.ndarray
        Shape (Ns, D), the queries
    desc_template : np.ndarray
        Shape (Nt, D), the small database
    ratio : float
        Lowe's threshold

    Returns
    -------
    np.ndarray
        Shape (M, 2) of (template index, scene index), best match first.
        (0, 2) if nothing survives. Note the column order: template first,
        which is what predict_centres expects.
    """
    # return np.zeros((0, 2), dtype=int)  # comment this line and write your code for the function
    desc_scene = np.asarray(desc_scene, dtype=np.float64)
    desc_template = np.asarray(desc_template, dtype=np.float64)
    if len(desc_scene) == 0 or len(desc_template) < 2:
        return np.zeros((0, 2), dtype=int)

    # rows = scene keypoints (queries), columns = template parts (database)
    d = descriptor_distances(desc_scene, desc_template)
    rows = np.arange(len(d))
    two = np.argpartition(d, 1, axis=1)[:, :2]
    d_two = d[rows[:, None], two]
    swap = d_two[:, 0] > d_two[:, 1]
    best = np.where(swap, two[:, 1], two[:, 0])
    d1 = np.minimum(d_two[:, 0], d_two[:, 1])
    d2 = np.maximum(d_two[:, 0], d_two[:, 1])

    # Lowe: keep only if the best template part is clearly better than the
    # runner-up. No mutual check -- many scene points may pick the same part.
    has_desc = np.linalg.norm(desc_scene, axis=1) > 0
    ok = (d1 < ratio * d2) & has_desc
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(d1[idx], kind="stable")]           # best match first
    return np.column_stack([best[idx], idx]).astype(int).reshape(-1, 2)


def predict_centres(kp_template: np.ndarray, kp_scene: np.ndarray,
                    matches: np.ndarray, ori_template: np.ndarray,
                    ori_scene: np.ndarray, template_shape: tuple) -> np.ndarray:
    """Turn every single match into a full guess at where Waldo is.

    This is the idea the whole task turns on. A matched keypoint carries a
    scale and an orientation as well as a position, so comparing them gives
    the size change and the rotation -- and once you know those, ONE match is
    enough to point at the centre of the whole object.

    For a match pairing template keypoint (y_t, x_t, sigma_t) at angle
    theta_t with scene keypoint (y_s, x_s, sigma_s) at theta_s, and template
    shape (H_t, W_t):

        s      = sigma_s / sigma_t                     the size change
        dtheta = theta_s - theta_t                     the rotation, degrees
        ox, oy = W_t / 2 - x_t,  H_t / 2 - y_t         offset to the centre
        cx     = x_s + s * (cos(dtheta) * ox - sin(dtheta) * oy)
        cy     = y_s + s * (sin(dtheta) * ox + cos(dtheta) * oy)

    A correct match puts (cx, cy) on Waldo. A wrong one puts it anywhere at
    all, possibly off the image. That is expected, and Task 4.4 cleans it up.

    Parameters
    ----------
    kp_template, kp_scene : np.ndarray
        Shapes (Nt, 3) and (Ns, 3), columns (y, x, sigma)
    matches : np.ndarray
        Shape (M, 2) from match_ratio_test
    ori_template, ori_scene : np.ndarray
        Orientations in degrees, one per keypoint
    template_shape : tuple
        (H_t, W_t) of the template these keypoints came from

    Returns
    -------
    np.ndarray
        Shape (M, 3), columns (cx, cy, s), in scene pixels. (0, 3) if no
        matches. Vectorise it -- no Python loop over matches.
    """
    # return np.zeros((0, 3))  # comment this line and write your code for the function
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    if len(matches) == 0:
        return np.zeros((0, 3))
    ti, si = matches[:, 0], matches[:, 1]
    yt, xt, st = kp_template[ti, 0], kp_template[ti, 1], kp_template[ti, 2]
    ys, xs, ss = kp_scene[si, 0], kp_scene[si, 1], kp_scene[si, 2]
    ht, wt = template_shape[:2]

    s = ss / st                                              # size change
    dtheta = np.radians(np.asarray(ori_scene)[si] - np.asarray(ori_template)[ti])
    ox, oy = wt / 2.0 - xt, ht / 2.0 - yt                    # offset to centre
    c, sn = np.cos(dtheta), np.sin(dtheta)
    cx = xs + s * (c * ox - sn * oy)
    cy = ys + s * (sn * ox + c * oy)
    return np.column_stack([cx, cy, s])


def consensus_centre(predictions: np.ndarray, tol: float = 12.0,
                     min_votes: int = 8) -> tuple:
    """Find where the guesses pile up -- the DENSEST cluster, not the average.

    Only about one match in eight is correct here. Any method that assumes
    the good ones are a majority will fail, and that includes taking the
    median: the median of all the predictions lands in the middle of the
    scattered wrong ones and points at nothing. Measure that for yourself,
    it is one of the questions.

    Instead:
      1. Compute the distance from every prediction to every other one.
      2. Count, for each prediction, how many lie strictly within tol of it.
      3. The one with the highest count is the seed.
      4. The consensus is every prediction strictly within tol of that seed.
      5. Fewer than min_votes in it means no detection.
      6. Otherwise report the MEDIAN centre and MEDIAN scale of that cluster.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (M, 3) from predict_centres
    tol : float
        How close two guesses must be to count as agreeing, in pixels
    min_votes : int
        Fewest agreeing guesses that counts as a detection

    Returns
    -------
    tuple
        centre (cx, cy) or None, scale (float) or None, and a boolean mask of
        shape (M,) marking which predictions joined the consensus
    """
    # return None, None, np.zeros(0, dtype=bool)  # comment this line and write your code for the function
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1, 3)
    m = len(predictions)
    if m == 0:
        return None, None, np.zeros(0, dtype=bool)

    xy = predictions[:, :2]
    # 1. every prediction to every other
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # 2. how many neighbours strictly within tol (not counting itself)
    close = dist < tol
    counts = close.sum(axis=1) - 1
    # 3. the most-agreed-with guess is the seed
    seed = int(np.argmax(counts))
    # 4. everything strictly within tol of the seed (the seed included)
    members = close[seed].copy()
    # 5. too few voters -> no detection
    if members.sum() < min_votes:
        return None, None, np.zeros(m, dtype=bool)
    # 6. median centre and median scale of that cluster
    cx, cy = np.median(xy[members], axis=0)
    scale = float(np.median(predictions[members, 2]))
    return (float(cx), float(cy)), scale, members


def find_waldo(scene: np.ndarray, template: np.ndarray,
               upsample: float = 2.0) -> dict:
    """Put it all together: detect, describe, match, vote.

    Parameters
    ----------
    scene : np.ndarray
        Grayscale scene
    template : np.ndarray
        Grayscale template at its native size
    upsample : float
        How much to enlarge the template before detecting

    Returns
    -------
    dict
        Keys: "box" ((x, y, w, h) or None), "n_matches", "n_votes",
        "matches", "votes", "kp_template", "kp_scene". Return None for the
        box when there is no consensus -- do not invent one.
    """
    # return {"box": None, "n_matches": 0, "n_votes": 0,
    #         "matches": np.zeros((0, 2), int),
    #         "votes": np.zeros(0, bool),
    #         "kp_template": np.zeros((0, 3)),
    #         "kp_scene": np.zeros((0, 3))}  # comment this line and write your code for the function
    # 1. detect: enlarge the small template so it has enough blobs to find
    big = resize_template(template, upsample)
    # a finer scale ladder than the default (k=1.15 instead of 1.25), so the
    # size ratio sigma_s / sigma_t is not rounded as coarsely
    ladder = dict(sigma0=0.8, n_scales=12, k=1.15)
    kp_t = with_unit_scale(dog_keypoints(big, max_keypoints=TEMPLATE_KEYPOINTS, **ladder))
    kp_s = with_unit_scale(dog_keypoints(scene, max_keypoints=SCENE_KEYPOINTS, **ladder))

    # 2. describe, each keypoint in its own rotated + scaled frame
    ori_t = np.array([dominant_orientation(big, y, x, s) for y, x, s in kp_t])
    ori_s = np.array([dominant_orientation(scene, y, x, s) for y, x, s in kp_s])
    desc_t = describe_keypoints(big, kp_t, ori_t)
    desc_s = describe_keypoints(scene, kp_s, ori_s)

    # 3. match: every scene point asks which template part it looks like
    matches = match_ratio_test(desc_s, desc_t)

    # 4. vote: each match predicts a centre, find where they pile up
    predictions = predict_centres(kp_t, kp_s, matches, ori_t, ori_s, big.shape)
    centre, scale, votes = consensus_centre(predictions)

    box = None
    if centre is not None:
        # scale is relative to the enlarged template, so use its shape
        box = box_from_centre(centre[0], centre[1], scale, big.shape[:2])

    # report template keypoints in native template pixels (for draw_matches)
    kp_t_native = kp_t.copy()
    kp_t_native[:, :3] /= upsample
    return {"box": box, "n_matches": int(len(matches)),
            "n_votes": int(votes.sum()) if len(votes) else 0,
            "matches": matches, "votes": votes,
            "kp_template": kp_t_native, "kp_scene": kp_s}


if __name__ == "__main__":

    # The two pipelines want different inputs, and the difference is the
    # point. Correlation (Task 2) keeps all three channels, because Waldo's
    # red is most of what identifies him. The feature pipeline (Task 3)
    # needs a scalar field to take gradients of, so it gets luminance.
    template_rgb = load_image_as_rgb("imgs/waldo_template.png")
    template = to_grayscale(template_rgb)
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>9}{:>10}".format(
        "scene", "matches", "votes", "IoU", "agree?"))
    for name in scene_names:
        scene_rgb = load_image_as_rgb("imgs/{}.png".format(name))
        scene = to_grayscale(scene_rgb)
        truth = load_ground_truth(name)

        # ###############################################
        # Comment these lines and write your code here
        found = find_waldo(scene, template)
        correlation = detect_multiscale(scene_rgb, template_rgb)
        # ###############################################

        feature_box = found["box"]
        corr_box = None
        if correlation:
            best = correlation[0]
            corr_box = (best["x"], best["y"], best["w"], best["h"])

        # Scored against the truth here, for your report only. Notice that
        # find_waldo itself never sees it.
        overlap = iou(feature_box, truth) if feature_box else 0.0
        # Do the two completely different methods agree with each other?
        agreement = (iou(feature_box, corr_box)
                     if feature_box and corr_box else 0.0)
        print("{:<16}{:>9}{:>8}{:>9.3f}{:>10.3f}".format(
            name, found["n_matches"], found["n_votes"], overlap, agreement))

        draw_matches(template, scene_rgb, found["kp_template"],
                        found["kp_scene"], found["matches"], found["votes"],
                        f"plots/task4_{name}_matches.png")

        shown = draw_box(scene_rgb, truth, (0.1, 0.4, 1.0))
        if feature_box:
            shown = draw_box(shown, feature_box,
                             (0, 1, 0) if overlap >= 0.5 else (1, 0, 0))
        plt.figure(figsize=(9, 7))
        plt.imshow(shown)
        plt.title("{}: blue = truth, green/red = your detection".format(name))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("plots/task4_{}.png".format(name),
                    metadata={"Author": getpass.getuser()})
        plt.close()

    print("\nwrote plots/task4_matches.png and plots/task4_<scene>.png")
    print("The last column is agreement between Task 2 and Task 4, computed")
    print("without any ground truth. Compare it against the IoU column.")
