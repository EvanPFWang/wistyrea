#python regions.py "Mural_Crown_of_Italian_City.svg.png"   --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7 --debug

from __future__ import annotations
import os, sys, glob
from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np

def auto_canny(gray: np.ndarray, sigma: float = 0.33,
               aperture_size: int = 3, L2: bool = True) -> np.ndarray:
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv.Canny(gray, lo, hi, apertureSize=aperture_size, L2gradient=L2)


def close_gaps(edges: np.ndarray, ksize: int = 3,
               shape: int = cv.MORPH_ELLIPSE, iters: int = 1) -> np.ndarray:
    if ksize is None or ksize <= 1:
        return edges
    k = cv.getStructuringElement(shape, (ksize, ksize))
    return cv.morphologyEx(edges, cv.MORPH_CLOSE, k, iterations=iters)


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return th



def _find_border_seed(blockers01: np.ndarray) -> Tuple[int, int]:
    """Return a background pixel on the border; fallback (0,0)."""
    h, w = blockers01.shape
    for x in range(w):
        if blockers01[0, x] == 0: return (x, 0)
        if blockers01[h-1, x] == 0: return (x, h-1)
    for y in range(h):
        if blockers01[y, 0] == 0: return (0, y)
        if blockers01[y, w-1] == 0: return (w-1, y)
    return (0, 0)


def fill_regions_from_edges(edges: np.ndarray, seed: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """Return uint8 {0,255} of interior regions delimited by edges (treated as barriers)."""
    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    h, w = edges.shape[:2]

    # floodFill requires (h+2, w+2) mask; non-zero blocks filling
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    ff_mask[1:h+1, 1:w+1][edges > 0] = 1

    if seed is None:
        seed = _find_border_seed((edges > 0).astype(np.uint8))

    dummy = np.zeros((h, w), np.uint8)

    # Flags = connectivity(4) | (newMaskVal<<8) | MASK_ONLY
    flags = 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY

    # IMPORTANT: positional loDiff/upDiff (0, 0) – no keywords.
    cv.floodFill(dummy, ff_mask, seed, 0, 0, 0, flags)

    # Outside became 255 in mask (excluding 1px border); interiors stayed 0.
    inside = (ff_mask[1:h+1, 1:w+1] == 0).astype(np.uint8) * 255
    return inside


# ---------- Contours + hierarchy ----------

def find_contours_tree(filled: np.ndarray, min_area: int = 150) -> Tuple[List[np.ndarray], np.ndarray]:
    cnts, hier = cv.findContours(filled, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return [], np.empty((0, 4), np.int32)

    # filter and rebuild hierarchy indexing
    keep_idx = [i for i, c in enumerate(cnts) if cv.contourArea(c) >= float(min_area)]
    keep = [cnts[i] for i in keep_idx]
    if not keep:
        return [], np.empty((0, 4), np.int32)

    oldh = hier[0]
    idx_map = {old: new for new, old in enumerate(keep_idx)}
    new_h = np.full((len(keep), 4), -1, dtype=np.int32)
    for new_i, old_i in enumerate(keep_idx):
        nxt, prv, ch, par = oldh[old_i]
        new_h[new_i, 0] = idx_map.get(nxt, -1)
        new_h[new_i, 1] = idx_map.get(prv, -1)
        new_h[new_i, 2] = idx_map.get(ch,  -1)
        new_h[new_i, 3] = idx_map.get(par, -1)
    return keep, new_h


# ---------- Drawing helpers ----------

def _draw_subtree_gray(mask: np.ndarray, contours: List[np.ndarray], hier: np.ndarray,
                       root_idx: int, root_color: int = 255, hole_color: int = 0) -> None:
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        color = root_color if (depth % 2 == 0) else hole_color
        cv.drawContours(mask, contours, i, color=int(color), thickness=-1, lineType=cv.LINE_AA)
        child = hier[i, 2]
        while child != -1:
            stack.append((child, depth + 1))
            child = hier[child, 0]


def _draw_subtree_color(img: np.ndarray, contours: List[np.ndarray], hier: np.ndarray,
                        root_idx: int, root_color: Tuple[int,int,int],
                        hole_color: Tuple[int,int,int] = (0,0,0)) -> None:
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        color = root_color if (depth % 2 == 0) else hole_color
        cv.drawContours(img, contours, i, color=color, thickness=-1, lineType=cv.LINE_AA)
        child = hier[i, 2]
        while child != -1:
            stack.append((child, depth + 1))
            child = hier[child, 0]


def export_shape_masks(contours: List[np.ndarray], hier: np.ndarray,
                       shape_hw: Tuple[int, int], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = shape_hw
    saved: List[Path] = []
    for i in range(len(contours)):
        if hier[i, 3] != -1:
            continue  # only top-level
        mask = np.zeros((h, w), np.uint8)
        _draw_subtree_gray(mask, contours, hier, i, 255, 0)
        p = out_dir / f"shape_{i:03d}.png"
        cv.imwrite(str(p), mask)
        saved.append(p)
    return saved

def colorize_regions(contours: List[np.ndarray], hier: np.ndarray,
                     shape_hw: Tuple[int,int],
                     palette: Optional[List[Tuple[int,int,int]]] = None) -> np.ndarray:
    h, w = shape_hw
    color_map = np.zeros((h, w, 3), np.uint8)

    # determine top-level shapes in a stable order
    top = _stable_top_level_indices(hier, contours)
    n = len(top)

    # if no palette supplied, fall back to random
    if palette is None or len(palette) < n:
        if palette is None: palette = []
        needed = n - len(palette)
        rng = np.random.default_rng(42)
        palette += [tuple(int(x) for x in rng.integers(64, 256, size=3)) for _ in range(needed)]

    # draw each top-level with its colour; holes are black
    for j, i in enumerate(top):
        _draw_subtree_color(color_map, contours, hier, i, root_color=palette[j], hole_color=(0,0,0))
    return color_map



def draw_overlay(img_bgr: np.ndarray, contours: List[np.ndarray],
                 color=(0,255,0), thick=2) -> np.ndarray:
    overlay = img_bgr.copy()
    cv.drawContours(overlay, contours, -1, color, thick, lineType=cv.LINE_AA)
    return overlay

# ---------- Palette: MiniBatch K-Means in RGB (with OpenCV fallback) ----------

def _stable_top_level_indices(hier: np.ndarray, contours: List[np.ndarray]) -> List[int]:
    """Deterministic order for top-level shapes: sort by (y, x) of contour centroid."""
    idxs = [i for i in range(len(contours)) if hier[i, 3] == -1]
    def centroid(i):
        m = cv.moments(contours[i]); x = (m["m10"]/(m["m00"]+1e-9)); y = (m["m01"]/(m["m00"]+1e-9))
        return (y, x)
    idxs.sort(key=centroid)
    return idxs

def _rgb_sample_grid(step: int = 7, lo: int = 24, hi: int = 231) -> np.ndarray:
    """
    Uniform grid sample of RGB cube (uint8). `step` controls resolution:
    count ≈ ((hi-lo)/step + 1)^3. Defaults avoid near-black/near-white.
    """
    vals = np.arange(lo, hi+1, step, dtype=np.uint8)
    r, g, b = np.meshgrid(vals, vals, vals, indexing="xy")
    X = np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.float32)
    return X

def palette_kbatchmeans_rgb(n: int,
                            sample_step: int = 7,
                            rng_seed: int = 42,
                            max_iter: int = 200,
                            batch_size: int = 2048) -> List[Tuple[int,int,int]]:
    """
    Build n roughly evenly-spaced RGB colours by clustering a uniform RGB sample.
    Uses sklearn MiniBatchKMeans if available; otherwise falls back to cv2.kmeans.
    Returns list of (B,G,R) tuples for OpenCV drawing.
    """
    X = _rgb_sample_grid(step=sample_step)  # float32 in [0,255]
    centers = None

    # Try scikit-learn MiniBatchKMeans (fast “k-batch-means”).
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
        kmb = MiniBatchKMeans(
            n_clusters=n,
            batch_size=8192,
            max_iter=max_iter,
            init="k-means++",
            random_state=rng_seed,
            n_init="auto",
            verbose=0,
        )
        kmb.fit(X)
        centers = kmb.cluster_centers_
    except Exception:
        # Fallback: OpenCV kmeans (standard Lloyd's).
        X32 = X.astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _compact, _labels, centers = cv.kmeans(
            X32, K=n, bestLabels=None, criteria=criteria, attempts=6, flags=cv.KMEANS_PP_CENTERS
        )

    centers = np.clip(centers, 0, 255).astype(np.uint8)  # RGB uint8
    # Convert RGB to BGR for OpenCV drawing
    centers_bgr = [(int(c[2]), int(c[1]), int(c[0])) for c in centers]
    return centers_bgr


# ---------- Main pipeline (this is the function your CLI calls) ----------

def process_image(
    img_path: str,
    out_overlay_path: str = "contours_overlay.png",
    out_filled_path: str  = "filled_mask.png",
    out_color_path: str   = "coloured_regions.png",
    out_masks_dir: str    = "shape_masks",
    have_edges: Optional[np.ndarray] = None,
    min_area: int = 20,
    gap_close: int = 1,
    do_threshold: bool = False,
    blur_ksize: int = 1,
    canny_sigma: float = 0.33,
    palette_mode: str = "kbatch",   # NEW: {"kbatch","random"}
    sample_step: int = 7,
):
    """Run: edges → close gaps → fill regions → contours → colour + export masks."""
    img = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1) Edge map
    if have_edges is not None:
        edges = have_edges.astype(np.uint8)
    else:
        g = gray
        if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
            g = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = auto_canny(g, sigma=canny_sigma, aperture_size=3, L2=True)

    # 2) Close small gaps (optional)
    edges_closed = close_gaps(edges, ksize=gap_close, shape=cv.MORPH_ELLIPSE, iters=1)

    # 3) Regions from edges
    filled = fill_regions_from_edges(edges_closed, seed=None)

    if do_threshold:
        th = threshold_otsu(gray)
        filled = cv.bitwise_and(filled, th)

    # 4) Contours + hierarchy
    contours, hierarchy = find_contours_tree(filled, min_area=min_area)

    # Build palette if requested
    palette = None
    if palette_mode.lower() == "kbatch":
        n_top = sum(1 for i in range(len(contours)) if hierarchy[i, 3] == -1)
        palette = palette_kbatchmeans_rgb(n_top, sample_step=sample_step)

    color_map = colorize_regions(contours, hierarchy, filled.shape, palette=palette)
    overlay   = draw_overlay(img, contours, color=(0,255,0), thick=2)

    cv.imwrite(out_overlay_path, overlay)
    cv.imwrite(out_filled_path,  filled)
    cv.imwrite(out_color_path,   color_map)
    saved_masks = export_shape_masks(contours, hierarchy, filled.shape, Path(out_masks_dir))
    return edges_closed, filled, color_map, contours, hierarchy, saved_masks
# ---------- CLI ----------

def _find_default_image() -> Optional[str]:
    env = os.getenv("INPUT_IMAGE")
    if env and Path(env).exists():
        return env
    for pat in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"):
        for p in glob.glob(pat):
            if Path(p).is_file():
                return p
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge → Region → Colour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="Path to input image", default=None)
    parser.add_argument("--image", dest="image_opt", help="Alternative to positional image path")
    parser.add_argument("--overlay", default="contours_overlay.png", help="Output: contours overlay (BGR)")
    parser.add_argument("--filled",  default="filled_mask.png",      help="Output: filled binary mask")
    parser.add_argument("--color",   default="coloured_regions.png", help="Output: per-region coloured map")
    parser.add_argument("--masks_dir", default="shape_masks",        help="Output dir for per-region masks")
    parser.add_argument("--blur", type=int, default=1, help="Gaussian blur kernel (odd). 1 disables smoothing")
    parser.add_argument("--canny_sigma", type=float, default=0.33, help="Auto-Canny sigma")
    parser.add_argument("--close", dest="gap_close", type=int, default=1,
                        help="Closing kernel (odd). Default 1 disables closing entirely.")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum contour area to keep (default 20)")
    parser.add_argument("--threshold", action="store_true",
                        help="AND filled mask with Otsu threshold of the grayscale image")
    parser.add_argument("--edge", help="Optional precomputed edge map path (uint8 0/255)", default=None)
    parser.add_argument("--palette_mode", choices=["kbatch", "random"], default="kbatch",
                        help="Colour assignment strategy")
    parser.add_argument("--sample_step", type=int, default=7,
                        help="RGB grid step for k-batch-means (smaller = denser)")
    args = parser.parse_args()

    img_path = args.image_opt or args.image or _find_default_image()
    if not img_path or not Path(img_path).exists():
        parser.print_help(sys.stderr)
        print("\nERROR: No input image supplied and none found in the current directory. "
              "Provide an image path (positional or --image) or set INPUT_IMAGE.", file=sys.stderr)
        sys.exit(2)

    have_edges = None
    if args.edge:
        em = cv.imread(args.edge, cv.IMREAD_GRAYSCALE)
        if em is None:
            print(f"WARNING: Could not read --edge '{args.edge}'. Falling back to Canny.", file=sys.stderr)
        else:
            have_edges = (em > 0).astype("uint8") * 255

    edges, filled, color_map, contours, hierarchy, saved = process_image(
        img_path,
        out_overlay_path=args.overlay,
        out_filled_path=args.filled,
        out_color_path=args.color,
        out_masks_dir=args.masks_dir,
        have_edges=have_edges,
        min_area=args.min_area,
        gap_close=args.gap_close,
        do_threshold=args.threshold,
        blur_ksize=args.blur,
        canny_sigma=args.canny_sigma,
        palette_mode=args.palette_mode,  # NEW
        sample_step=args.sample_step,  # NEW
    )

    print(f"[OK] Saved:\n  • {args.overlay}\n  • {args.filled}\n  • {args.color}\n"
          f"  • {len(saved)} masks in '{args.masks_dir}/'")
    print(f"Contours kept: {len(contours)}")
