#python regions.py "C:\Users\User\Documents\GitHub\wistyrea\Mural_Crown_of_Italian_City.svg.png"   --blur 1 --close 1 --min_area 5 --canny_sigma 0.25 --mask_mode "all" --palette_mode kbatch --sample_step 7

"""Canny sigma:--canny_sigma 0.25 is a good start for clear line art; 0.33–0.5 for softer edges. (auto-thresholding strategy is solid; it’s common practice.)

Gap closing: --close 1 = off; try 3 if tiny edge breaks, but beware over-merging (it reduces region count).

Min area: keep 20, or make it image-scale aware: max(20, int(0.00002 * H * W)).

AA policy: only the web-facing assets (id_map.png, shape_*.png) must be LINE_8; keep LINE_AA for overlay and coloured_regions.png for visual quality. (OpenCV docs show the default lineType and alternatives


python3 regions.py "Mural_Crown_of_Italian_City.svg.png" --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7
-> Produces: contours_overlay.png, filled_mask.png, coloured_regions.png, mask_background.png, id_map.png, metadata.json, palette.json, and shape_masks/shape_000.png ...

On the web: load id_map.png into a hidden canvas (no scaling), use coalesced events inside requestAnimationFrame, and change UI only when id changes. (If you ever offload to a worker, use OffscreenCanvas.)

If you want me to tailor tiny diffs for your exact file names/paths (interactive-svg-mural.html, mural-typescript.ts, build script), paste those snippets and I’ll mark the exact insert/replace lines.

Sources


"""
#python regions.py "Mural_Crown_of_Italian_City.svg.png"   --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7 --debug

from __future__ import annotations
import os, sys, glob
from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np
import json

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


def _depth(idx: int, hier: np.ndarray) -> int:
    """Return the depth of contour `idx` in the hierarchy."""
    d = 0
    parent = hier[idx, 3]
    while parent != -1:
        d += 1
        parent = hier[parent, 3]
    return d

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

    #floodFill requires (h+2, w+2) mask; non-zero blocks filling
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    ff_mask[1:h+1, 1:w+1][edges > 0] = 1

    if seed is None:
        seed = _find_border_seed((edges > 0).astype(np.uint8))

    dummy = np.zeros((h, w), np.uint8)

    #Flags = connectivity(4) | (newMaskVal<<8) | MASK_ONLY
    flags = 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY

    #IMPORTANT: positional loDiff/upDiff (0, 0) – no keywords.
    cv.floodFill(dummy, ff_mask, seed, 0, 0, 0, flags)

    #Outside became 255 in mask (excluding 1px border); interiors stayed 0.
    inside = (ff_mask[1:h+1, 1:w+1] == 0).astype(np.uint8) * 255
    return inside


#---------- Contours + hierarchy ----------

def find_contours_tree(filled: np.ndarray, min_area: int = 150) -> Tuple[List[np.ndarray], np.ndarray]:
    cnts, hier = cv.findContours(filled, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return [], np.empty((0, 4), np.int32)

    #filter and rebuild hierarchy indexing
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


#---------- Drawing helpers ----------

def _draw_subtree_gray(
    mask: np.ndarray,
    contours: List[np.ndarray],
    hier: np.ndarray,
    root_idx: int,
    root_color: int = 255,
    hole_color: int = 0,
    *,
    line_type: int = cv.LINE_AA,
) -> None:
    """
    Draw a contour subtree into a single-channel mask.  Even depths (region interiors)
    are painted with "root_color" while odd depths (holes) are painted with
    "hole_color".  By default anti‑aliased edges are used; pass "line_type=cv.LINE_8"
    for crisp, non–anti‑aliased masks (useful for ID maps).
    """
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        color = root_color if (depth % 2 == 0) else hole_color
        #Note: thickness=-1 fills the entire contour region.
        cv.drawContours(
            mask,
            contours,
            i,
            color=int(color),
            thickness=-1,
            lineType=line_type,
        )
        child = hier[i, 2]
        while child != -1:
            stack.append((child, depth + 1))
            child = hier[child, 0]


def _draw_subtree_color(
    img: np.ndarray,
    contours: List[np.ndarray],
    hier: np.ndarray,
    root_idx: int,
    root_color: Tuple[int, int, int],
    hole_color: Tuple[int, int, int] = (0, 0, 0),
    *,
    line_type: int = cv.LINE_AA,
) -> None:
    """
    Draw a contour subtree into a three‑channel image.  Even depths use
    "root_color"; odd depths use "hole_color".  Pass "line_type=cv.LINE_8"
    to disable anti‑aliasing when generating ID maps.
    """
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        color = root_color if (depth % 2 == 0) else hole_color
        cv.drawContours(
            img,
            contours,
            i,
            color=color,
            thickness=-1,
            lineType=line_type,
        )
        child = hier[i, 2]
        while child != -1:
            stack.append((child, depth + 1))
            child = hier[child, 0]


def export_shape_masks(
    contours: List[np.ndarray],
    hier: np.ndarray,
    shape_hw: Tuple[int, int],
    out_dir: Path,
    mask_mode: str = "top",
) -> List[Path]:
    """
    Export binary masks for contours.  mask_mode controls which contours are saved:
      "top": only top‑level contours (current behaviour)
      "even": only even‑depth contours (filled interiors, holes skipped)
      "all": every contour, including holes
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = shape_hw
    saved: List[Path] = []

    #select which contours to save
    if mask_mode == "top":
        idxs = [i for i in range(len(contours)) if hier[i, 3] == -1]
        draw = lambda m, i: _draw_subtree_gray(m, contours, hier, i, 255, 0, line_type=cv.LINE_8)
    elif mask_mode == "even":
        idxs = [i for i in range(len(contours)) if _depth(i, hier) % 2 == 0]
        draw = lambda m, i: cv.drawContours(m, contours, i, 255, thickness=-1, lineType=cv.LINE_8)
    elif mask_mode == "all":
        idxs = list(range(len(contours)))
        draw = lambda m, i: cv.drawContours(m, contours, i, 255, thickness=-1, lineType=cv.LINE_8)
    else:
        raise ValueError("mask_mode must be one of {'top','even','all'}")

    #sort by centroid for a deterministic ordering
    def centroid(i):
        M = cv.moments(contours[i])
        return (M["m01"]/(M["m00"]+1e-9), M["m10"]/(M["m00"]+1e-9))

    idxs.sort(key=centroid)

    for j, i in enumerate(idxs, start=1):
        m = np.zeros((h, w), np.uint8)
        draw(m, i)
        p = out_dir / f"shape_{(j-1):03d}.png"
        cv.imwrite(str(p), m)
        saved.append(p)
    return saved

def colorize_regions(contours: List[np.ndarray], hier: np.ndarray,
                     shape_hw: Tuple[int,int],
                     palette: Optional[List[Tuple[int,int,int]]] = None) -> np.ndarray:
    h, w = shape_hw
    color_map = np.zeros((h, w, 3), np.uint8)

    #determine top-level shapes in a stable order
    top = _stable_top_level_indices(hier, contours)
    n = len(top)

    #if no palette supplied, fall back to random
    if palette is None or len(palette) < n:
        if palette is None: palette = []
        needed = n - len(palette)
        rng = np.random.default_rng(42)
        palette += [tuple(int(x) for x in rng.integers(64, 256, size=3)) for _ in range(needed)]

    #draw each top-level with its colour; holes are black
    for j, i in enumerate(top):
        _draw_subtree_color(color_map, contours, hier, i, root_color=palette[j], hole_color=(0,0,0))
    return color_map



def draw_overlay(img_bgr: np.ndarray, contours: List[np.ndarray],
                 color=(0,255,0), thick=2) -> np.ndarray:
    overlay = img_bgr.copy()
    cv.drawContours(overlay, contours, -1, color, thick, lineType=cv.LINE_AA)
    return overlay

#---------- ID map and metadata exporters ----------

def export_background_and_idmap(
    contours: List[np.ndarray],
    hier: np.ndarray,
    shape_hw: Tuple[int, int],
    *,
    out_bg: str = "mask_background.png",
    out_idmap: str = "id_map.png",
) -> int:
    """
    Generate two raster outputs: "mask_background.png" and "id_map.png".

    * "mask_background.png" is a single‑channel image where all pixels
      belonging to any region are 0 and the background is 255.  This
      simplifies background detection on the front end.

    * "id_map.png" encodes the region id in the three RGB channels.  The
      encoding uses a 24‑bit integer: "R = (id >> 16) & 255",
      "G = (id >> 8) & 255", "B = id & 255".  Region ids start at
      1; id 0 represents background.  Use "cv.LINE_8" when drawing
      to avoid anti‑aliased edges.

    Returns the total number of top‑level regions.
    """
    h, w = shape_hw
    bg = np.full((h, w), 255, np.uint8)
    id_map = np.zeros((h, w, 3), np.uint8)
    top = _stable_top_level_indices(hier, contours)
    for j, i in enumerate(top, start=1):
        #create temp mask for region i
        m = np.zeros((h, w), np.uint8)
        _draw_subtree_gray(m, contours, hier, i, 255, 0, line_type=cv.LINE_8)
        bg[m > 0] = 0
        #encode j into B,G,R (OpenCV uses BGR ordering)
        r = (j >> 16) & 0xFF
        g = (j >> 8) & 0xFF
        b = j & 0xFF
        id_map[m > 0] = (b, g, r)
    cv.imwrite(out_bg, bg)
    cv.imwrite(out_idmap, id_map)
    return len(top)


def export_metadata(
    contours: List[np.ndarray],
    hier: np.ndarray,
    shape_hw: Tuple[int, int],
    *,
    out_json: str = "metadata.json",
    masks_dir: str = "shape_masks",
) -> None:
    """
    Write a JSON file describing each region.  The structure is:

    "`json
    {
      "version": "1.0",
      "dimensions": {"width": W, "height": H},
      "total_regions": N,
      "background_id": 0,
      "regions": [
        {
          "id": 1,
          "bbox": {"x": ..., "y": ..., "width": ..., "height": ...},
          "centroid": {"x": ..., "y": ...},
          "mask": "shape_000.png"
        },
        ...
      ]
    }
    "`
    The region ids correspond to the order produced by
    "_stable_top_level_indices()".  The "mask" field points to the
    corresponding binary mask file within "masks_dir".
    """
    h, w = shape_hw
    top = _stable_top_level_indices(hier, contours)
    regions = []
    for j, i in enumerate(top, start=1):
        #generate binary mask for bbox and centroid calculation
        m = np.zeros((h, w), np.uint8)
        _draw_subtree_gray(m, contours, hier, i, 255, 0, line_type=cv.LINE_8)
        #bounding box
        ys, xs = np.where(m > 0)
        if xs.size > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = {
                "x": x0,
                "y": y0,
                "width": x1 - x0 + 1,
                "height": y1 - y0 + 1,
            }
        else:
            bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
        #centroid (geometric center of mask)
        M = cv.moments(m, binaryImage=True)
        cx = float(M["m10"] / (M["m00"] + 1e-9))
        cy = float(M["m01"] / (M["m00"] + 1e-9))
        regions.append(
            {
                "id": j,
                "bbox": bbox,
                "centroid": {"x": cx, "y": cy},
                "mask": f"{masks_dir}/shape_{(j-1):03d}.png",
            }
        )
    meta = {
        "version": "1.0",
        "dimensions": {"width": w, "height": h},
        "total_regions": len(regions),
        "background_id": 0,
        "regions": regions,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def export_palette_json(
    palette_bgr: List[Tuple[int, int, int]],
    *,
    out_json: str = "palette.json",
) -> None:
    """
    Write a mapping of region id to RGB values.  "palette_bgr[j-1]"
    corresponds to region id "j".  OpenCV uses BGR ordering, so convert
    back to RGB for the JSON file.
    """
    #Map id -> color (in RGB)
    mapping = {
        int(i + 1): {"r": int(r), "g": int(g), "b": int(b)}
        for i, (b, g, r) in enumerate(palette_bgr)
    }
    data = {"background_id": 0, "map": mapping}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

#---------- Palette: MiniBatch K-Means in RGB (with OpenCV fallback) ----------

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
    X = _rgb_sample_grid(step=sample_step)  #float32 in [0,255]
    centers = None

    #Try scikit-learn MiniBatchKMeans (fast “k-batch-means”).
    try:
        from sklearn.cluster import MiniBatchKMeans  #type: ignore
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
        #Fallback: OpenCV kmeans (standard Lloyd's).
        X32 = X.astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _compact, _labels, centers = cv.kmeans(
            X32, K=n, bestLabels=None, criteria=criteria, attempts=6, flags=cv.KMEANS_PP_CENTERS
        )

    centers = np.clip(centers, 0, 255).astype(np.uint8)  #RGB uint8
    #Convert RGB to BGR for OpenCV drawing
    centers_bgr = [(int(c[2]), int(c[1]), int(c[0])) for c in centers]
    return centers_bgr


#---------- Main pipeline (this is the function your CLI calls) ----------

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
    mask_mode: str = "top",
    palette_mode: str = "kbatch",   #NEW: {"kbatch","random"}
    sample_step: int = 7,
):
    """Run: edges -> close gaps -> fill regions -> contours -> colour + export masks."""
    img = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #1) Edge map
    if have_edges is not None:
        edges = have_edges.astype(np.uint8)
    else:
        g = gray
        if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
            g = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = auto_canny(g, sigma=canny_sigma, aperture_size=3, L2=True)

    #2) Close small gaps (optional)
    ksize = gap_close if gap_close and gap_close > 1 else 2
    edges_closed = close_gaps(edges, ksize=ksize, shape=cv.MORPH_ELLIPSE, iters=1)

    #3) Regions from edges
    filled = fill_regions_from_edges(edges_closed, seed=None)

    if do_threshold:
        th = threshold_otsu(gray)
        filled = cv.bitwise_and(filled, th)

    #4) Contours + hierarchy
    contours, hierarchy = find_contours_tree(filled, min_area=min_area)

    #Build palette if requested
    palette = None
    if palette_mode.lower() == "kbatch":
        n_top = sum(1 for i in range(len(contours)) if hierarchy[i, 3] == -1)
        if n_top > 0:   palette = palette_kbatchmeans_rgb(n_top, sample_step=sample_step)

    color_map = colorize_regions(contours, hierarchy, filled.shape, palette=palette)
    overlay   = draw_overlay(img, contours, color=(0,255,0), thick=2)

    #Write visual outputs
    cv.imwrite(out_overlay_path, overlay)
    cv.imwrite(out_filled_path, filled)
    cv.imwrite(out_color_path, color_map)

    #Export per‑region masks in stable order
    saved_masks = export_shape_masks(contours, hierarchy, filled.shape, Path(out_masks_dir), mask_mode=mask_mode)


    #Export background + ID map and metadata
    num_regions = export_background_and_idmap(
        contours,
        hierarchy,
        filled.shape,
        out_bg="mask_background.png",
        out_idmap="id_map.png",
    )
    export_metadata(
        contours,
        hierarchy,
        filled.shape,
        out_json="metadata.json",
        masks_dir=out_masks_dir,
    )
    #If palette is defined, persist it so the web app can use the same colours
    if palette is not None:
        export_palette_json(palette, out_json="palette.json")

    return edges_closed, filled, color_map, contours, hierarchy, saved_masks
#---------- CLI ----------

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
        description="Edge -> Region -> Colour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="Path to input image", default=r"\..")
    parser.add_argument("--image", dest="image_opt", help="Alternative to positional image path")
    """        out_overlay_path=args.overlay,
        out_filled_path=args.filled,
        out_color_path=args.color,"""
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
    parser.add_argument("--mask_mode",choices=["top", "even", "all"],default="top",
                        help="Which contours to export: top-level only, even depths or all contours.")

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
        palette_mode=args.palette_mode,  #NEW
        sample_step=args.sample_step,  #NEW
    )

    print(f"[OK] Saved:\n  • {args.overlay}\n  • {args.filled}\n  • {args.color}\n"
          f"  • {len(saved)} masks in '{args.masks_dir}/'")
    print(f"Contours kept: {len(contours)}")
