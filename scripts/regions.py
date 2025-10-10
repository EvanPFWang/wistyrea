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
from typing import List, Tuple, Optional, Sequence, Dict, Union

import cv2 as cv
import numpy as np
import json

from scipy.stats import alpha


def _repo_root_from_this_file() -> Path:
    #assumes this file lives at repo_root/scripts/regions.py (adjust .. count if needed)
    return Path(__file__).resolve().parents[1]

def _abs_out(p: Union[str, Path], base: Path) -> Path:
    """
    If p is absolute (or drive-rooted on Windows like '\\foo'), use it as-is,
    otherwise write under 'base'.
    """
    p = Path(p)
    return p if p.is_absolute() else (base / p)
def to_rgba(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr,  cv.COLOR_BGRA2RGBA)

def to_bgra(img_rgb: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_rgb, cv.COLOR_RGBA2BGRA)

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
    root_colour: int = 255,
    hole_colour: int = 0,
    *,
    line_type: int = cv.LINE_AA,
) -> None:
    """
    Draw a contour subtree into a single-channel mask.  Even depths (region interiors)
    are painted with "root_colour" while odd depths (holes) are painted with
    "hole_colour".  By default anti‑aliased edges are used; pass "line_type=cv.LINE_8"
    for crisp, non–anti‑aliased masks (useful for ID maps).
    """
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        colour = root_colour if (depth % 2 == 0) else hole_colour
        #Note: thickness=-1 fills the entire contour region.
        cv.drawContours(mask,
            contours,
            i,
            color=int(colour),
            thickness=-1,
            lineType=line_type,
        )
        child = hier[i, 2]
        while child != -1:
            stack.append((child, depth + 1))
            child = hier[child, 0]


def _draw_subtree_colour(
    img: np.ndarray,
    contours: List[np.ndarray],
    hier: np.ndarray,
    root_idx: int,
    root_colour: Tuple[int, int, int],
    hole_colour: Tuple[int, int, int] = (0, 0, 0),
    *,
    line_type: int = cv.LINE_AA,
) -> None:
    """
    Draw a contour subtree into a three‑channel image.  Even depths use
    "root_colour"; odd depths use "hole_colour".  Pass "line_type=cv.LINE_8"
    to disable anti‑aliasing when generating ID maps.
    """
    stack = [(root_idx, 0)]
    while stack:
        i, depth = stack.pop()
        colour = root_colour if (depth % 2 == 0) else hole_colour
        cv.drawContours(
            img,
            contours,
            i,
            color=colour,
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
) -> Tuple[List[Path], Dict[int, int], List[int]]:
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
    elif mask_mode == ("even"):
        idxs = [i for i in range(len(contours)) if _depth(i, hier) % 2 == 0]
        draw = lambda m, i: cv.drawContours(m, contours, i, 255, thickness=-1, lineType=cv.LINE_8)
    elif mask_mode == "all":
        idxs = list(range(len(contours)))
        draw = lambda m, i: cv.drawContours(m, contours, i, 255, thickness=-1, lineType=cv.LINE_8)
    else:
        raise ValueError("mask_mode must be one of {'top','even','all'}")

    """def _centroid(i):
        m = cv.moments(contours[i]);
        x = (m["m10"] / (m["m00"] + 1e-9));
        y = (m["m01"] / (m["m00"] + 1e-9))
        return (x, y)"""
    original_idxs = np.array(idxs, dtype=int)
    centroids = [cv.moments(contours[i]) for i in original_idxs]
    cx = np.array([m["m10"] / (m["m00"] + 1e-9) for m in centroids])
    cy = np.array([m["m01"] / (m["m00"] + 1e-9) for m in centroids])


    cx0 = (w - 1) / 2.0;cy0 = (h - 1) / 2.0;d = np.hypot(cx - cx0, cy - cy0)
    sort_order = np.argsort(d,kind="stable")
    sorted_original_idxs = original_idxs[sort_order]#for stable sorting based on multiple keys, np.lexsort is the right tool.

    idxs    =   np.array(idxs).tolist()
    mapping_dict = {orig_idx: new_id for new_id, orig_idx in enumerate(sorted_original_idxs, start=1)}

    dtype   =   np.uint16   if len(idxs) >= 256 else np.uint8
    dtype   =   np.uint32 if len(idxs)>=256*256 else dtype
    unified =   np.zeros((h, w), dtype=dtype)
    bit_count_str   =   str(dtype)[3:]#"8","16","32"
    for j, i in enumerate(idxs, start=1):
        m = np.zeros((h, w), np.uint8)
        draw(m, i)  # draw the j‑th contour into m
        unified[m > 0] = j  # paint region ID into the unified mask

        # save individual mask maybe convert this part to 1-based
        p = out_dir / f"shape_{(j-1):03d}.png"
        cv.imwrite(str(p), m)
        saved.append(p)
    R = (unified        & 0xFF).astype(np.uint8)     # byte 0 (LSB)
    G = ((unified >> 8) & 0xFF).astype(np.uint8)     # byte 1
    B = ((unified >>16) & 0xFF).astype(np.uint8)     # byte 2 (MSB)
    alpha   =   ((unified >>24) & 0xFF).astype(np.uint8)
    unified_bgra = np.dstack([B, G, R, alpha ])
    unified_rgba = np.dstack([R, G, B,   alpha])
    # after loop finishes, save unified mask in the same directory
    cv.imwrite(str(out_dir / "unified_mask.png"), unified_bgra)
    # write little-endian bytes for web use (u8/u16/u32)
    if dtype == np.uint8:raw = unified.astype("|u1", copy=False)
    elif dtype == np.uint16:raw = unified.astype("<u2", copy=False)
    else:raw = unified.astype("<u4", copy=False)
    with open(out_dir / f"unified_mask.u{bit_count_str}", "wb") as f:
        f.write(raw.tobytes())
    # return masks, mapping (orig_idx -> 1-based id), and the sorted contour indices
    return saved, mapping_dict, idxs

def colorize_regions(contours: List[np.ndarray], hier: np.ndarray,
                     shape_hw: Tuple[int,int],
                     palette: Optional[List[Tuple[int,int,int]]] = None) -> np.ndarray:
    h, w = shape_hw
    colour_map = np.zeros((h, w, 3), np.uint8)

    #determine top-level shapes in a stable order
    top = _stable_top_level_indices(hier, contours,shape_hw)
    n = len(top)

    #if no palette supplied, fall back to random
    if palette is None or len(palette) < n:
        if palette is None: palette = []
        needed = n - len(palette)
        rng = np.random.default_rng(42)
        palette += [tuple(int(x) for x in rng.integers(64, 256, size=3)) for _ in range(needed)]  #RGB
    palette_bgr = [(r2, g2, b2)[::-1] for (r2, g2, b2) in palette]
    #draw each top-level with its colour; holes are black
    for j, i in enumerate(top):
        _draw_subtree_colour(colour_map, contours, hier, i, root_colour=palette_bgr[j], hole_colour=(0,0,0))#draws with bgr palette
    return colour_map, palette_bgr#colour_map is palette but cnetroid ordered
    #conv to rgb @savetime


def draw_overlay(img_bgr: np.ndarray, contours: List[np.ndarray],
                 colour=(0,255,0), thick=2) -> np.ndarray:
    overlay = img_bgr.copy()
    cv.drawContours(overlay, contours, -1, colour, thick, lineType=cv.LINE_AA)
    return overlay

#---------- ID map and metadata exporters ----------

def export_background_and_idmap(
    contours: List[np.ndarray],
    hier: np.ndarray,
    shape_hw: Tuple[int, int],
    idxs:List[int],
    *,
    out_bg: str = "\public\data\mask_background.png",
    out_idmap: str = "\public\data\id_map.png",
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
    id_map_rgba = np.zeros((h, w, 4), np.uint8)
    id_map_bgra = np.zeros((h, w, 4), np.uint8)
    #top = _stable_top_level_indices(hier, contours)
    for j, i in enumerate(idxs, start=1):
        #create temp mask for region i
        m = np.zeros((h, w), np.uint8)
        _draw_subtree_gray(m, contours, hier, i, 255, 0, line_type=cv.LINE_8)
        #encode j into B,G,R (OpenCV uses BGR ordering)
        select = m > 0
        bg[select] = 0

        # pack j into (R,G,B) so id = R + (G<<8) + (B<<16)
        R =  (j       ) & 0xFF
        G =  (j >> 8  ) & 0xFF
        B =  (j >> 16 ) & 0xFF
        alpha   =   (j >> 24 ) & 0xFF

        id_map_rgba[select, 0] = R
        id_map_rgba[select, 1] = G
        id_map_rgba[select, 2] = B
        id_map_rgba[select, 3] = alpha
    cv.imwrite(out_bg, bg)
    cv.imwrite(out_idmap, to_bgra(id_map_rgba))
    return idxs


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
                ""
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
    palette: List[Tuple[int, int, int]],
        colourmap,
    *,
    out_json: str = "palette.json",#this will be in bgra

    mapping_dict:  Optional[Dict[int, int]] = None,
    palette_keys: Optional[Sequence[int]] = None,  # when palette is keyed by orig_idx, pass that list here
) -> None:
    """
    palette keyed by new_id (default): palette[j-1] is colour for region id j.
    palette keyed by orig_idx: pass `palette_keys` (a sequence of original contour indices in palette order)
        and `mapping_dict={orig_idx->new_id}`; we'll remap to new ids.

    corresponds to region id "j".  DOESNT DO THIS ```OpenCV uses BGR ordering, so convert
    back to RGB for the JSON file```.
    region ids 1-based and colors RGB triplets.
    """
    out_colourMap: str = "colourmap.json"
    #takes in rgb palette
    #Map id -> colour (in RGB)
    #[POST - HELLO CHECK to see if indexes

    if palette_keys is None:
        # Palette already in new-id order (1..N)
        new_to_rgb = {j: {"r": int(r), "g": int(g), "b": int(b)}
                      for j, (r, g, b) in enumerate(palette, start=1)}
    else:
        if mapping_dict is None:
            raise ValueError("palette_keys provided but mapping_dict is None")
        if len(palette_keys) != len(palette):
            raise ValueError("palette_keys and palette must have same length")
        new_to_rgb = {}
        for (orig_idx, (r, g, b)) in zip(palette_keys, palette):
            new_id = mapping_dict.get(orig_idx)
            if new_id is None:
                #skip or raise; choose raise for early surfacing of mismatches
                raise KeyError(f"orig_idx {orig_idx} missing in mapping_dict")
            new_to_rgb[int(new_id)] = {"r": int(r), "g": int(g), "b": int(b)}

    data = {"background_id": 0, "map": new_to_rgb}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    #keep colourmap sidecar
    with open("colourmap.json", "w", encoding="utf-8") as f:
        json.dump(colourmap, f, ensure_ascii=False, indent=2)



def _stable_top_level_indices(
    hier: np.ndarray,
    contours: List[np.ndarray],
    shape_hw: Tuple[int, int],
    *,
    return_mapping: bool = False,
) -> Union[List[int], Tuple[List[int], Dict[int, int]]]:
    """
    Deterministic order for *top-level* shapes:
    sort by Euclidean distance from the image center (nearest first).

    If return_mapping=True, also return {original_contour_idx -> 1-based new id}.
    """
    h, w = shape_hw

    #top-level contours: parent == -1
    #(OpenCV hierarchy layout: [next, prev, first_child, parent])
    orig_idxs = np.where(hier[:, 3] == -1)[0]
    if orig_idxs.size == 0:
        return ([], {}) if return_mapping else []

    #centroids via image moments: Cx = m10/m00, Cy = m01/m00
    cx = np.empty(orig_idxs.size, dtype=float)
    cy = np.empty(orig_idxs.size, dtype=float)
    for k, i in enumerate(orig_idxs):
        M = cv.moments(contours[i])
        cx[k] = M["m10"] / (M["m00"] + 1e-9)
        cy[k] = M["m01"] / (M["m00"] + 1e-9)

    #image center
    cx0 = (w - 1) / 2.0
    cy0 = (h - 1) / 2.0

    #Euclidean distance to center (numerically well-behaved)
    d = np.hypot(cx - cx0, cy - cy0)

    #nearest-to-center first; stable keep ties deterministic
    order = np.argsort(d, kind="stable")
    idxs = orig_idxs[order].tolist()

    if return_mapping:
        mapping_dict = {orig_idx: new_id for new_id, orig_idx in enumerate(idxs, start=1)}
        return idxs, mapping_dict
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
    Returns list of (R,G,B) tuples for OpenCV drawing.
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
    #no longeer Convert RGB to BGR for OpenCV drawing
    centers_rgb = np.clip(centers, 0, 255).astype(np.uint8) # The clustered colors are in RGB
    centers_bgr = [(int(c[2]), int(c[1]), int(c[0])) for c in centers_rgb] # Explicitly convert RGB to BG
    return centers_rgb


#---------- Main pipeline (this is the function your CLI calls) ----------

def process_image(
    img_path: str,
    out_overlay_path: str = "contours_overlay.png",
    out_filled_path: str  = "filled_mask.png",
    out_colour_path: str   = "coloured_regions.png",
    out_masks_dir: str    = "shape_masks",
    have_edges: Optional[np.ndarray] = None,
    min_area: int = 20,
    gap_close: int = 1,
    do_threshold: bool = False,
    blur_ksize: int = 1,
    canny_sigma: float = 0.33,
    mask_mode: str = "all",
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

    #2) Close small gaps (optional) 1 disables closing;
        # >=3 performs morphological closing with that kernel
    ksize = gap_close if gap_close and gap_close > 1 else 1
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

    repo_root = _repo_root_from_this_file()
    data_dir = (repo_root / "public" / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    #masks_dir: honor absolute paths; otherwise force under public/data
    masks_dir_path = _abs_out(out_masks_dir, data_dir)
    masks_dir_path.mkdir(parents=True, exist_ok=True)

    saved_masks,ordered_palette_to_centroid_ordering_dict,original_idxs = export_shape_masks(
        contours=contours,
        hier=hierarchy,
        shape_hw=filled.shape,
        out_dir=masks_dir_path,
        mask_mode=mask_mode,
    )

    colour_map_bgr,palette_bgr   =   colorize_regions(contours, hierarchy, filled.shape, palette=palette)
    colour_map_rgb = cv.cvtColor(colour_map_bgr, cv.COLOR_BGR2RGB)
    overlay   = draw_overlay(img, contours, color=(0,255,0), thick=2)



    #Normalize preview outputs too: if caller passed bare names, send to public/data
    overlay_path = _abs_out(out_overlay_path, data_dir)
    filled_path = _abs_out(out_filled_path, data_dir)
    colour_path = _abs_out(out_colour_path, data_dir)

    # Write visual outputs (anchored)
    cv.imwrite(str(overlay_path), to_rgba(overlay))
    cv.imwrite(str(filled_path), filled)
    cv.imwrite(str(colour_path), colour_map_rgb)


    masks_dir_path = data_dir / "shape_masks"
    masks_dir_path.mkdir(parents=True, exist_ok=True)



    try:
        #pref returned order else fallback to alphabetical in folder
        mask_paths = list(saved_masks) if saved_masks else sorted(masks_dir_path.glob("*.png"))
        for idx, p in enumerate(mask_paths):
            p = Path(p)
            target = masks_dir_path / f"shape_{idx:03d}.png"
            if p.resolve() != target.resolve():
                if target.exists():
                    target.unlink()
                p.rename(target)
    except Exception as e:
        print(f"[warn] rename masks: {e}")



    print(f"Shape masks are being exported to: {masks_dir_path}")

    #Export background + ID map and metadata
    regions = export_background_and_idmap(
        contours,
        hierarchy,
        filled.shape,
        saved_masks,
        out_bg=str(data_dir / "mask_background.png"),
        out_idmap=str(data_dir / "coloured_map.png"),
    )
    export_metadata(
        contours,
        hierarchy,
        filled.shape,
        out_json=str(data_dir / "metadata.json"),
        masks_dir=masks_dir_path.name if masks_dir_path.is_absolute() else str(Path("shape_masks")),)
    #If palette is defined, persist it so the web app can use the same colours
    if palette is not None:#palette is RGB
        #COME BACK HERE for idxs being a dict from originally ordered palette
        export_palette_json(palette,colour_map_rgb,
                            out_json=str(data_dir / "palette.json"),
                            mapping_dict=ordered_palette_to_centroid_ordering_dict)#takes in rgb
    return edges_closed, filled, colour_map_rgb, contours, hierarchy, saved_masks
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("image", nargs="?", help="Path to input image", default=r"\..")
    parser.add_argument("--image", dest="image_opt", help="Alternative to positional image path")
    """        out_overlay_path=args.overlay,
        out_filled_path=args.filled,
        out_colour_path=args.color,"""
    parser.add_argument("--overlay", default="contours_overlay.png", help="Output: contours overlay (BGR)")
    parser.add_argument("--filled",  default="filled_mask.png",      help="Output: filled binary mask")
    parser.add_argument("--color",   default="coloured_regions.png", help="Output: per-region coloured map")
    parser.add_argument("--masks_dir", default="public\data\shape_masks",        help="Output dir for per-region masks")
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
    edges, filled, colour_map, contours, hierarchy, saved = process_image(
        img_path, out_overlay_path=args.overlay,
        out_filled_path=args.filled, out_colour_path=args.color,
        out_masks_dir=args.masks_dir, have_edges=have_edges,
        min_area=args.min_area, gap_close=args.gap_close,
        do_threshold=args.threshold, blur_ksize=args.blur,
        canny_sigma=args.canny_sigma,palette_mode=args.palette_mode,  #NEW
        sample_step=args.sample_step,)  #NEW
    print(f"[OK] Saved:\n  • {args.overlay}\n  • {args.filled}\n  • {args.color}\n"
          f"  • {len(saved)} masks in '{args.masks_dir}/'")
    print(f"Contours kept: {len(contours)}")