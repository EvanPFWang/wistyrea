from __future__ import annotations


def _sorted_original_from_mapping(mapping_dict: Dict) -> List[int]:
    items = [(int(k),int(v)) for k,v in mapping_dict.items()]
    items.sort(key=lambda kv: kv[1])
    new_ids = [v for _,v in items]
    if new_ids != list(range(1,len(new_ids) + 1)):  #sanity check
        raise ValueError("mapping_dict new_ids must be contiguous 1..N with no gaps/duplicates")
    return [orig for orig,_ in items]


_tile_px = 8


def export_metadata(
        contours: List[np.ndarray],
        hier: np.ndarray,
        shape_hw: Tuple[int,int],

        *,
        sorted_idxs: Optional[List[int]] = None,
        mapping_dict_to_morton_sort: Optional[Dict[int,int]] = None,
        out_json: str = "metadata.json",
        masks_dir: str = "shape_masks",) -> None:
    """
    write JSON file describing each region with structure:

    "`json
    {
      "version": "1.0",
      "dimensions": {"width": W,"height": H},
      "total_regions": N,
      "background_id": 0,
      "regions": [
        {
          "id": 1,
          "bbox": {"x": ...,"y": ...,"width": ...,"height": ...},
          "centroid": {"x": ...,"y": ...},
          "mask": "shape_000.bin"
        },
        ...
      ]
    }
    "`
    region ids correspond to  order produced by
    mapping_dict IN export_shape_,,,,that uses reIDX    "mask" field points to 
    corresponding binary mask file within "masks_dir".
    """
    h,w = shape_hw

    #normalize hierarchy shape if needed
    if hier is not None and hier.ndim == 3:
        hier = hier[0]

    #mapping_dict: {orig_idx: new_id} where new_id is 1..N

    #sorted_idxs = [orig for orig,new_id in sorted(mapping_dict.items(),key=lambda kv: kv[1])]
    if sorted_idxs is None:
        if mapping_dict_to_morton_sort is None:
            raise ValueError("Provide either sorted_original_idxs or mapping_dict")
        sorted_idxs = _sorted_original_from_mapping(mapping_dict_to_morton_sort)

    if mapping_dict_to_morton_sort is None:
        mapping_dict_to_morton_sort = {int(orig): int(j) for j,orig in enumerate(sorted_idxs,start=1)}

    #top,mapping_dict = _stable_top_level_indices(hier,contours,shape_hw,return_mapping=True)
    regions = []
    for j,i in enumerate(sorted_idxs,start=1):
        #generate binary mask for bbox and centroid calculation
        m = np.zeros((h,w),np.uint8)
        _draw_subtree_gray(m,contours,hier,int(i),255,0,line_type=cv.LINE_8)
        #bounding box
        ys,xs = np.where(m > 0)
        if xs.size > 0:
            x0,x1 = int(xs.min()),int(xs.max())
            y0,y1 = int(ys.min()),int(ys.max())
            bbox = {
                "x": x0,
                "y": y0,
                "width": x1 - x0 + 1,
                "height": y1 - y0 + 1,
            }
        else:
            bbox = {"x": 0,"y": 0,"width": 0,"height": 0}
        #centroid (geometric center of mask)
        M = cv.moments(m,binaryImage=True)
        cx = float(M["m10"] / (M["m00"] + 1e-9))
        cy = float(M["m01"] / (M["m00"] + 1e-9))
        regions.append(
            {
                "id": j,
                "orig_contour_idx": int(i), #debugging / traceability
                "bbox": bbox,
                "centroid": {"x": cx,"y": cy},
                #reference RLE-encoded mask (.bin) instead of old PNG mask
                "mask": f"{masks_dir}/shape_{(j - 1):03d}.bin",
            }
        )
    meta = {
        "version": "1.0",
        "dimensions": {"width": w,"height": h},
        "total_regions": len(regions),
        "background_id": 0,
        "regions": regions,
        "mapping_centroid_ordering": mapping_dict_to_morton_sort
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True,exist_ok=True)
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)


def export_palette_json(
        palette: List[Tuple[int,int,int]],
        colourmap,
        *,
        out_json: str = "palette.json", #this will be in bgra

        mapping_dict: Optional[Dict[int,int]] = None,
        palette_keys: Optional[Sequence[int]] = None, #when palette is keyed by orig_idx,pass that list here
) -> None:
    """
    palette keyed by new_id (default): palette[j-1] is colour for region id j.
    palette keyed by orig_idx: pass `palette_keys` (a sequence of original contour indices in palette order)
        and `mapping_dict={orig_idx->new_id}`; we'll remap to new ids.

    corresponds to region id "j".  DOESNT DO THIS ```OpenCV uses BGR ordering,so convert
    back to RGB for
     JSON file```.
    region ids 1-based and colors RGB triplets.
    """
    out_colourMap: str = "colourmap.json"
    #takes in rgb palette
    #Map id -> colour (in RGB)
    #[POST - HELLO CHECK to see if indexes

    if palette_keys is None:
        #Palette already in new-id order (1..N)
        new_to_rgb = {j: {"r": int(r),"g": int(g),"b": int(b)}
                      for j,(r,g,b) in enumerate(palette,start=1)}
    else:
        if mapping_dict is None:
            raise ValueError("palette_keys provided but mapping_dict is None")
        if len(palette_keys) != len(palette):
            raise ValueError("palette_keys and palette must have same length")
        new_to_rgb = {}
        for (orig_idx,(r,g,b)) in zip(palette_keys,palette):
            new_id = mapping_dict.get(orig_idx)
            if new_id is None:
                #skip or raise; choose raise for early surfacing of mismatches
                raise KeyError(f"orig_idx {orig_idx} missing in mapping_dict")
            new_to_rgb[int(new_id)] = {"r": int(r),"g": int(g),"b": int(b)}

    data = {"background_id": 0,"map": new_to_rgb}
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    #keep colourmap sidecar
    with open("colourmap.json","w",encoding="utf-8") as f:
        json.dump(colourmap,f,ensure_ascii=False,indent=2)


#python regions.py "C:\Users\User\Documents\GitHub\wistyrea\Mural_Crown_of_Italian_City.svg.png"   --blur 1 --close 1 --min_area 5 --canny_sigma 0.25 --mask_mode "all" --palette_mode kbatch --sample_step 7

#python regions.py "Mural_Crown_of_Italian_City.svg.png"   --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7 --debug

"""Canny sigma:--canny_sigma 0.25 is good start for clear line art; 0.33–0.5 for softer edges. (auto-thresholding strategy is solid; it’s common practice.)

Gap closing: --close 1 = off; try 3 if tiny edge breaks,but beware over-merging (it reduces region count).

Min area: keep 20,or make it image-scale aware: max(20,int(0.00002 * H * W)).

AA policy: onlyweb-facing assets (id_map.png,shape_*.png) must be LINE_8; keep LINE_AA for overlay and coloured_regions.png for visual quality. (OpenCV docs showdefault lineType and alternatives


python3 regions.py "Mural_Crown_of_Italian_City.svg.png" --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7
-> Produces: contours_overlay.png,filled_mask.png,coloured_regions.png,mask_background.png,id_map.png,metadata.json,palette.json,and shape_masks/shape_000.png ...

Onweb: load id_map.png into hidden canvas (no scaling),use coalesced events inside requestAnimationFrame,and change UI only when id changes. (If you ever offload to worker,use OffscreenCanvas.)

If you want me to tailor tiny diffs for your exact file names/paths (interactive-svg-mural.html,mural-typescript.ts,build script),paste those snippets and I’ll markexact insert/replace lines.

Sources


"""
import os,sys,glob
from pathlib import Path

from typing import List,Tuple,Optional,Sequence,Dict,Union
import cv2 as cv
import numpy as np

import json

from scipy.stats import alpha


def _part1by1(n: np.ndarray) -> np.ndarray:
    n = n & 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def _morton2d(qx: np.ndarray,qy: np.ndarray) -> np.ndarray:
    qx = qx.astype(np.uint32,copy=False)
    qy = qy.astype(np.uint32,copy=False)
    return _part1by1(qx) | (_part1by1(qy) << 1)


#tile_px default to either 4 or 16
def _spatial_reIDX(contours,idxs,img_w,img_h,tile_px):
    original_idxs = np.array(idxs,dtype=np.int32)

    #robust centroid (fallback to bbox center if moments are degenerate)
    cx = np.empty(len(original_idxs),dtype=np.float64)
    cy = np.empty(len(original_idxs),dtype=np.float64)
    area = np.empty(len(original_idxs),dtype=np.float64)

    tile_px = int(tile_px)
    if tile_px <= 0:
        raise ValueError("tile_px must be > 0")

    for k,i in enumerate(original_idxs):
        m = cv.moments(contours[i])
        area[k] = cv.contourArea(contours[i])
        if abs(m["m00"]) > 1e-9:
            cx[k] = m["m10"] / m["m00"]
            cy[k] = m["m01"] / m["m00"]
        else:
            x,y,w,h = cv.boundingRect(contours[i])
            cx[k] = x + 0.5 * w
            cy[k] = y + 0.5 * h
    idX = np.clip(np.rint(cx),0,img_w - 1).astype(np.uint32)
    idY = np.clip(np.rint(cy),0,img_h - 1).astype(np.uint32)

    tileX = (idX // tile_px).astype(np.uint32)
    tileY = (idY // tile_px).astype(np.uint32)

    #local coords in tile [0..tile_px-1]
    #lx = (idX   - tileX * tile_px).astype(np.uint32)
    #ly = (idY   - tileY * tile_px).astype(np.uint32)
    lx = (idX % tile_px).astype(np.uint32)
    ly = (idY % tile_px).astype(np.uint32)

    mkey = _morton2d(lx,ly)
    aux = (original_idxs,mkey,tileX,tileY)
    order = np.lexsort(aux)  #USEFOLLOWING IF ANY JITTER

    #order = np.lexsort((original_idxs,area,mkey,tileX,tileY))
    tile_then_morton_sorted_idxs = original_idxs[order]

    opencv_contour_to_new_idx = np.zeros(len(original_idxs),dtype=np.uint32)
    opencv_contour_to_new_idx[tile_then_morton_sorted_idxs] = (
        np.arange(1,len(tile_then_morton_sorted_idxs) + 1,dtype=np.uint32))
    return tile_then_morton_sorted_idxs,opencv_contour_to_new_idx,aux


def _repo_root_from_this_file() -> Path:
    #assumes this file lives at repo_root/scripts/regions.py (adjust .. count if needed)
    return Path(__file__).resolve().parents[1]


def _abs_out(p: Union[str,Path],base: Path) -> Path:
    """
    If p is absolute (or drive-rooted on Windows like '\\foo'),use it as-is,
    otherwise write under 'base'.
    """
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def to_rgba(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr,cv.COLOR_BGRA2RGBA)


def to_bgra(img_rgb: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_rgb,cv.COLOR_RGBA2BGRA)


def auto_canny(gray: np.ndarray,sigma: float = 0.33,
               aperture_size: int = 3,L2: bool = True) -> np.ndarray:
    v = float(np.median(gray))
    lo = int(max(0,(1.0 - sigma) * v))
    hi = int(min(255,(1.0 + sigma) * v))
    return cv.Canny(gray,lo,hi,apertureSize=aperture_size,L2gradient=L2)


def close_gaps(edges: np.ndarray,ksize: int = 3,
               shape: int = cv.MORPH_ELLIPSE,iters: int = 1) -> np.ndarray:
    if ksize is None or ksize <= 1:
        return edges
    k = cv.getStructuringElement(shape,(ksize,ksize))
    return cv.morphologyEx(edges,cv.MORPH_CLOSE,k,iterations=iters)


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    _,th = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    return th


def _depth(idx: int,hier: np.ndarray) -> int:
    """Returndepth of contour `idx` in hierarchy."""
    d = 0
    parent = hier[idx,3]
    while parent != -1:
        d += 1
        parent = hier[parent,3]
    return d


def _find_border_seed(blockers01: np.ndarray) -> Tuple[int,int]:
    """Return background pixel on  border; fallback (0,0)."""
    h,w = blockers01.shape
    for x in range(w):
        if blockers01[0,x] == 0: return (x,0)
        if blockers01[h - 1,x] == 0: return (x,h - 1)
    for y in range(h):
        if blockers01[y,0] == 0: return (0,y)
        if blockers01[y,w - 1] == 0: return (w - 1,y)
    return (0,0)


#---------- Contours + hierarchy ----------

def fill_regions_from_edges(edges: np.ndarray,seed: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """Return uint8 {0,255} of interior regions delimited by edges (treated as barriers)."""
    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    h,w = edges.shape[:2]

    #floodFill requires (h+2,w+2) mask; non-zero blocks filling
    ff_mask = np.zeros((h + 2,w + 2),np.uint8)
    ff_mask[1:h + 1,1:w + 1][edges > 0] = 1

    if seed is None:
        seed = _find_border_seed((edges > 0).astype(np.uint8))

    dummy = np.zeros((h,w),np.uint8)

    #Flags = connectivity(4) | (newMaskVal<<8) | MASK_ONLY
    flags = 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY

    #IMPORTANT: positional loDiff/upDiff (0,0) – no keywords.
    cv.floodFill(dummy,ff_mask,seed,0,0,0,flags)

    #Outside became 255 in mask (excluding 1px border); interiors stayed 0.
    inside = (ff_mask[1:h + 1,1:w + 1] == 0).astype(np.uint8) * 255
    return inside


#---------- Drawing helpers ----------

def find_contours_tree(filled: np.ndarray,min_area: int = 150) -> Tuple[List[np.ndarray],np.ndarray]:
    cnts,hier = cv.findContours(filled,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return [],np.empty((0,4),np.int32)

    #filter and rebuild hierarchy indexing
    keep_idx = [i for i,c in enumerate(cnts) if cv.contourArea(c) >= float(min_area)]
    keep = [cnts[i] for i in keep_idx]
    if not keep:
        return [],np.empty((0,4),np.int32)

    oldh = hier[0]
    idx_map = {old: new for new,old in enumerate(keep_idx)}
    new_h = np.full((len(keep),4),-1,dtype=np.int32)
    for new_i,old_i in enumerate(keep_idx):
        nxt,prv,ch,par = oldh[old_i]
        new_h[new_i,0] = idx_map.get(nxt,-1)
        new_h[new_i,1] = idx_map.get(prv,-1)
        new_h[new_i,2] = idx_map.get(ch,-1)
        new_h[new_i,3] = idx_map.get(par,-1)
    return keep,new_h


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
    Draw contour subtree into single-channel mask.  Even depths (region interiors)
    are painted with "root_colour" while odd depths (holes) are painted with
    "hole_colour".  By default anti‑aliased edges are used; pass "line_type=cv.LINE_8"
    for crisp,non–anti‑aliased masks (useful for ID maps).
    """
    stack = [(root_idx,0)]
    while stack:
        i,depth = stack.pop()
        colour = root_colour if (depth % 2 == 0) else hole_colour
        #Note: thickness=-1 fills  entire contour region.
        cv.drawContours(mask,
                        contours,
                        i,
                        color=int(colour),
                        thickness=-1,
                        lineType=line_type,
                        )
        child = hier[i,2]  #child = hier[0,i,2] if hier.ndim > 1 else hier[i,2]
        while child != -1:
            stack.append((child,depth + 1))
            child = hier[child,0]  #child = hier[0,child,0] if hier.ndim > 1 else hier[child,0]


def _draw_subtree_colour(
        img: np.ndarray,
        contours: List[np.ndarray],
        hier: np.ndarray,
        root_idx: int,
        root_colour: Tuple[int,int,int],
        hole_colour: Tuple[int,int,int] = (0,0,0),
        *,
        line_type: int = cv.LINE_AA,
) -> None:
    """
    Draw contour subtree into three‑channel image.  Even depths use
    "root_colour"; odd depths use "hole_colour".  Pass "line_type=cv.LINE_8"
    to disable anti‑aliasing when generating ID maps.
    """
    stack = [(root_idx,0)]
    while stack:
        i,depth = stack.pop()
        colour = root_colour if (depth % 2 == 0) else hole_colour
        cv.drawContours(
            img,
            contours,
            i,
            color=colour,
            thickness=-1,
            lineType=line_type,
        )
        child = hier[i,2]
        while child != -1:
            stack.append((child,depth + 1))
            child = hier[child,0]


def _rle_encode_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Encode binary mask using run-length encoding.

    Mask flattened in row-major order and values are treated as 0 or 1.
    returned array begins with  start value (0 or 1) followed by the
    lengths of alternating runs of that value. All values encoded as
    32-bit unsigned integers.


    mask : np.ndarray
        2D uint8 array where non-zero pixels are considered 1 and zero pixels
        are considered 0.

    return 1D array of dtype np.uint32 containing [start_val,run_len1,run_len2,...].
    """
    #Flatten mask to 0/1 values in row-major order
    flat = (mask > 0).astype(np.uint8).reshape(-1)
    total = flat.size
    if total == 0:
        #Empty mask: start with 0 and no runs
        return np.array([0],dtype=np.uint32)
    start_val = np.uint32(flat[0])
    #Find change points where value switches between 0 and 1
    #np.diff yields non-zero where change occurs; positions refer to end of previous run
    changes = np.nonzero(np.diff(flat))[0] + 1
    #Calculate run lengths by taking differences between consecutive change points
    #Include start (0) and end (total)
    positions = np.concatenate(([0],changes,[total])).astype(np.int64)
    run_lengths = np.diff(positions).astype(np.uint32)
    #Prependstart value torun lengths
    return np.concatenate(([start_val],run_lengths))


def morton_reIDX_export_shape_masks(
        contours: List[np.ndarray],
        hier: np.ndarray,
        shape_hw: Tuple[int,int],
        out_dir: Path,
        mask_mode: str = "top",
        vectorization_support_bool: bool = False,
) -> Tuple[List[Path],Dict[int,int],List[int]]:
    """
    Spatially reindex contours using Morton ordering and export each selected region
    as run-length encoded binary mask (.bin)

    returned mapping
    dictionary maps original contour indices to new 1-based region IDs.

    contours : List[np.ndarray]
        Contours detected inimage.
    hier : np.ndarray
        Hierarchy array from cv.findContours.
    shape_hw : Tuple[int,int]
        Height and width ofimage (h,w).
    out_dir : Path
        Directory where.bin files will be written.
    mask_mode : str,optional
        Determines which contours to export: "top" for top-level contours,
        "even" for even-depth contours,or "all" for every contour. Default is "top".
    vectorization_support_bool : bool,optional
        Retained for backwards compatibility; unused in this implementation.

    returns list of paths towritten .bin files,mapping from original
        contour index to 1-based new ID,and list oforiginal contour
        indices selected for export

    Tuple[List[Path],Dict[int,int],List[int]]
    """
    out_dir.mkdir(parents=True,exist_ok=True)
    h,w = shape_hw
    saved: List[Path] = []

    if hier is None:
        return [],{},[]
    if hier.ndim == 3:
        hier = hier[0]

    #Select which contours to save based on mask_mode
    if mask_mode == "top":
        old_idxs = [i for i in range(len(contours)) if hier[i,3] == -1]
        draw = lambda m,i: _draw_subtree_gray(m,contours,hier,i,255,0,line_type=cv.LINE_8)
    elif mask_mode == "even":
        old_idxs = [i for i in range(len(contours)) if _depth(i,hier) % 2 == 0]
        draw = lambda m,i: cv.drawContours(m,contours,i,255,thickness=-1,lineType=cv.LINE_8)
    elif mask_mode == "all":
        old_idxs = list(range(len(contours)))
        draw = lambda m,i: cv.drawContours(m,contours,i,255,thickness=-1,lineType=cv.LINE_8)
    else:
        raise ValueError("mask_mode must be one of {'top','even','all'}")

    #Spatially reorder contours using Morton ordering
    sorted_idxs,_contour_to_new_idx,_aux = _spatial_reIDX(
        contours=contours,idxs=old_idxs,img_w=w,img_h=h,tile_px=_tile_px
    )

    #Build mapping from original contour index to new ID (1-based)
    mapping_dict = {int(orig_i): int(new_id) for new_id,orig_i in enumerate(sorted_idxs,start=1)}

    #Generate per-region binary masks and save as RLE-encoded .bin
    for j,orig_idx in enumerate(sorted_idxs,start=1):
        m = np.zeros((h,w),np.uint8)
        draw(m,int(orig_idx))
        #Encode mask to RLE binary data
        rle = _rle_encode_binary_mask(m)
        #Write RLE buffer to .bin file (little-endian 32-bit unsigned integers)
        p = out_dir / f"shape_{(j - 1):03d}.bin"
        rle_bytes = rle.astype("<u4",copy=False).tobytes()
        with open(p,"wb") as f:
            f.write(rle_bytes)
        saved.append(p)

    #Returnlist of saved .bin paths,mapping dictionary,andoriginal indices list
    return saved,mapping_dict,list(old_idxs)


def colorize_regions(contours: List[np.ndarray],hier: np.ndarray,
                     shape_hw: Tuple[int,int],mapping_dict_to_morton_sort,
                     #mapping is dict forcontours to their new indexes to be colorized in
                     palette: Optional[List[Tuple[int,int,int]]] = None) -> np.ndarray:
    h,w = shape_hw
    colour_map = np.zeros((h,w,3),np.uint8)

    if hier is not None and hier.ndim == 3: hier = hier[0]
    #determine top-level shapes in stable order
    if mapping_dict_to_morton_sort is None:
        all_idxs = list(range(len(contours)))
        sorted_idxs,mapping_dict_to_morton_sort = morton_order_for_idxs(
            contours=contours,idxs=all_idxs,img_w=w,img_h=h,tile_px=_tile_px,mapping_dict=None
        )
    else:
        sorted_idxs = _sorted_original_from_mapping(mapping_dict_to_morton_sort)

    n = len(sorted_idxs)
    if n == 0:
        return colour_map,[]

    if palette is None or len(palette) < n:  #mayb just regenerate kbatch
        palette = palette_kbatchmeans_rgb(n)
        if isinstance(palette,np.ndarray):
            palette = [tuple(int(x) for x in row) for row in palette.reshape(-1,3)]
    else:
        palette = list(palette)

    palette_bgr = [(b,g,r) for (r,g,b) in palette[:n]]

    for j,orig_idx in enumerate(sorted_idxs,start=1):
        cv.drawContours(
            colour_map,
            contours,
            int(orig_idx),
            color=palette_bgr[j - 1],
            thickness=-1,
            lineType=cv.LINE_AA,
        )  #LINE_8 if you need crisp edges
        #COME BACK HERE to check for rgb or bgr
    return colour_map,palette_bgr
    #conv to rgb @savetime


#---------- ID map and metadata exporters ----------

def draw_overlay(img_bgr: np.ndarray,contours: List[np.ndarray],
                 colour=(0,255,0),thick=2) -> np.ndarray:
    overlay = img_bgr.copy()
    cv.drawContours(overlay,contours,-1,colour,thick,lineType=cv.LINE_AA)
    return overlay


def morton_order_for_idxs(
        contours,
        idxs: List[int],
        img_w: int,
        img_h: int,
        tile_px: int,
        *,
        mapping_dict: Optional[Dict[int,int]] = None,
) -> Tuple[List[int],Dict[int,int]]:
    """Morton ordering:
      if mapping_dict is provided,reconstruct order from it (sanity checked)

      else compute Morton order from contours + idxs and build mapping_dict
    """
    if mapping_dict is not None:    return _sorted_original_from_mapping(mapping_dict),mapping_dict

    morton_sorted_idxs,contour_to_newid,_aux = _spatial_reIDX(contours=contours,idxs=idxs,
                                                                img_w=img_w,img_h=img_h,tile_px=_tile_px)
    sorted_original_idxs = morton_sorted_idxs.astype(int).tolist()

    #only include selected idxs in dict (clean + matches selection)
    mapping_dict = {int(orig): int(contour_to_newid[int(orig)]) for orig in idxs}
    return sorted_original_idxs,mapping_dict


def export_background_and_idmap(
        contours: List[np.ndarray],
        hier: np.ndarray,
        shape_hw: Tuple[int,int],
        mapping_dict: Dict[int,int],
        idxs: List[int],
        *,
        out_idmap: str = "\public\data\id_map.png",
) -> int:
    """
    generate colour-encoded ID map forgiven regions.

   returned image encodes each region's 1-based ID intothree
    channels such that ``id = R + (G << 8) + (B << 16)``.alpha channel
    remains unused in this implementation. Region ID 0 (background) is left
    at 0.

    contours : List[np.ndarray]
        Contours detected inimage.
    hier : np.ndarray
        Hierarchy array from cv.findContours.
    shape_hw : Tuple[int,int]
        Height and width ofimage (h,w).
    mapping_dict : Dict[int,int]
        Mapping from original contour indices to new 1-based region IDs.
    idxs : List[int]
        List of original contour indices to include inID map.
    out_idmap : str,optional
        Path to saveresulting ID map PNG. Default is ``"\public\data\id_map.png"``.

    returns number of regions encoded inID map.
    """
    h,w = shape_hw
    id_map_rgba = np.zeros((h,w,4),np.uint8)

    #Resolvecontour indices according toprovided mapping_dict and
    #ensure they are within bounds and unique.
    remapped_idxs = [mapping_dict.get(i,i) for i in idxs] if mapping_dict else list(idxs)
    remapped_idxs = [i for i in remapped_idxs if 0 <= i < len(contours)]
    remapped_idxs = list(dict.fromkeys(remapped_idxs))

    for j,i in enumerate(remapped_idxs,start=1):
        m = np.zeros((h,w),np.uint8)
        _draw_subtree_gray(m,contours,hier,i,255,0,line_type=cv.LINE_8)
        select = m > 0
        #Pack j into (R,G,B) channels
        R = (j) & 0xFF
        G = (j >> 8) & 0xFF
        B = (j >> 16) & 0xFF
        alpha = (j >> 24) & 0xFF
        id_map_rgba[select,0] = R
        id_map_rgba[select,1] = G
        id_map_rgba[select,2] = B
        id_map_rgba[select,3] = alpha
    #WriteID map as BGRA for consistency with OpenCV
    cv.imwrite(out_idmap,to_bgra(id_map_rgba))
    return len(remapped_idxs)


def _rgb_sample_grid(step: int = 7,lo: int = 24,hi: int = 231) -> np.ndarray:
    """
    unif grid sample of RGB cube (uint8). `step` controls resolution:
    count ≈ ((hi-lo)/step + 1)^3. Defaults avoid near-black/near-white.
    """
    vals = np.arange(lo,hi + 1,step,dtype=np.uint8)
    r,g,b = np.meshgrid(vals,vals,vals,indexing="xy")
    X = np.stack([r,g,b],axis=-1).reshape(-1,3).astype(np.float32)
    return X


def palette_kbatchmeans_rgb(n: int,
                            sample_step: int = 7,
                            rng_seed: int = 42,
                            max_iter: int = 200,
                            batch_size: int = 2048) -> List[Tuple[int,int,int]]:
    """
    build n roughly evenly-spaced RGB colours by clustering uniform RGB sample.
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
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,50,0.5)
        _compact,_labels,centers = cv.kmeans(
            X32,K=n,bestLabels=None,criteria=criteria,attempts=6,flags=cv.KMEANS_PP_CENTERS
        )

    centers = np.clip(centers,0,255).astype(np.uint8)  #RGB uint8
    #no longeer Convert RGB to BGR for OpenCV drawing
    centers_rgb = np.clip(centers,0,255).astype(np.uint8)  #clustered colors are in RGB
    centers_bgr = [(int(c[2]),int(c[1]),int(c[0])) for c in centers_rgb]  #Explicitly convert RGB to BG
    return centers_rgb


#---------- Main pipeline (this isfunction your CLI calls) ----------

def process_image(
        img_path: str,
        out_overlay_path: str = "contours_overlay.png",
        out_filled_path: str = "filled_mask.png",
        out_colour_path: str = "coloured_regions.png",
        out_masks_dir: str = "shape_masks",
        have_edges: Optional[np.ndarray] = None,
        min_area: int = 20,
        gap_close: int = 1,
        do_threshold: bool = False,
        blur_ksize: int = 1,
        canny_sigma: float = 0.33,
        mask_mode: str = "all",
        palette_mode: str = "kbatch", #NEW: {"kbatch","random"}
        sample_step: int = 7,
):
    """Run: edges -> close gaps -> fill regions -> contours -> colour + export masks."""
    img = cv.imread(str(img_path),cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #1) Edge map
    if have_edges is not None:
        edges = have_edges.astype(np.uint8)
    else:
        g = gray
        if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
            g = cv.GaussianBlur(gray,(blur_ksize,blur_ksize),0)
        edges = auto_canny(g,sigma=canny_sigma,aperture_size=3,L2=True)

    #2) Close small gaps (optional) 1 disables closing;
    #>=3 performs morphological closing with that kernel
    ksize = gap_close if gap_close and gap_close > 1 else 1
    edges_closed = close_gaps(edges,ksize=ksize,shape=cv.MORPH_ELLIPSE,iters=1)

    #3) Regions from edges
    filled = fill_regions_from_edges(edges_closed,seed=None)

    if do_threshold:
        th = threshold_otsu(gray)
        filled = cv.bitwise_and(filled,th)

    #4) Contours + hierarchy

    #load +-40
    #opencv uses its OWN indices
    contours,hierarchy = find_contours_tree(filled,min_area=min_area)  #matched up 0-based idx
    #tree traversal of hier with BFS
    #concurrently replace each
    #Build palette if requested
    palette = None
    if palette_mode.lower() == "kbatch":
        n_top = sum(1 for i in range(len(contours)) if hierarchy[i,3] == -1)
        if n_top > 0:   palette = palette_kbatchmeans_rgb(n_top,sample_step=sample_step)

    repo_root = _repo_root_from_this_file()
    data_dir = (repo_root / "public" / "data").resolve()
    data_dir.mkdir(parents=True,exist_ok=True)

    #masks_dir: honor absolute paths; otherwise force under public/data
    masks_dir_path = _abs_out(out_masks_dir,data_dir)
    masks_dir_path.mkdir(parents=True,exist_ok=True)

    saved_masks,ordered_palette_to_centroid_ordering_dict,original_idxs = (
        morton_reIDX_export_shape_masks(
            contours=contours,
            hier=hierarchy,
            shape_hw=filled.shape,
            out_dir=masks_dir_path,
            mask_mode=mask_mode,
            vectorization_support_bool=False
        ))
    #conforms to ordered_palette_to_centroid_ordering_dict
    #We no longer generate coloured overlays or per-region coloured maps here.
    #Instead,onlyfilled mask and RLE-encoded shape masks are produced.

    #Normalize preview output path for filled mask
    filled_path = _abs_out(out_filled_path,data_dir)
    cv.imwrite(str(filled_path),filled)

    #Informuser whereshape masks have been exported
    print(f"Shape masks are being exported to: {masks_dir_path}")

    #Export ID map forfront end; do not generate separate background mask
    regions_count = export_background_and_idmap(
        contours,
        hierarchy,
        filled.shape,
        ordered_palette_to_centroid_ordering_dict,
        original_idxs,
        out_idmap=str(data_dir / "id_map.png"),
    )

    #Export metadata describingregions
    export_metadata(
        contours,
        hierarchy,
        filled.shape,
        mapping_dict_to_morton_sort=ordered_palette_to_centroid_ordering_dict,
        out_json=str(data_dir / "metadata.json"),
        masks_dir=masks_dir_path.name if masks_dir_path.is_absolute() else str(Path("shape_masks")),
    )

    #Persistpalette for use byweb app (if generated)
    if palette is not None:
        #We pass an empty list forcolourmap parameter since we no longer generate preview image
        export_palette_json(
            palette,
            [],
            out_json=str(data_dir / "palette.json"),
            mapping_dict=ordered_palette_to_centroid_ordering_dict,
        )

    return edges_closed,filled,contours,hierarchy,saved_masks


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
    parser.add_argument("image",nargs="?",help="Path to input image",default=r"\..")
    parser.add_argument("--image",dest="image_opt",help="Alternative to positional image path")
    """        out_overlay_path=args.overlay,
        out_filled_path=args.filled,
        out_colour_path=args.color,"""
    parser.add_argument("--overlay",default="contours_overlay.png",help="Output: contours overlay (BGR)")
    parser.add_argument("--filled",default="filled_mask.png",help="Output: filled binary mask")
    parser.add_argument("--color",default="coloured_regions.png",help="Output: per-region coloured map")
    parser.add_argument("--masks_dir",default="public\data\shape_masks",help="Output dir for per-region masks")
    parser.add_argument("--blur",type=int,default=1,help="Gaussian blur kernel (odd). 1 disables smoothing")
    parser.add_argument("--canny_sigma",type=float,default=0.33,help="Auto-Canny sigma")
    parser.add_argument("--close",dest="gap_close",type=int,default=1,
                        help="Closing kernel (odd). Default 1 disables closing entirely.")
    parser.add_argument("--min_area",type=int,default=20,help="Minimum contour area to keep (default 20)")
    parser.add_argument("--threshold",action="store_true",
                        help="AND filled mask with Otsu threshold ofgrayscale image")
    parser.add_argument("--edge",help="Optional precomputed edge map path (uint8 0/255)",default=None)
    parser.add_argument("--mask_mode",choices=["top","even","all"],default="top",
                        help="Which contours to export: top-level only,even depths or all contours.")

    parser.add_argument("--palette_mode",choices=["kbatch","random"],default="kbatch",
                        help="Colour assignment strategy")
    parser.add_argument("--sample_step",type=int,default=7,
                        help="RGB grid step for k-batch-means (smaller = denser)")
    args = parser.parse_args()

    img_path = args.image_opt or args.image or _find_default_image()
    if not img_path or not Path(img_path).exists():
        parser.print_help(sys.stderr)
        print("\nERROR: No input image supplied and none found incurrent directory. "
              "Provide an image path (positional or --image) or set INPUT_IMAGE.",file=sys.stderr)
        sys.exit(2)

    have_edges = None
    if args.edge:
        em = cv.imread(args.edge,cv.IMREAD_GRAYSCALE)
        if em is None:
            print(f"WARNING: Could not read --edge '{args.edge}'. Falling back to Canny.",file=sys.stderr)
        else:
            have_edges = (em > 0).astype("uint8") * 255
    edges,filled,contours,hierarchy,saved = process_image(
        img_path,out_overlay_path=args.overlay,
        out_filled_path=args.filled,out_colour_path=args.color,
        out_masks_dir=args.masks_dir,have_edges=have_edges,
        min_area=args.min_area,gap_close=args.gap_close,
        do_threshold=args.threshold,blur_ksize=args.blur,
        canny_sigma=args.canny_sigma,palette_mode=args.palette_mode, #NEW
        sample_step=args.sample_step,)  #NEW
    print(f"[OK] Saved:\n  • {args.filled}\n  • {len(saved)} masks in '{args.masks_dir}/'")
    print(f"Contours kept: {len(contours)}")