# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive web visualization of the "Crown of Italian City" mural. Users hover/click on ~626 brick-shaped regions to explore embedded project demos. Hit detection is O(1) via a pre-rendered ID map; masks are RLE-compressed and decoded on-demand with a WebGPU compute shader.

## Commands

| Task | Command |
|------|---------|
| Dev server (port 3000) | `npm run dev` |
| Full build (process image + Vite) | `npm run build` |
| Vite build only (skip Python) | `npm run build:vite-only` |
| Preview production build | `npm run preview` |
| Regenerate region data | `npm run process:image` |
| Install Python deps | `npm run pip:install` |
| Type check | `npx tsc --noEmit` |

No test framework or linter is configured.

## Tech Stack

* **Frontend**: React 18, Vite, TypeScript (ES2022), WebGPU, Tailwind CSS, Radix UI (shadcn/ui).
* **Backend** (optional): Cloudflare Workers + Durable Objects (SQLite-backed). Currently the app is fully client-side; the worker in `worker.ts` provides optional shared state.
* **Processing**: Python 3.10 — `opencv-python-headless==4.8.1.78`, `numpy==1.24.3`, `scikit-learn==1.3.0`.
* **Path alias**: `@/*` maps to `./src/*`.

## Architecture

### Data Pipeline

```
Mural_Crown_of_Italian_City.svg.png
  → scripts/regions.py (Canny edges → contours → Morton sort → RLE encode)
  → public/data/
      ├── metadata.json    (region definitions: id, bbox, centroid, hierarchy, mask path)
      ├── palette.json     (RGB colors keyed by string region ID)
      ├── id_map.png       (24-bit encoded region IDs: R | G<<8 | B<<16)
      └── shape_masks/     (shape_NNN.bin — RLE binary masks, cropped to bbox)
```

### Frontend Rendering Layers (bottom to top)

1. **Base canvas** — full mural image drawn once.
2. **ShapeLayer** (DOM) — `<img>` elements for `.webp` shape outlines loaded by Morton family. Project-linked shapes get a CSS `drop-shadow` glow animation via `--glow-color` variable.
3. **Overlay canvas** — semi-transparent colored mask highlight on hover, composited via `destination-in`.

### Core State Flow

```
pointer move → readIdAt(id_map.png) → Morton region ID
  → useMuralController: hoveredRegion state (RAF-throttled, only fires on ID change)
  → useFamilyLoader: loads ±50 contiguous Morton IDs (debounced 50ms)
  → MuralCanvas: fetches RLE mask → WebGPU decode → overlay render
  → if region.project: ProjectPreviewCard (quadrant-positioned iframe)
  → on click: ProjectDemoDialog (modal with iframe)
```

### Key Modules

* **`useMuralController`** — central hook: loads metadata/palette/id_map, manages LRU mask cache (max 30), in-flight fetch deduplication, abort signal cleanup, RAF-throttled hover detection.
* **`useFamilyLoader`** — computes Morton family window around hovered region. 50ms debounce prevents thrashing during rapid cursor movement.
* **`RLEDecoder`** (`src/lib/rle-decoder.ts`) — WebGPU compute shader for RLE decompression. Handles device loss recovery. All GPU buffers destroyed in `finally` blocks.
* **`computeQuadrantPreviewRect`** (`src/lib/quadrant.ts`) — positions preview UI in center 2/3 of the largest viewport quadrant (formed by bbox center crosshair). Deterministic TL tie-breaker.
* **`src/config/projects.ts`** — maps region IDs to `ProjectInfo` (title, href, blurb, keywords). This is where you assign projects to bricks.

## Hard Constraints

* **Cloudflare 128MB limit**: No `OffscreenCanvas` or `new Image()` in worker code. Keep mask processing buffer-based.
* **WebGPU 16-byte alignment**: WGSL struct fields must be 16-byte aligned. The `Params` uniform buffer is padded to 16 bytes even though it only holds one `u32`.
* **RLE masks cropped to bbox**: `.bin` files encode only the bounding-box-sized region, not the full 2560×1305 canvas. Decode dimensions come from `region.bbox.width/height`.
* **id_map.png must be unscaled**: Pixel-level sampling requires 1:1 resolution. The hidden canvas used for `getImageData` must match the image's natural dimensions.
* **No SVG path rendering for regions**: 626 SVG `<path>` elements would tank performance. Use canvas compositing or DOM `<img>` elements for shape rendering.
* **Palette is RGB** (not BGR): `palette.json` stores `{r, g, b}` values 0–255. Fallback color is gold `{255, 215, 0}`.
