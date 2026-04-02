// src/lib/quadrant.ts
// Spatial math: viewport quadrant calculation for dynamic UI placement.

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PreviewRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Divides the viewport into 4 quadrants using the bbox center as a crosshair.
 * Returns the center 2/3 of the largest quadrant.
 *
 * Tie-breaker: strict `>` means TL wins when areas are equal (deterministic).
 */
export function computeQuadrantPreviewRect(
  viewportBbox: Rect,
  vw: number,
  vh: number,
): PreviewRect {
  const cx = viewportBbox.x + viewportBbox.width / 2;
  const cy = viewportBbox.y + viewportBbox.height / 2;

  const quadrants = [
    { area: cx * cy, x: 0, y: 0, w: cx, h: cy },                           // TL
    { area: (vw - cx) * cy, x: cx, y: 0, w: vw - cx, h: cy },              // TR
    { area: cx * (vh - cy), x: 0, y: cy, w: cx, h: vh - cy },              // BL
    { area: (vw - cx) * (vh - cy), x: cx, y: cy, w: vw - cx, h: vh - cy }, // BR
  ];

  const largest = quadrants.reduce((a, b) => (b.area > a.area ? b : a));

  const m = 1 / 6;
  return {
    left: largest.x + largest.w * m,
    top: largest.y + largest.h * m,
    width: largest.w * (2 / 3),
    height: largest.h * (2 / 3),
  };
}
