// src/components/ProjectPreviewCard.tsx
import { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Region } from '../types';

interface ProjectPreviewCardProps {
  region: Region;
  /** bbox of the active region in viewport (screen) coordinates */
  viewportBbox: { x: number; y: number; width: number; height: number };
}

/**
 * Calculates the largest quadrant formed by extending the bbox center
 * as a crosshair to viewport edges, then returns the center 2/3 rect.
 */
function computePreviewRect(
  vbbox: { x: number; y: number; width: number; height: number },
  vw: number,
  vh: number,
) {
  const cx = vbbox.x + vbbox.width / 2;
  const cy = vbbox.y + vbbox.height / 2;

  // Four quadrant areas
  const areas = [
    { area: cx * cy, x: 0, y: 0, w: cx, h: cy },                   // TL
    { area: (vw - cx) * cy, x: cx, y: 0, w: vw - cx, h: cy },      // TR
    { area: cx * (vh - cy), x: 0, y: cy, w: cx, h: vh - cy },      // BL
    { area: (vw - cx) * (vh - cy), x: cx, y: cy, w: vw - cx, h: vh - cy }, // BR
  ];

  const largest = areas.reduce((a, b) => (b.area > a.area ? b : a));

  // Center 2/3 of the largest quadrant
  const margin = 1 / 6;
  return {
    left: largest.x + largest.w * margin,
    top: largest.y + largest.h * margin,
    width: largest.w * (2 / 3),
    height: largest.h * (2 / 3),
  };
}

export function ProjectPreviewCard({ region, viewportBbox }: ProjectPreviewCardProps) {
  if (!region.project) return null;

  const { title, blurb, href, keywords } = region.project;

  const rect = useMemo(
    () => computePreviewRect(viewportBbox, window.innerWidth, window.innerHeight),
    [viewportBbox],
  );

  // Minimum usable size for preview
  if (rect.width < 200 || rect.height < 150) return null;

  return (
    <Card
      className="fixed z-50 overflow-hidden animate-in fade-in-0 zoom-in-95 duration-200 flex flex-col"
      style={{
        left: `${rect.left}px`,
        top: `${rect.top}px`,
        width: `${rect.width}px`,
        height: `${rect.height}px`,
      }}
    >
      <CardHeader className="pb-2 flex-shrink-0">
        <CardTitle className="text-base text-primary">{title}</CardTitle>
        <CardDescription className="text-xs line-clamp-2">{blurb}</CardDescription>
        <div className="flex flex-wrap gap-1.5 pt-1">
          {keywords.map((kw) => (
            <Badge key={kw} variant="secondary" className="text-xs">
              {kw}
            </Badge>
          ))}
        </div>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pb-3">
        <div className="w-full h-full rounded-md overflow-hidden border bg-background">
          <iframe
            src={href}
            className="w-full h-full"
            title={`${title} Preview`}
            sandbox="allow-scripts allow-same-origin"
            loading="lazy"
          />
        </div>
      </CardContent>
    </Card>
  );
}
