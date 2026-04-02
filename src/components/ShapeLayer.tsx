// src/components/ShapeLayer.tsx
// DOM-based shape overlay layer.
// Renders .webp images for the loaded Morton family.
// Project-linked shapes receive a CSS glow animation in their palette color.

import { useState, useCallback } from 'react';
import { Region, RGB } from '../types';

interface ShapeLayerProps {
  /** Currently loaded family region IDs */
  familyIds: Set<number>;
  /** Full regions map for bbox/project lookup */
  regions: Map<number, Region>;
  /** Palette color lookup */
  getRegionColor: (regionId: number) => RGB;
  /** Canvas natural dimensions (for percentage-based positioning) */
  canvasWidth: number;
  canvasHeight: number;
  /** Base path for .webp shape assets */
  shapePath?: string;
}

export function ShapeLayer({
  familyIds,
  regions,
  getRegionColor,
  canvasWidth,
  canvasHeight,
  shapePath = 'data/shape_masks',
}: ShapeLayerProps) {
  // Track which .webp files failed to load (don't exist yet)
  const [failedIds, setFailedIds] = useState<Set<number>>(new Set());

  const handleImgError = useCallback((id: number) => {
    setFailedIds(prev => {
      if (prev.has(id)) return prev;
      const next = new Set(prev);
      next.add(id);
      return next;
    });
  }, []);

  if (canvasWidth === 0 || canvasHeight === 0) return null;

  const entries: { region: Region; color: RGB; isProject: boolean }[] = [];
  familyIds.forEach(id => {
    if (failedIds.has(id)) return;
    const region = regions.get(id);
    if (!region) return;
    entries.push({
      region,
      color: getRegionColor(id),
      isProject: !!region.project,
    });
  });

  return (
    <div
      className="absolute top-0 left-0 pointer-events-none max-w-full max-h-screen overflow-hidden"
      style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px` }}
    >
      {entries.map(({ region, color, isProject }) => {
        const { bbox, id } = region;
        const fileIdx = (id - 1).toString().padStart(3, '0');
        const shadowColor = `rgb(${color.r}, ${color.g}, ${color.b})`;

        return (
          <img
            key={id}
            src={`${shapePath}/shape_${fileIdx}.webp`}
            alt=""
            onError={() => handleImgError(id)}
            className={isProject ? 'absolute project-glow' : 'absolute'}
            style={{
              left: `${bbox.x}px`,
              top: `${bbox.y}px`,
              width: `${bbox.width}px`,
              height: `${bbox.height}px`,
              ...(isProject
                ? {
                    willChange: 'filter',
                    '--glow-color': shadowColor,
                  } as React.CSSProperties
                : {}),
            }}
          />
        );
      })}
    </div>
  );
}
