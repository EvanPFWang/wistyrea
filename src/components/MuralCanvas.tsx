// src/components/MuralCanvas.tsx
// Base canvas + hover overlay + DOM shape layer for family loading & project glow.
import { useEffect, useRef } from 'react';
import { Region, RGB } from '../types';
import { ShapeLayer } from './ShapeLayer';

interface MuralCanvasProps {
  baseImage: HTMLImageElement | null;
  hoveredRegion: Region | null;
  /** Set of Morton IDs in the currently loaded family */
  familyIds: Set<number>;
  /** Full regions map */
  regions: Map<number, Region>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onClick: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  getRegionColor: (regionId: number) => RGB;
  fetchMask: (regionId: number) => Promise<{ imageData: ImageData; x: number; y: number } | null>;
}

export function MuralCanvas({
  baseImage,
  hoveredRegion,
  familyIds,
  regions,
  canvasRef,
  onPointerMove,
  onClick,
  getRegionColor,
  fetchMask,
}: MuralCanvasProps) {
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Reusable temp canvas to avoid allocation on every hover
  const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Track previous mask bbox for targeted clear
  const prevBboxRef = useRef<{ x: number; y: number; w: number; h: number } | null>(null);

  // Set up base canvas
  useEffect(() => {
    if (!baseImage || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = baseImage.width;
    canvas.height = baseImage.height;
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (ctx) ctx.drawImage(baseImage, 0, 0);
  }, [baseImage, canvasRef]);

  // Set up overlay canvas
  useEffect(() => {
    if (!baseImage || !overlayCanvasRef.current) return;
    const overlay = overlayCanvasRef.current;
    overlay.width = baseImage.width;
    overlay.height = baseImage.height;
    if (!tempCanvasRef.current) {
      tempCanvasRef.current = document.createElement('canvas');
    }
    tempCanvasRef.current.width = baseImage.width;
    tempCanvasRef.current.height = baseImage.height;
  }, [baseImage]);

  // Draw hover overlay (mask highlight)
  useEffect(() => {
    let cancelled = false;

    const drawOverlay = async () => {
      if (!overlayCanvasRef.current || !canvasRef.current) return;
      const overlayCtx = overlayCanvasRef.current.getContext('2d', { alpha: true });
      if (!overlayCtx) return;

      overlayCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);

      if (!hoveredRegion) {
        if (prevBboxRef.current && tempCanvasRef.current) {
          const tmpCtx = tempCanvasRef.current.getContext('2d');
          if (tmpCtx) {
            const { x, y, w, h } = prevBboxRef.current;
            tmpCtx.clearRect(x, y, w, h);
          }
          prevBboxRef.current = null;
        }
        return;
      }

      const currentRegionId = hoveredRegion.id;
      const maskResult = await fetchMask(currentRegionId);
      if (cancelled || !maskResult) return;

      const tmp = tempCanvasRef.current;
      if (!tmp) return;
      const tmpCtx = tmp.getContext('2d');
      if (!tmpCtx) return;

      if (prevBboxRef.current) {
        const { x, y, w, h } = prevBboxRef.current;
        tmpCtx.clearRect(x, y, w, h);
      } else {
        tmpCtx.clearRect(0, 0, tmp.width, tmp.height);
      }
      tmpCtx.putImageData(maskResult.imageData, maskResult.x, maskResult.y);
      prevBboxRef.current = {
        x: maskResult.x, y: maskResult.y,
        w: maskResult.imageData.width, h: maskResult.imageData.height,
      };

      const color = getRegionColor(currentRegionId);
      overlayCtx.globalCompositeOperation = 'source-over';
      overlayCtx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.4)`;
      overlayCtx.fillRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      overlayCtx.globalCompositeOperation = 'destination-in';
      overlayCtx.drawImage(tmp, 0, 0);
      overlayCtx.globalCompositeOperation = 'source-over';
    };

    drawOverlay();
    return () => { cancelled = true; };
  }, [hoveredRegion, fetchMask, getRegionColor, canvasRef]);

  const cw = baseImage?.width ?? 0;
  const ch = baseImage?.height ?? 0;

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <div className="relative">
        {/* Base mural image */}
        <canvas
          ref={canvasRef}
          onPointerMove={onPointerMove}
          onClick={onClick}
          className="max-w-full max-h-screen cursor-crosshair"
          style={{ imageRendering: 'crisp-edges' }}
        />
        {/* DOM shape layer: .webp family members + project glow */}
        <ShapeLayer
          familyIds={familyIds}
          regions={regions}
          getRegionColor={getRegionColor}
          canvasWidth={cw}
          canvasHeight={ch}
        />
        {/* Hover mask overlay */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute top-0 left-0 pointer-events-none max-w-full max-h-screen"
          style={{ imageRendering: 'crisp-edges' }}
        />
      </div>
    </div>
  );
}
