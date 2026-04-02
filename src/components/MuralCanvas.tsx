//src/components/MuralCanvas.tsx
//FIXED: Reusable temp canvas, async race condition handling
import { useEffect, useRef } from 'react';
import { Region, RGB } from '../types';

interface MuralCanvasProps {
  baseImage: HTMLImageElement | null;
  hoveredRegion: Region | null;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onClick: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  getRegionColor: (regionId: number) => RGB;
  fetchMask: (regionId: number) => Promise<{ imageData: ImageData; x: number; y: number } | null>;
}

export function MuralCanvas({
  baseImage,
  hoveredRegion,
  canvasRef,
  onPointerMove,
  onClick,
  getRegionColor,
  fetchMask,
}: MuralCanvasProps) {
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  //reusable temp canvas to avoid allocation on every hover
  const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);

  //track previous mask bbox for targeted clear (avoids 3.3M pixel clearRect)
  const prevBboxRef = useRef<{ x: number; y: number; w: number; h: number } | null>(null);

  //set up canvas dimensions
  useEffect(() => {
    if (!baseImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    canvas.width = baseImage.width;
    canvas.height = baseImage.height;

    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (ctx) {
      ctx.drawImage(baseImage, 0, 0);
    }
  }, [baseImage, canvasRef]);

  //set up overlay canvas
  useEffect(() => {
    if (!baseImage || !overlayCanvasRef.current) return;

    const overlay = overlayCanvasRef.current;
    overlay.width = baseImage.width;
    overlay.height = baseImage.height;
    
    //init reusable temp canvas
    if (!tempCanvasRef.current) {
      tempCanvasRef.current = document.createElement('canvas');
    }
    tempCanvasRef.current.width = baseImage.width;
    tempCanvasRef.current.height = baseImage.height;
  }, [baseImage]);

  //draw hover overlay
  //cancellation flag prevent stale mask renders
  useEffect(() => {
    let cancelled = false;
    
    const drawOverlay = async () => {
      if (!overlayCanvasRef.current || !canvasRef.current) return;

      const overlayCtx = overlayCanvasRef.current.getContext('2d', { alpha: true });
      if (!overlayCtx) return;

      //clear overlay
      overlayCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);

      if (!hoveredRegion) {
        //clear temp canvas bbox residue when mouse leaves all regions
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

      //current region ID before async operation
      const currentRegionId = hoveredRegion.id;
      
      //ftch mask
      const maskResult = await fetchMask(currentRegionId);

      //check if request still relevant after async fetch
      if (cancelled) return;
      if (!maskResult) return;

      //reuse temp canvas instead of creating new one
      const tmp = tempCanvasRef.current;
      if (!tmp) return;

      const tmpCtx = tmp.getContext('2d');
      if (!tmpCtx) return;

      //clear only previous mask bbox instead of full 2560×1305 canvas
      if (prevBboxRef.current) {
        const { x, y, w, h } = prevBboxRef.current;
        tmpCtx.clearRect(x, y, w, h);
      } else {
        tmpCtx.clearRect(0, 0, tmp.width, tmp.height);
      }
      tmpCtx.putImageData(maskResult.imageData, maskResult.x, maskResult.y);
      prevBboxRef.current = {
        x: maskResult.x, y: maskResult.y,
        w: maskResult.imageData.width, h: maskResult.imageData.height
      };

      //get region color
      const color = getRegionColor(currentRegionId);

      //draw colored overlay
      overlayCtx.globalCompositeOperation = 'source-over';
      overlayCtx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.4)`;
      overlayCtx.fillRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);

      //apply mask
      overlayCtx.globalCompositeOperation = 'destination-in';
      overlayCtx.drawImage(tmp, 0, 0);

      overlayCtx.globalCompositeOperation = 'source-over';
    };

    drawOverlay();
    
    //cleanup function to cancel stale renders
    return () => {
      cancelled = true;
    };
  }, [hoveredRegion, fetchMask, getRegionColor, canvasRef]);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <div className="relative">
        <canvas
          ref={canvasRef}
          onPointerMove={onPointerMove}
          onClick={onClick}
          className="max-w-full max-h-screen cursor-crosshair"
          style={{ imageRendering: 'crisp-edges' }}
        />
        <canvas
          ref={overlayCanvasRef}
          className="absolute top-0 left-0 pointer-events-none max-w-full max-h-screen"
          style={{ imageRendering: 'crisp-edges' }}
        />
      </div>
    </div>
  );
}
