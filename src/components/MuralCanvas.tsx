//src/components/MuralCanvas.tsx
//FIXED: Reusable temp canvas, async race condition handling
import { useEffect, useRef } from 'react';
import { Region, RGB } from '../types';

interface MuralCanvasProps {
  baseImage: HTMLImageElement | null;
  hoveredRegion: Region | null;
  activeRegions: Region[];
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onClick: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  getRegionColor: (regionId: number) => RGB;
  fetchMask: (regionId: number) => Promise<{ imageData: ImageData; x: number; y: number } | null>;
}

export function MuralCanvas({
  baseImage,
  hoveredRegion,
  activeRegions,
  canvasRef,
  onPointerMove,
  onClick,
  getRegionColor,
  fetchMask,
}: MuralCanvasProps) {
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const outlineCanvasRef = useRef<HTMLCanvasElement>(null);

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

  //draw active outlines (once, for regions with projects)
  useEffect(() => {
    if (!baseImage || !outlineCanvasRef.current || activeRegions.length === 0) return;

    const outlineCanvas = outlineCanvasRef.current;
    outlineCanvas.width = baseImage.width;
    outlineCanvas.height = baseImage.height;
    const ctx = outlineCanvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let cancelled = false;

    const drawOutlines = async () => {
      for (const region of activeRegions) {
        if (cancelled) return;
        const maskResult = await fetchMask(region.id);
        if (cancelled || !maskResult) continue;

        const { imageData, x: ox, y: oy } = maskResult;
        const { width: w, height: h } = imageData;
        const src = imageData.data;
        const color = getRegionColor(region.id);

        //edge detection: pixel is edge if alpha>0 and any 4-neighbor has alpha=0
        const outlineData = ctx.createImageData(w, h);
        const dst = outlineData.data;

        for (let py = 0; py < h; py++) {
          for (let px = 0; px < w; px++) {
            const i = (py * w + px) * 4;
            if (src[i + 3] === 0) continue; //not part of mask

            const isEdge =
              px === 0 || px === w - 1 || py === 0 || py === h - 1 ||
              src[((py - 1) * w + px) * 4 + 3] === 0 ||
              src[((py + 1) * w + px) * 4 + 3] === 0 ||
              src[(py * w + px - 1) * 4 + 3] === 0 ||
              src[(py * w + px + 1) * 4 + 3] === 0;

            if (isEdge) {
              dst[i] = color.r;
              dst[i + 1] = color.g;
              dst[i + 2] = color.b;
              dst[i + 3] = 255;
            }
          }
        }

        //thicken outline by drawing shifted copies
        const thickCanvas = document.createElement('canvas');
        thickCanvas.width = w + 4;
        thickCanvas.height = h + 4;
        const thickCtx = thickCanvas.getContext('2d')!;
        const tmpImg = document.createElement('canvas');
        tmpImg.width = w;
        tmpImg.height = h;
        tmpImg.getContext('2d')!.putImageData(outlineData, 0, 0);

        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx * dx + dy * dy <= 4) { //circular kernel
              thickCtx.drawImage(tmpImg, 2 + dx, 2 + dy);
            }
          }
        }

        ctx.drawImage(thickCanvas, ox - 2, oy - 2);
      }
    };

    drawOutlines();
    return () => { cancelled = true; };
  }, [baseImage, activeRegions, fetchMask, getRegionColor]);

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
          ref={outlineCanvasRef}
          className="absolute top-0 left-0 pointer-events-none max-w-full max-h-screen"
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
