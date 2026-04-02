//src/hooks/useMuralController.ts
//GPU cleanup, RAF throttling, LRU cache, in-flight dedup, abort signals
import { useState, useEffect, useRef, useCallback } from 'react';
import { Metadata, Region, Palette, RGB } from '../types';
import { RLEDecoder } from '../lib/rle-decoder';
import { PROJECT_ASSIGNMENTS } from '../config/projects';

const RAW_BASE = import.meta.env.BASE_URL ?? '/';
const BASE = RAW_BASE.endsWith('/') ? RAW_BASE : RAW_BASE + '/';
const ABS_BASE = new URL(BASE, document.baseURI);
const ABSOLUTE_RE = /^[a-zA-Z][\w+.-]*:|^\/\//;

export const absUrl = (p: string) => {
  if (ABSOLUTE_RE.test(p)) return p;
  const clean = p.replace(/^\/+/, '').replace(/\\/g, '/');
  return new URL(clean, ABS_BASE).href;
};

//cache behavior configurable (remove 'no-cache' for production)
async function fetchJSON<T>(path: string, useCache = true): Promise<T | null> {
  const res = await fetch(absUrl(path), useCache ? {} : { cache: 'no-cache' });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

//LRU Cache class to prevent unbounded memory growth
class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number = 30) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    //remove if exists (will re-add at end)
    this.cache.delete(key);
    
    //add to end
    this.cache.set(key, value);
    
    //evict oldest if over limit
    if (this.cache.size > this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }
}

interface UseMuralControllerOptions {
  dataPath?: string;
  metadataPath?: string;
  palettePath?: string;
  idMapPath?: string;
  baseImagePath?: string;
  maxCacheSize?: number;  //configurable cache size
  disableHttpCache?: boolean;  //option to disable HTTP cache (for dev)
}

export function useMuralController(options: UseMuralControllerOptions = {}) {
  const {
    dataPath = 'data',
    metadataPath = `${dataPath}/metadata.json`,
    palettePath = `${dataPath}/palette.json`,
    idMapPath = `${dataPath}/id_map.png`,
    baseImagePath = 'Mural_Crown_of_Italian_City.svg.png',
    maxCacheSize = 30,
    disableHttpCache = false,
  } = options;

  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [palette, setPalette] = useState<Palette | null>(null);
  const [regions, setRegions] = useState<Map<number, Region>>(new Map());
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [baseImage, setBaseImage] = useState<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState(true);
  
  const idMapRef = useRef<ImageData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const decoderRef = useRef<RLEDecoder>(new RLEDecoder());
  const metaDirRef = useRef<string>('');
  
  //LRU cache instead of unbounded Map
  const maskCacheRef = useRef<LRUCache<number, ImageData>>(new LRUCache(maxCacheSize));
  
  //track in-flight fetches prevent duplicates
  const pendingFetchesRef = useRef<Map<number, Promise<ImageData | null>>>(new Map());
  
  //RAF-based throttling refs
  const rafRef = useRef<number | null>(null);
  const lastRegionIdRef = useRef<number>(0);
  
  //abort controller cleanup
  const abortControllerRef = useRef<AbortController>(new AbortController());
  //init data
  useEffect(() => {
    const init = async () => {
      try {
        const [meta, pal] = await Promise.all([
          fetchJSON<Metadata>(metadataPath, !disableHttpCache),
          fetchJSON<any>(palettePath, !disableHttpCache),
        ]);

        if (!meta) throw new Error('Failed to load metadata');
        
        setMetadata(meta);
        
        //norm palette
        const normalizedPalette: Palette = {
          background_id: pal?.background_id ?? 0,
          map: pal?.map ?? {},
        };
        setPalette(normalizedPalette);

        //build regions map, applying project assignments
        const regionsMap = new Map<number, Region>();
        meta.regions.forEach(r => {
          const project = PROJECT_ASSIGNMENTS[r.id];
          regionsMap.set(r.id, project ? { ...r, project } : r);
        });
        setRegions(regionsMap);

        //extract metadata directory
        const lastSlash = metadataPath.lastIndexOf('/');
        metaDirRef.current = lastSlash >= 0 ? metadataPath.substring(0, lastSlash) : '';

        //load base image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = absUrl(baseImagePath);
        });
        await img.decode();
        setBaseImage(img);

        //load ID map
        const idImg = new Image();
        idImg.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
          idImg.onload = resolve;
          idImg.onerror = reject;
          idImg.src = absUrl(idMapPath);
        });
        await idImg.decode();

        const idCanvas = document.createElement('canvas');
        idCanvas.width = idImg.width;
        idCanvas.height = idImg.height;
        const idCtx = idCanvas.getContext('2d', { willReadFrequently: true })!;
        idCtx.drawImage(idImg, 0, 0);
        idMapRef.current = idCtx.getImageData(0, 0, idCanvas.width, idCanvas.height);

        setLoading(false);
      } catch (error) {
        console.error('Failed to initialize mural:', error);
        setLoading(false);
      }
    };

    init();

    //cleanup on unmount
    return () => {
      //abort  pending operations
      abortControllerRef.current.abort();
      
      //cancel pending RAF
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
      
      //destroy WebGPU resources
      decoderRef.current.destroy();
      
      //clear caches
      maskCacheRef.current.clear();
      pendingFetchesRef.current.clear();
    };
  }, [metadataPath, palettePath, idMapPath, baseImagePath, disableHttpCache]);

  //read region ID from pixel coordinates
  const readIdAt = useCallback((clientX: number, clientY: number): number => {
    if (!idMapRef.current || !canvasRef.current) return 0;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left) * (idMapRef.current.width / rect.width));
    const y = Math.floor((clientY - rect.top) * (idMapRef.current.height / rect.height));

    if (x < 0 || x >= idMapRef.current.width || y < 0 || y >= idMapRef.current.height) return 0;

    const idx = (y * idMapRef.current.width + x) * 4;
    const data = idMapRef.current.data;
    return data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16);
  }, []);

  //fetch + decode mask for region
  //in-flight dedupe + abort signal support
  //returns bbox-sized ImageData and its top-left position within the full canvas
  const fetchMask = useCallback(async (regionId: number): Promise<{ imageData: ImageData; x: number; y: number } | null> => {
    const region = regions.get(regionId);
    if (!region?.mask) return null;

    //check cache — re-attach position from metadata (not stored in cache)
    const cached = maskCacheRef.current.get(regionId);
    if (cached) {
      return { imageData: cached, x: region.bbox.x, y: region.bbox.y };
    }

    //check if fetch already in progress (dedupe)
    const pending = pendingFetchesRef.current.get(regionId);
    if (pending) {
      return pending.then(imgData =>
        imgData ? { imageData: imgData, x: region.bbox.x, y: region.bbox.y } : null
      );
    }

    //create tracked promise for dedupe (stores only ImageData, not the wrapper)
    const fetchPromise = (async (): Promise<ImageData | null> => {
      try {
        const maskPath = region.mask.startsWith('http')
          ? region.mask
          : `${metaDirRef.current}/${region.mask}`;

        const res = await fetch(absUrl(maskPath), {
          signal: abortControllerRef.current.signal
        });
        if (!res.ok) return null;

        //if aborted before expensive GPU work
        if (abortControllerRef.current.signal.aborted) return null;

        const rleBuffer = await res.arrayBuffer();
        //decode at bbox dimensions — .bin is cropped, not full 2560×1305
        const { width: bw, height: bh } = region.bbox;
        const outputSize = bw * bh;

        const imgData = await decoderRef.current.decode(
          rleBuffer,
          outputSize,
          bw,
          bh,
          abortControllerRef.current.signal
        );

        //check after decode
        if (abortControllerRef.current.signal.aborted) return null;

        maskCacheRef.current.set(regionId, imgData);
        return imgData;
      } catch (error) {
        //silently ignore abort errors
        if (error instanceof DOMException && error.name === 'AbortError') {
          return null;
        }
        console.error('Failed to fetch mask:', error);
        return null;
      } finally {
        //Remove from pending on completion
        pendingFetchesRef.current.delete(regionId);
      }
    })();

    pendingFetchesRef.current.set(regionId, fetchPromise);
    return fetchPromise.then(imgData =>
      imgData ? { imageData: imgData, x: region.bbox.x, y: region.bbox.y } : null
    );
  }, [regions]);

  //handle pointer move
  //RAF-based throttling to prevent excessive state updates
  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    //cancel any pending RAF
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }

    //sched update on next animation frame
    rafRef.current = requestAnimationFrame(() => {
      const id = readIdAt(e.clientX, e.clientY);
      
      //only update state if region actually changed
      if (id !== lastRegionIdRef.current) {
        lastRegionIdRef.current = id;
        const region = id > 0 ? regions.get(id) ?? null : null;
        setHoveredRegion(region);
      }
    });
  }, [readIdAt, regions]);

  //handle click
  const handleClick = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const id = readIdAt(e.clientX, e.clientY);
    const region = id > 0 ? regions.get(id) ?? null : null;
    setSelectedRegion(region);
  }, [readIdAt, regions]);

  //get color for region
  const getRegionColor = useCallback((regionId: number): RGB => {
    return palette?.map[String(regionId)] ?? { r: 255, g: 215, b: 0 };
  }, [palette]);

  //regions that have projects assigned (for outline rendering)
  const activeRegions = useCallback((): Region[] => {
    const result: Region[] = [];
    regions.forEach(r => { if (r.project) result.push(r); });
    return result;
  }, [regions]);

  //convert image-space bbox to viewport-space rect
  const bboxToViewport = useCallback((bbox: { x: number; y: number; width: number; height: number }) => {
    if (!canvasRef.current) return { x: 0, y: 0, width: 0, height: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const idMap = idMapRef.current;
    if (!idMap) return { x: 0, y: 0, width: 0, height: 0 };
    const scaleX = rect.width / idMap.width;
    const scaleY = rect.height / idMap.height;
    return {
      x: rect.left + bbox.x * scaleX,
      y: rect.top + bbox.y * scaleY,
      width: bbox.width * scaleX,
      height: bbox.height * scaleY,
    };
  }, []);

  return {
    metadata,
    palette,
    regions,
    hoveredRegion,
    selectedRegion,
    baseImage,
    loading,
    canvasRef,
    handlePointerMove,
    handleClick,
    fetchMask,
    getRegionColor,
    setSelectedRegion,
    activeRegions,
    bboxToViewport,
  };
}
