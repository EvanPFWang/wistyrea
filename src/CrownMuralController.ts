// src/CrownMuralController.ts
import {RGB, Hex, Palette, Metadata,Region, ProjectType, ControllerConfig} from './types.ts';
//import {l} from 'vite/dist/node/types.d-aGj9QkWt';
//import basePngUrl from '/Mural_Crown_of_Italian_City.svg.png?url';
const RAW_BASE = import.meta.env.BASE_URL ?? '/';
const BASE     = RAW_BASE.endsWith('/') ? RAW_BASE : RAW_BASE + '/';
const ABS_BASE = new URL(BASE, document.baseURI); //e.g. https://site.tld/subapp/


const ABSOLUTE_RE = /^[a-zA-Z][\w+.-]*:|^\/\//;
export const absUrl = (p: string) => {
  if (ABSOLUTE_RE.test(p)) return p;//already abs/proto-relative
  const clean = p.replace(/^\/+/, '').replace(/\\/g, '/');

  return new URL(clean, ABS_BASE).href}// fully qualified URL
export const pathUrl = (p: string) => {
  const u = new URL(absUrl(p));
  return u.pathname + u.search + u.hash}//originless path

export const url    =   absUrl
export const pathnameOf = (p: string) => new URL(absUrl(p)).pathname;


async function fetchJSON<T>(path: string, { optional = false } = {}): Promise<T | null> {
  const res = await fetch(url(path), { cache: 'no-cache' }); // or 'force-cache' for CDN caching
  if (!res.ok) {
    if (optional && res.status === 404) return null;
    throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}
//const HEX_RE = /^#?(?<r>[0-9a-fA-F]{2})(?<g>[0-9a-fA-F]{2})(?<b>[0-9a-fA-F]{2})$/;
//kept sync for less overhead/await backup
function normalizePaletteToRGB(raw: unknown): Palette {
  const any = raw as any;

  if (any && typeof any === 'object' && 'map' in any) {
    const bg = Number(any.background_id ?? 0);
    const m = any.map as Record<string, unknown>;
    const first = Object.values(m)[0];

    if (first && typeof first === 'object' && first !== null && 'r' in (first as any)) {
      // Already RGB
      return { background_id: bg, map: m as Record<string, RGB> };
    }
    // Assume hex → RGB
    const map = Object.fromEntries(
      Object.entries(m).map(([k, v]) => [k, typeof v === 'string' ? hexToRGB(v) : { r: 0, g: 0, b: 0 }])
    ) as Record<string, RGB>;
    return { background_id: bg, map };
  }

  // Plain map at root
  const m = (any ?? {}) as Record<string, unknown>;
  const map = Object.fromEntries(
    Object.entries(m).map(([k, v]) =>
      [k, typeof v === 'string' ? hexToRGB(v) : (v as RGB) ?? { r: 0, g: 0, b: 0 }]
    )
  ) as Record<string, RGB>;
  return { background_id: 0, map };
}

const toHex2 = (n: number) => Math.max(0, Math.min(255, n|0)).toString(16).padStart(2, '0');
export const rgbToHex = (c: RGB): Hex =>
  (`#${toHex2(c.r)}${toHex2(c.g)}${toHex2(c.b)}` as Hex);

//accept #RGB, #RRGGBB, #RGBA, #RRGGBBAA (alpha ignored)
export const hexToRGB = (hex: string): RGB => {
  const s = hex.trim().replace(/^#/, '');
  if (s.length === 3 || s.length === 4) {
    const r = s[0] + s[0], g = s[1] + s[1], b = s[2] + s[2];
    return { r: parseInt(r, 16), g: parseInt(g, 16), b: parseInt(b, 16) };
  }
  if (s.length === 6 || s.length === 8) {
    return {
      r: parseInt(s.slice(0, 2), 16),
      g: parseInt(s.slice(2, 4), 16),
      b: parseInt(s.slice(4, 6), 16),
    };
  }
  //fallback for invalid strings
  return { r: 0, g: 0, b: 0 };
};


export const MASK_INDEX_RE = /(?:^|\/)(?:shape|mask)_(?<index>\d+)\.png$/i;


type ResolvedConfig = Required<
  Pick<
    ControllerConfig, 'dataPath'|'enableDebug'|'maxFPS'|'maskBackgroundIndex'>
  > & Omit<
    ControllerConfig, 'dataPath'|'enableDebug'|'maxFPS'|'maskBackgroundIndex'
> & {metadataPath: string;
  unifiedMaskPath: string;
  colouredMaskPath: string;
  idMapPath: string;
  storagePath: string;
  baseImagePath: string;
  palettePath: string};

const DEFAULTS: ResolvedConfig = {
  dataPath: 'data',
  enableDebug: false,
  maxFPS: 60,
  maskBackgroundIndex: 0,

  metadataPath: 'data/metadata.json',
  unifiedMaskPath: 'data/shape_masks/unified_mask',
  colouredMaskPath: 'data/coloured_map.png',
  idMapPath: 'data/id_map.png',
  storagePath: 'data/shape_masks',
  baseImagePath: 'Mural_Crown_of_Italian_City.svg.png',
  palettePath: 'data/palette.json',
};


export class CrownMuralController {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private idCanvas: HTMLCanvasElement;
  private idCtx: CanvasRenderingContext2D;
  private tooltip: HTMLDivElement;
  private tooltipTitle: HTMLElement;
  private tooltipBlurb: HTMLElement;
  private config: ResolvedConfig;
  // Performance optimizations
  private regions: Map<number, Region> = new Map();
  private hoveredRegion: number | null = null;
  private prevId: number = -1;
  private latestEvent: PointerEvent | null = null;
  private needsRedraw: boolean = false;
  
  // Cached data
  private metadata: Metadata | null = null;
  private palette: Palette | null = null;
  private baseImage: HTMLImageElement | null = null;
  private idImageData: ImageData | null = null;


  private backgroundId: number = 0;

  private maskBackgroundIndex: number = 0;              // unified_mask background index (your regions.py uses 0)
  private paletteBackgroundIndex: number = 0;           // background_id coming from palette.json (often 0)
  private colorFallback: RGB = { r: 255, g: 255, b: 0 };//gold fallback for safety


  // @ts-ignore
    private unifiedMaskWidth = 0;
  // @ts-ignore
    private unifiedMaskHeight = 0;
  private unifiedMaskIndex32: Uint32Array | null = null;//region indices per pixel loaded from the unified mask
  /**private highlightColors: RGB[] = [];//highlight colors, repeating as needed
  private lastHighlight: { regionIdx: number; pixels: number[] } | null = null;//recorded last highlighted region and pixel offsets
  private currentRegionIndex: number = 0;//curr 1‑based region idx being highlighted
  private readonly config: ControllerConfig;//config passed in via constructor
   */
  private overlay: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private overlayDirty: { x: number; y: number; width: number; height: number } | null = null;// unifiedIndex -> mask <img>






  //private maskCache = new Map<number, HTMLImageElement>();// unifiedIndex -> mask <img>
  private regionByShapeIndex = new Map<number, Region>();  // iii (0-based) -> Region

  // Animation frame
  private animationId: number = 0;
  private fpsElement = document.getElementById('fps');
  private fpsLastUpdate = 0;
  private lastTimestamp: number = 0;

  // Precomputed lookup table for ID decoding
  // @ts-ignore
  //  private readonly idLookup = new Uint32Array(256 * 256 * 256);
    private displayIndexByPixelId = new Map<number, number>();


  private resizeCanvasesForDPR(w: number, h: number,status: boolean) {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    if (status){
      // @ts-ignore
        const set = (c: HTMLCanvasElement, ctx:
      CanvasRenderingContext2D, alpha=false) => {
        c.width  = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
        c.style.width  = w + 'px';
        c.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)}
      set(this.canvas,  this.ctx);
      set(this.idCanvas, this.idCtx);
      set(this.overlay, this.overlayCtx, true)}else{
      // @ts-ignore
        const set = (c: HTMLCanvasElement, ctx:
      CanvasRenderingContext2D, alpha=false) => {
        c .width  = Math.round(w);
        c.height = Math.round(h);
        c.style.width  = w + 'px';
        c.style.height = h + 'px';
        ctx.setTransform(1, 0, 0, 1, 0, 0)}
      set(this.canvas,  this.ctx);
      set(this.idCanvas, this.idCtx);
      set(this.overlay, this.overlayCtx, true)}}

  constructor(config: ControllerConfig={}) {
    const dp    =   config.dataPath ??  DEFAULTS.dataPath;
    this.config = {
      ...DEFAULTS,
      ...config,
      metadataPath:     config.metadataPath     ?? `${dp}/metadata.json`,
      unifiedMaskPath:  config.unifiedMaskPath  ?? `${dp}/shape_masks/unified_mask`,
      colouredMaskPath: config.colouredMaskPath ?? `${dp}/coloured_map.png`,
      idMapPath:        config.idMapPath        ?? `${dp}/id_map.png`,
      storagePath:      config.storagePath      ?? `${dp}/shape_masks`,
      palettePath:      config.palettePath      ?? `${dp}/palette.json`,
      baseImagePath:    config.baseImagePath    ?? DEFAULTS.baseImagePath,
    };
    this.canvas = document.getElementById('mural-canvas') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d', {
      alpha: false,
      desynchronized: true // Hint for better performance
    })!;

    this.overlay = document.createElement('canvas');
    this.overlayCtx = this.overlay.getContext('2d', { alpha: true })!;

    this.idCanvas = document.getElementById('id-canvas') as HTMLCanvasElement;
    this.idCtx = this.idCanvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false
    })!;

    this.tooltip = document.getElementById('tooltip') as HTMLDivElement;
    this.tooltipTitle = this.tooltip.querySelector('strong')!;
    this.tooltipBlurb = this.tooltip.querySelector('.blurb')!;

    this.init();
  }

  private async init(): Promise<void> {
    try {
        // Parallel loading for faster startup
      const [metadata, rawPalette] = await Promise.all([
        fetchJSON<Metadata>('data/metadata.json'),
        fetchJSON<unknown>('data/palette.json', {optional: true}),
      ]);
      if (!metadata) throw new Error('metadata.json missing or invalid JSON');
      this.metadata = metadata!;

      const {width, height} = this.metadata!.dimensions;
      this.resizeCanvasesForDPR(width, height, false);
        /*
        {
          "id": unifiedIndex=1,
          "bbox": {"x": ..., "y": ..., "width": ..., "height": ...},
          "centroid": {"x": ..., "y": ...},
          "mask": "shape_000.png"
        }
         */
      for (const region of metadata.regions) {//load regionss
        const mask = MASK_INDEX_RE.exec(pathnameOf(region.mask));
        if (!mask?.groups?.index) continue;
        const iii = parseInt(mask.groups.index, 10);
        this.displayIndexByPixelId.set(region.id, iii);
        this.regionByShapeIndex.set(iii, region);}


      if (rawPalette) {
        const normalized = normalizePaletteToRGB(rawPalette);
        this.palette = normalized as Palette;
        this.backgroundId = normalized.background_id | 0;
        this.paletteBackgroundIndex = this.backgroundId;
      } else {
        this.palette = {background_id: 0, map: {}};
        this.backgroundId = 0;
        this.paletteBackgroundIndex = 0}


      this.maskBackgroundIndex = this.config.maskBackgroundIndex ?? 0;
      //generate project data and populate regions map
      //load images in parallel
      this.generateProjectData();
      await Promise.all([
        this.loadBaseImage(),
        this.loadColouredMap(),
        this.loadUnifiedMask()
      ]);


        //COME BACK HERE
        // Set optimized event handler and hide loading indicator
      this.setupEventHandlers();
      document.getElementById('loading')?.style.setProperty('display', 'none');
      this.animate(0);}catch (err){
        //caught error
      console.error('init Failed: ', err)
      const loading = document.getElementById('loading');
      if (loading) {
        loading.textContent = err instanceof Error ? err.message : 'Failed to load mural data';
        loading.style.color = '#ff4444';
      }
    }
  }
  private tmp?: HTMLCanvasElement;
  private tmpCtx?: CanvasRenderingContext2D;


  private generateProjectData(): void {if (!this.metadata) return;

    const projectTypes: ProjectType[] = [
      { prefix: 'tower-', titles: ['Defence System', 'Watch Tower', 'Guard Post'] },
      { prefix: 'arch-', titles: ['Gateway', 'Portal', 'Entrance'] },
      { prefix: 'brick-', titles: ['Wall Section', 'Foundation', 'Battlement'] },
      { prefix: 'window-', titles: ['Lookout', 'Arrow Slit', 'Opening'] }
    ];

    for (const region of this.metadata.regions) {
      const type = projectTypes[region.id % projectTypes.length];
      const titleIdx = Math.floor(region.id / projectTypes.length) % type.titles.length;

      region.project = {
        title: `${type.titles[titleIdx]} #${region.id}`,
        href: `/projects/${type.prefix}${region.id}`,
        blurb: `Crown element ${region.id}: ${type.titles[titleIdx].toLowerCase()}`
      };

      // Store in map for O(1) lookup
      this.regions.set(region.id, region);
    }

    const regionCount = document.getElementById('region-count');
    if (regionCount) regionCount.textContent = String(this.regions.size);
  }


  private async loadBaseImage(): Promise<void> {
    const rawSource = url(this.config.baseImagePath ?? 'Mural_Crown_of_Italian_City.svg.png');

    const img = new Image();
    img.decoding = 'async';
    const source  =   new URL(rawSource, document.baseURI)
    if (source.origin !== location.origin) img.crossOrigin = 'anonymous';

      //set src
    img.src = source.href;
    try {
      if ('decode' in img) {
        await img.decode();
      } else {
        await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = () => reject(new Error('load error'));
        });
      }
    } catch {
      let ok = false
      let isImage = false;
      try{
        const r = await fetch(source.href, { method: 'HEAD', cache: 'no-store' });
        ok    =   r.ok;
        isImage   =   (r.headers.get('content-type') || '')
            .toLowerCase().startsWith('image/');
      }catch{}//distinguish 404 vs decode failure for clearer logs
      throw new Error(
        ok && isImage
            ? `Failed to decode base image at ${source} (format/corruption?).`
            : `Base image not found at ${source}. Ensure it lives in /public or is imported via ?url, and that the filename + case match exactly.`);
    }
  this.baseImage = img;
  //canvas dims alr sized + set DPR transform in init().
  this.needsRedraw = true;
}

  private colorForUnifiedIndex(idx: number): RGB {
    const m = this.palette?.map;
    if (!m || idx <= 0) return this.colorFallback;

    //palette.map keys are typically strings: "1", "2", ...
    const kStr = String(idx) as keyof typeof m;
    const kNum = idx as unknown as keyof typeof m; // belt-and-suspenders
    const c: any = (m as any)[kStr] ?? (m as any)[kNum];

    if (c && typeof c.r === 'number' && typeof c.g === 'number' && typeof c.b === 'number') {
      // clamp + coerce to int
      return { r: c.r | 0, g: c.g | 0, b: c.b | 0 };
    }
    return this.colorFallback}
  /**private async loadUnifiedMAsk1(): Promise<void> {
    const source = url(this.config.unifiedMaskPath || 'data/shape_masks/unified_mask.png');
    try {
        const res   =   await fetch(source, {cache:'no-store'});
        if (!res.ok)    throw new Error(`HTTP ${res.status} for ${source}`);

        const buffer    =   await res.arrayBuffer();
        const dView =   new DataView(buffer);
        const count =   buffer.byteLength   >>> 1;
        const out   =   new Uint16Array(count);

        for (let i=0,o=0; i <   count; i++,o+=2)    {
            out[i] =   dView.getUint16(o,true)}
        this.unifiedMaskIndex32=out;
        if (!this.metadata) throw new Error('metadata.json not loaded before unified mask');
        this.unifiedMaskHeight  = this.metadata.dimensions.height  | 0;
        this.unifiedMaskWidth  = this.metadata.dimensions.width  | 0;
        if (out.length !== this.unifiedMaskWidth * this.unifiedMaskHeight) {
          console.warn(`[unified_mask.u16] length mismatch: have ${out.length}, ` +
            `expected ${this.unifiedMaskWidth * this.unifiedMaskHeight}`);
        }

    } catch (err) {console.warn('Unified .u16 mask ' +
        'load failed; no per-pixel indices available:', err)}
  }*/
  private async loadUnifiedMask(): Promise<void> {
    const maskPath = this.config.unifiedMaskPath || 'data/shape_masks/unified_mask';
    if (!this.metadata) {const metaURL = this.config.metadataPath || 'data/metadata.json';
      const res = await fetch(metaURL, { cache: 'no-store' });
      if (!res.ok) throw new Error(`Failed to load metadata.json (HTTP ${res.status})`);
      this.metadata = await res.json();}
    const W = (this.unifiedMaskWidth  = this.metadata!.dimensions.width  | 0);
    const H = (this.unifiedMaskHeight = this.metadata!.dimensions.height | 0);
    const N = W * H;

    const loadBinary = async (ext: 'u32' | 'u16' | 'u8', expectedBytes: number) => {
      const urlStr = `${maskPath}.${ext}`;
      const res = await fetch(urlStr, { cache: 'no-store' });
      if (!res.ok) return null;
      const buf = await res.arrayBuffer();
      if (buf.byteLength !== expectedBytes) {return null;}//size mismatch – treat as not-found and try next
    return new DataView(buf)} // parse with DataView to control endianness
    try {
      let dv: DataView | null = null;

      dv = await loadBinary('u32', N * 4);
      if (dv) {
        const out = new Uint32Array(N);
        for (let i = 0, o = 0; i < N; i++, o += 4) out[i] = dv.getUint32(o, /*littleEndian=*/true);
        this.unifiedMaskIndex32 = out;
        return}

      dv = await loadBinary('u16', N * 2);
      if (dv) {
        const out = new Uint32Array(N);
          for (let i = 0, o = 0; i < N; i++, o += 2) out[i] = dv.getUint16(o, /*littleEndian=*/true);
        this.unifiedMaskIndex32 = out;
        return}

      dv = await loadBinary('u8', N);
      if (dv) {
        const out = new Uint32Array(N);//fast path: bulk view then widen
        const u8 = new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength);
        for (let i = 0; i < N; i++) out[i] = u8[i];
        this.unifiedMaskIndex32 = out;
        return}

      //PNG fallback - reconstruct 32 bits from RGBA
        await new Promise<void>((resolve) => {
          const img = new Image();
          const source = url(`${maskPath}.png`);
          if (new URL(source, document.baseURI).origin !== location.origin) img.crossOrigin = 'anonymous';
          img.onload = () => {
            const off = document.createElement('canvas');
            off.width = img.width;  off.height = img.height;
            const ctx = off.getContext('2d', { willReadFrequently: true, alpha: false })!;
            ctx.drawImage(img, 0, 0);
            const id = ctx.getImageData(0, 0, img.width, img.height);
            const W = id.width, H = id.height, src = id.data;
            this.unifiedMaskWidth = W; this.unifiedMaskHeight = H;

            //NOTE: Grayscale 32-bit becomes 8-bit in canvas; use R channel only (<=255 regions).
            const out = new Uint32Array(W * H);

            for (let i = 0, p = 0; i < out.length; i++, p += 4) {out[i] = (src[p+3] << 24) |  (src[p+2] << 16) | (src[p+1] << 8)  | src[p];} // R only
            this.unifiedMaskIndex32 = out;
            resolve()
          };
          img.onerror = () => { console.warn('Unified mask PNG fallback failed'); resolve(); };
          img.src = source});
    }catch (err) {console.error('loadUnifiedMask failed:', err);}
  }

    /**
     * return new Promise((resolve) => {
     *       const img = new Image();
     *
     *       img.decoding  =   'async'
     *
     *
     *       img.onload = async () => {
     *           //NOSCALING FOR NOW
     *         const off = document.createElement('canvas');
     *         off.width = img.width;
     *         off.height = img.height;
     *         const ctx = off.getContext('2d', { willReadFrequently: true, alpha: false });
     *         // @ts-ignore
     *           ctx.drawImage(img, 0, 0);
     *         // @ts-ignore
     *           let imagedata = ctx.getImageData(0, 0, img.width, img.height);
     *
     *
     *         const src   =   imagedata.data;
     *         const W = imagedata.width, H = imagedata.height;
     *         this.unifiedMaskWidth = W;
     *         this.unifiedMaskHeight = H;
     *
     *
     *         const out = new Uint32Array(W * H);
     *         for (let i = 0, p = 0; i < out.length; i++, p += 4) {
     *           const r = src[p + 0];
     *           const g = src[p + 1];
     *           const b = src[p + 2];
     *
     *           out[i] = (r) | (b << 8) | (g << 16)}
     *         this.unifiedMaskIndex32 = out;
     *         resolve()}
     *       img.onerror = () => {
     *         console.warn('Unified mask not found at', maskPath);
     *         resolve();
     *       };
     *       const source = url(maskPath);
     *       const sameOrigin = new URL(source, document.baseURI).origin === location.origin;
     *       if (!sameOrigin) img.crossOrigin = 'anonymous';
     *       img.src = source;
     *     })}
     *   }
     * @private
     */

  private async loadColouredMap(): Promise<void> {
    const source = url('data/coloured_map.png');
    const img = new Image();
    img.decoding = 'async';

    const sameOrigin = new URL(source, document.baseURI).origin === location.origin;
    if (!sameOrigin) img.crossOrigin = 'anonymous';

    img.src = source//safety
    try{await img.decode()}catch{throw new Error(`Failed to load coloured_map at ${source}`)}

    //draw to idCanvas then cache ImageData
    this.idCanvas.width = img.width;
    this.idCanvas.height = img.height;
    this.idCtx.drawImage(img, 0, 0);
    this.idImageData = this.idCtx.getImageData(0, 0, this.idCanvas.width, this.idCanvas.height);

    if (this.idImageData && this.metadata?.regions?.length) {
      const r0    =   this.metadata.regions[0];
      const testIdRGBPacked = (() => {
        const x = Math.floor(r0.centroid.x), y = Math.floor(r0.centroid.y);
        const idx = (y * this.idImageData.width + x) * 4, d = this.idImageData.data;
        return (d[idx] | (d[idx+1] << 8) | (d[idx+2] << 16) | (d[idx+3] << 24))})();
      const hasRGB = this.regions.has(testIdRGBPacked);

    }


  }
  //optimized ID reading using cached ImageData
  private readIdAt(clientX: number, clientY: number): number {
      if (!this.idImageData) return 0;

      const rect = this.canvas.getBoundingClientRect();
      const x = Math.floor((clientX - rect.left) * (this.canvas.width / rect.width));
      const y = Math.floor((clientY - rect.top) * (this.canvas.height / rect.height));

      // Bounds check
      if (x < 0 || x >= this.canvas.width || y < 0 || y >= this.canvas.height) {
          return 0
      }

      // Direct array access (much faster than getImageData per pixel)
      const idx = (y * this.idImageData.width + x) * 4;
      const data = this.idImageData.data;

      //return     (data[idx] << 16) | (data[idx + 1] << 8) | data[idx + 2];
      return     (data[idx+2] << 16) | (data[idx + 1] << 8) | data[idx];
  }

  private setupEventHandlers(): void {
    // Use passive listeners for better scrolling performance
    const options = { passive: true };
    
    // Throttled pointer move handler
    let rafScheduled = false;
    const handlePointerMove = (e: PointerEvent) => {
      this.latestEvent = e;
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(() => {
          this.processPointerEvents();
          rafScheduled = false;
        });
      }
    };
    
    this.canvas.addEventListener('pointermove', handlePointerMove, options);
    
    // High-frequency updates if supported
    if ('onpointerrawupdate' in this.canvas) {
      this.canvas.addEventListener('pointerrawupdate', handlePointerMove as any, options);
    }
    
    this.canvas.addEventListener('pointerleave', () => {
      this.latestEvent = null;
      this.clearHover();
    });
    
    this.canvas.addEventListener('click', () => {
      if (this.hoveredRegion) {
        const region = this.regions.get(this.hoveredRegion);
        if (region?.project) {
          console.log(`Navigate to: ${region.project.href}`);
        }
      }
    });
  }
  
  private processPointerEvents(): void {
    if (!this.latestEvent) return;
    
    // Get coalesced events for smooth detection
    const events = this.latestEvent.getCoalescedEvents?.() ?? [this.latestEvent];
    
    for (const evt of events) {
      const id = this.readIdAt(evt.clientX, evt.clientY);
      
      if (id !== this.prevId) {
        this.onRegionChange(id, evt.clientX, evt.clientY);
        this.prevId = id;
      }
    }
  }

  private maskCanvasCache = new Map<number, HTMLCanvasElement>();
  private async loadMaskImage(unifiedIndex: number): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => resolve(img);
      img.onerror = reject;
      const source    =   url(`data/shape_masks/shape_${String(unifiedIndex).padStart(3,'0')}.png`);
      const sameOrigin = new URL(source, document.baseURI).origin === location.origin;
      if (!sameOrigin) img.crossOrigin = 'anonymous';
      img.src = source;
    });
  }


  private maskCanvasPromiseCache    =   new Map<number,Promise<HTMLCanvasElement>>();
  private maskCanvasPending = new Map<number, Promise<HTMLCanvasElement>>();

  //takes in 1 index and conv to 0-based index shapeIndex
    // but unifiedIndex is 1-based
  private ensureMaskCanvas(unifiedIndex: number): HTMLCanvasElement | null {
    if (!this.unifiedMaskIndex32 || unifiedIndex <= 0) return null; // mask indices are 0-based; but region id 0 is background
    if (unifiedIndex === this.maskBackgroundIndex ||
        unifiedIndex === this.paletteBackgroundIndex) return null;

    let canvas = this.maskCanvasCache.get(unifiedIndex);
    if (canvas) return canvas;

    const unifiedWidth = this.unifiedMaskWidth | 0;
    const unifiedHeight = this.unifiedMaskHeight | 0;

    canvas  =   document.createElement('canvas');

    canvas.width = unifiedWidth;
    canvas.height = unifiedHeight;
    //const shapeIndex  =   unifiedIndex-1;
    //const img = await this.loadMaskImage(shapeIndex);
    const cctx = canvas.getContext('2d', { alpha: true })!;
    const img   =   cctx.createImageData(unifiedWidth,unifiedHeight);
    //draw mask as alpha: assume mask png is white-on-black turn white to opaque
    //cctx.drawImage(img, 0, 0);
    //const id = cctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const source    =   this.unifiedMaskIndex32;
    //treat any nonzero as part of mask put in alpha channel
    for (let i = 0; i < data.length; i += 4) {
      const v = data[i] | data[i+1] | data[i+2];
      data[i] = data[i+1] = data[i+2] = 255;  //draw highlight color later via globalCompositeOperation
      data[i+3] = v ? 255 : 0}              //alpha
    //RETURN HERE
    cctx.putImageData(id, 0, 0);
    this.maskCanvasCache.set(unifiedIndex, canvas);
    return canvas}
  private lastRequestedHighlight = 0;
  private async compositeHighlight(unifiedIndex: number): Promise<void> {
    const requestId = ++this.lastRequestedHighlight;
    const oc = this.overlayCtx;
    oc.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (unifiedIndex <= 0 || unifiedIndex === this.maskBackgroundIndex ||
    unifiedIndex === this.paletteBackgroundIndex) {
      this.overlayDirty = null;
      this.needsRedraw = true;
      return;
    }

    //build / fetch alpha-mask canvas for index
    const maskCanvas = await this.ensureMaskCanvas(unifiedIndex);
    if (!maskCanvas) {
      this.overlayDirty = null;
      this.needsRedraw = true;
      return;
    }
    if (requestId !== this.lastRequestedHighlight) return;
    const color = this.colorForUnifiedIndex(unifiedIndex);

    oc.globalCompositeOperation = 'source-over';//paind solid color overlay
    oc.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    oc.fillRect(0, 0, this.overlay.width, this.overlay.height);

    //intersect w/ mask alpha
    oc.globalCompositeOperation = 'destination-in';
    // @ts-ignore
      oc.drawImage(maskCanvas, 0, 0);

    //reset to normal draw mode
    oc.globalCompositeOperation = 'source-over';

    //dirty-rect from metadata (optional micro-optimization)
    const region = this.regionByShapeIndex.get(unifiedIndex - 1);
    this.overlayDirty = region
      ? { x: region.bbox.x, y: region.bbox.y, width: region.bbox.width, height: region.bbox.height }
      : { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height };
    this.needsRedraw = true}
  private clearOverlay() {this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height)}


  private async drawComposite(regionIds: number[]) {
    if (!regionIds?.length) { this.clearOverlay(); return; }
    const ctx   =   this.overlayCtx;
    this.clearOverlay();
    ctx.globalCompositeOperation    =  'source-over';
    const maskColl    =   [];
    //resolve -> mask indices (0-based iii)

    for (const unifiedIndex of regionIds) {
      if (unifiedIndex <= 0) continue; // skip background
      const mask = this.displayIndexByPixelId.get(unifiedIndex);
      if (mask == null) continue;
      const c = await this.ensureMaskCanvas(mask);
      if (c) maskColl.push(c);
    }
    const ctx = this.overlayCtx;
    this.clearOverlay();
    ctx.globalCompositeOperation = 'source-over';
    // Optional: tint color per region via palette.json
    for (let i = 0; i < maskColl.length; i++) {ctx.globalAlpha = 0.25; //soft composite
      ctx.drawImage(maskColl[i], 0, 0)}
    ctx.globalAlpha = 1;
  }



  private async onRegionChange(id: number, clientX: number, clientY: number) {
    this.hoveredRegion= id > 0 ? id : null;
    this.updateTooltip(this.hoveredRegion ? this.regions.get(this.hoveredRegion) ?? null : null, clientX, clientY);
    //choose composite set: hovered + close neighbors
    // by centroid distance
    const composite: number[] = [];
    if (id > 0) {
      composite.push(id);
      const base = this.regions.get(id);
      if (base) {
        const {x: cx, y: cy} = base.centroid as any;
        // naive neighbor search; replace with spatial grid if needed
        for (const [otherId, r] of this.regions) {if (otherId === id) continue;
          const dx = (r.centroid as any).x - cx;
          const dy = (r.centroid as any).y - cy;
          if ((dx * dx + dy * dy) <= (40 * 40)) composite.push(otherId);
          }}}//keep small  ```if (composite.length >= 6) break```
    await this.drawComposite(composite);
    this.needsRedraw = true}
  private clearHover(): void {
    if (this.hoveredRegion !== null) {
      this.hoveredRegion = null;
      this.prevId = -1;
      this.tooltip.style.display = 'none';
      const currentRegion = document.getElementById('current-region');
      if (currentRegion) currentRegion.textContent = '-';
      this.compositeHighlight(0);
      this.needsRedraw = true;
    }
  }



  private animate = (ts: number) => {
    this.animationId = requestAnimationFrame(this.animate);

    if (this.fpsElement && ts - this.fpsLastUpdate > 250) {
      if (this.lastTimestamp) {
        const delta = ts - this.lastTimestamp;           //ts RAF timestamp
        const fps = Math.max(1, Math.round(1000 / delta));
        this.fpsElement.textContent = String(fps);            //batch single write
      }this.fpsLastUpdate = ts}
    this.lastTimestamp = ts;

    if (!this.needsRedraw) return;
    this.needsRedraw = false;

    if (this.baseImage) {
      this.ctx.drawImage(this.baseImage, 0, 0);
    } else {this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)}


    //overlay (precomposited on own canvas) w/ opt dirty rect clip
    if (this.overlayDirty) {
      const { x, y, width, height } = this.overlayDirty;
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.rect(x, y, width, height);
      this.ctx.clip();
      this.ctx.drawImage(this.overlay, 0, 0);
      this.ctx.restore();
      } else {this.ctx.drawImage(this.overlay, 0, 0)}
    //HUD (bbox + centroid label) for hovered region.
    if(this.hoveredRegion !== null) {
      const region = this.regions.get(this.hoveredRegion);
      if(region) {
          this.ctx.save();

          const {x, y, width, height} = region.bbox;
          this.ctx.shadowColor = 'gold';
          this.ctx.shadowBlur = 20;
          this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
          this.ctx.lineWidth = 2;
          this.ctx.strokeRect(x, y, width, height);

          const {x: cx, y: cy} = region.centroid;
          this.ctx.shadowBlur = 5;
          this.ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
          this.ctx.fillRect(cx - 25, cy - 10, 50, 20);

          this.ctx.shadowBlur = 0;
          this.ctx.fillStyle = 'gold';
          this.ctx.font = 'bold 11px sans-serif';
          this.ctx.textAlign = 'center';
          this.ctx.textBaseline = 'middle';
          this.ctx.fillText(`#${this.hoveredRegion}`, cx, cy);

          this.ctx.restore()}
    }


    this.needsRedraw = false;
    // Base: if you have a base mural image, draw it here; else clear
    //this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    // Then draw overlay compositing
    //this.ctx.drawImage(this.overlay, 0, 0);
  };
  private updateTooltip(region: Region | null, x: number, y: number) {
    if (region?.project) {
      this.tooltipTitle.textContent = region.project.title;
      this.tooltipBlurb.textContent = region.project.blurb;
      this.tooltip.style.display = 'block';
      //left/top fine and transform translate() can reduce layout work if you animate it
      this.tooltip.style.left = `${x}px`;
      this.tooltip.style.top  = `${y}px`;
    } else {
      this.tooltip.style.display = 'none';
    }
  }
  public destroy(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}