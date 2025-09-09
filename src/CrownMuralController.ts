// src/CrownMuralController.ts
import {RGB, Hex, Palette, Metadata,Region, ProjectType, ControllerConfig} from "./types.ts";

const BASE = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '/');
const url = (p: string) => `${BASE}${p.replace(/^\/+/, '')}`;
// helper with good errors + optional null

async function fetchJSON<T>(path: string, { optional = false } = {}): Promise<T | null> {
  const res = await fetch(url(path), { cache: 'no-cache' }); // or 'force-cache' for CDN caching
  if (!res.ok) {
    if (optional && res.status === 404) return null;
    throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}
const HEX_RE = /^#?(?<r>[0-9a-fA-F]{2})(?<g>[0-9a-fA-F]{2})(?<b>[0-9a-fA-F]{2})$/;
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

export class CrownMuralController {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private idCanvas: HTMLCanvasElement;
  private idCtx: CanvasRenderingContext2D;
  private tooltip: HTMLDivElement;
  private tooltipTitle: HTMLElement;
  private tooltipBlurb: HTMLElement;
  private config;
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
  private unifiedMaskIndex16: Uint16Array | null = null;//region indices per pixel loaded from the unified mask
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


    private resizeCanvasesForDPR(w: number, h: number) {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const set = (c: HTMLCanvasElement, ctx:
                  CanvasRenderingContext2D, alpha=false) => {
        c .width  = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
        c.style.width  = w + 'px';
        c.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)}
      set(this.canvas,  this.ctx);
      set(this.idCanvas, this.idCtx);
      set(this.overlay, this.overlayCtx, true)}

    constructor(config: ControllerConfig={}) {

    this.config = config;
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
      this.resizeCanvasesForDPR(width, height);

      for (const region of metadata.regions) {//load images
        const m = /(shape|mask)_(\d+)\.png$/i.exec(region.mask);//old mask_(\d+)\
        if (m) {const iii = parseInt(m[2], 10);
          this.displayIndexByPixelId.set(region.id, iii);   //pixel-id → unifiedIndex
          this.regionByShapeIndex.set(iii, region)}         //unifiedIndex → Region
        }
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
        this.loadIDMap(),
        this.loadUnifiedMask()
      ]);




        //COME BACK HERE
        // Set optimized event handler and hide loading indicator
      this.setupEventHandlers();
      document.getElementById('loading')?.style.setProperty('display', 'none');
      this.animate(0);}
    catch (err){
      console.error('init Failed: ', err)
      const loading = document.getElementById('loading');
      if (loading) {
        loading.textContent = err instanceof Error ? err.message : 'Failed to load mural data';
        loading.style.color = '#ff4444';
      }
    }
  }

  private generateProjectData(): void {
    if (!this.metadata) return;

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
  /**
  private async loadBaseImage(): Promise<void> {
    await new Promise<void>((resolve, reject) => {

        const img = new Image();

      img.onload = () => {
          this.baseImage = img;
          this.canvas.width = img.width;
          this.canvas.height = img.height;
          this.overlay.width = img.width;
          this.overlay.height = img.height;

          this.ctx.drawImage(img, 0, 0);
          this.needsRedraw = true;//draw once
          resolve()};
      img.onerror = () => reject(new Error('Failed to load base image'));
      img.src = url('Mural_Crown_of_Italian_City.svg.png'); //[HELLO]base png decision
    });
  }*/

  private async loadBaseImage(): Promise<void>{
    const src =   url(this.config.baseImagePath ?? 'Mural_Crown_of_Italian_City.svg.png');

    const img =   new Image();
    img.decoding  =   'async'
    //if BASE_URL points to diff origin (e.g., CDN) enable CORS
    if (!src.startsWith(location.origin)) img.crossOrigin = 'anonymous';

    img.src =   src;
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
    const ok = await fetch(src, { method: 'HEAD', cache: 'no-store' })
      .then(r => r.ok).catch(() => false);
    throw new Error(ok? `Failed to decode base image at ${src} (format/corruption?).`
        : `Base image not found at ${src}. Ensure it lives in /public or is imported via ?url, and that the filename + case match exactly.`)}

    this.baseImage  =   img;
    this.needsRedraw    =   true;// dont reset canvas widths since alr sized + set DPR transform in init()
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
  private async loadUnifiedMask(): Promise<void> {
    const maskPath = this.config.unifiedMaskPath || 'data/shape_masks/unified_mask.png';
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const off = document.createElement('canvas');
        off.width = img.width;
        off.height = img.height;
        const ctx = off.getContext('2d', { willReadFrequently: true, alpha: false });
        // @ts-ignore
          ctx.drawImage(img, 0, 0);

          let id: any;
          // @ts-ignore
          id = ctx.getImageData(0, 0, img.width, img.height);
        const src   =   id.data;
        const W = id.width, H = id.height;
        this.unifiedMaskWidth = W;
        this.unifiedMaskHeight = H;


        const out = new Uint16Array(W * H);
        for (let i = 0, p = 0; i < out.length; i++, p += 4) {out[i] = (src[p] | (src[p + 1] << 8)) & 0xffff}
        this.unifiedMaskIndex16 = out;
        resolve();
      };
      img.onerror = () => {
        console.warn('Unified mask not found at', maskPath);
        resolve();
      };
      img.src = url(maskPath);
    });
  }



  private async loadIDMap(): Promise<void> {
    const src = url('data/id_map.png');
    const img = new Image();
    img.decoding = 'async';
    img.src = src//safety
    try{await img.decode()}catch{throw new Error(`Failed to load id_map at ${src}`)}

    //draw to idCanvas then cache ImageData
    this.idCanvas.width = img.width;
    this.idCanvas.height = img.height;
    this.idCtx.drawImage(img, 0, 0);
    this.idImageData = this.idCtx.getImageData(0, 0, this.idCanvas.width, this.idCanvas.height);
  }
  //optimized ID reading using cached ImageData
  private readIdAt(clientX: number, clientY: number): number {
    if (!this.idImageData) return 0;
    
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left) * (this.canvas.width / rect.width));
    const y = Math.floor((clientY - rect.top) * (this.canvas.height / rect.height));
    
    // Bounds check
    if (x < 0 || x >= this.canvas.width || y < 0 || y >= this.canvas.height) {
      return 0;
    }
    
    // Direct array access (much faster than getImageData per pixel)
    const idx = (y * this.idImageData.width + x) * 4;
    const data = this.idImageData.data;

    const bgrToId   =    (data[idx + 2] << 16) | (data[idx + 1] << 8) | data[idx];
    // Decode from BGR to ID
    return bgrToId;
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
      img.src = url(`data/shape_masks/shape_${String(unifiedIndex).padStart(3,'0')}.png`);
    });
  }



  private async ensureMaskCanvas(unifiedIndex: number): Promise<HTMLCanvasElement | null> {
    if (!this.unifiedMaskIndex16 || unifiedIndex <= 0) return null; // mask indices are 0-based; but region id 0 is background
    if (unifiedIndex === this.maskBackgroundIndex || unifiedIndex === this.paletteBackgroundIndex) return null;

    let c = this.maskCanvasCache.get(unifiedIndex);
    if (c) return c;

    const img = await this.loadMaskImage(unifiedIndex);
    c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const cctx = c.getContext('2d', { alpha: true })!;
    //draw mask as alpha: assume mask png is white-on-black turn white to opaque
    cctx.drawImage(img, 0, 0);
    const id = cctx.getImageData(0, 0, c.width, c.height);
    const data = id.data;
    //treat any nonzero as part of mask put in alpha channel
    for (let i = 0; i < data.length; i += 4) {const v = data[i] | data[i+1] | data[i+2];
      data[i] = data[i+1] = data[i+2] = 255;  //draw highlight color later via globalCompositeOperation
      data[i+3] = v ? 255 : 0}              //alpha

    cctx.putImageData(id, 0, 0);
    this.maskCanvasCache.set(unifiedIndex, c);
    return c}
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

    //resolve -> mask indices (0-based iii)
    const masks = [];
    for (const id of regionIds) {
      if (id <= 0) continue; // skip background
      const iii = this.displayIndexByPixelId.get(id);
      if (iii == null) continue;
      const c = await this.ensureMaskCanvas(iii);
      if (c) masks.push(c);
    }
    const ctx = this.overlayCtx;
    this.clearOverlay();
    ctx.globalCompositeOperation = 'source-over';
    // Optional: tint color per region via palette.json
    for (let i = 0; i < masks.length; i++) {ctx.globalAlpha = 0.25; //soft composite
      ctx.drawImage(masks[i], 0, 0)}
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