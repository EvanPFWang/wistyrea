// src/CrownMuralController.ts
import {RGB, Hex, Palette, Metadata,Region, ProjectType} from "./types.ts";

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
  private backgroundId: number  =   0;
  // Animation frame
  private animationId: number = 0;
  private lastTimestamp: number = 0;
  
  // Precomputed lookup table for ID decoding
  // @ts-ignore
    private readonly idLookup = new Uint32Array(256 * 256 * 256);
  
  constructor() {
    this.canvas = document.getElementById('mural-canvas') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d', {
      alpha: false,
      desynchronized: true // Hint for better performance
    })!;
    
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
      fetchJSON<unknown>('data/palette.json', { optional: true }),
    ]);
      this.metadata = metadata!;
    if (rawPalette) {
      const normalized: Palette = normalizePaletteToRGB(rawPalette);
      this.palette = normalized as Palette;
      this.backgroundId = normalized.background_id;
    } else {
        this.palette = { background_id: 0, map: {} };
    }
    // Generate project data and populate regions map
    // Load images in parallel
    this.generateProjectData();
    await Promise.all([
    this.loadBaseImage(),
    this.loadIDMap()
    ]);
    // Set optimized event handler and hide loading indicator
    this.setupEventHandlers();
    const loading = document.getElementById('loading');
    if (loading) loading.style.display = 'none';
    //start render loop
    this.animate(0);
    } catch (err) {
      console.error('Initialization failed:', err);
      const loading = document.getElementById('loading');
      if (loading) {
        loading.textContent = 'Failed to load mural data';
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
  
  private async loadBaseImage(): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.baseImage = img;
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load colored regions'));
      img.src = url('Mural_Crown_of_Italian_City.svg.png');
    });
  }
  
  private async loadIDMap(): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.idCanvas.width = img.width;
        this.idCanvas.height = img.height;
        this.idCtx.imageSmoothingEnabled = false;
        this.idCtx.drawImage(img, 0, 0);
        
        // Pre-cache entire ID map for ultra-fast lookups
        this.idImageData = this.idCtx.getImageData(0, 0, img.width, img.height);
        
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load ID map'));
      img.src = url('data/id_map.png');
    });
  }
  
  // Optimized ID reading using cached ImageData
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
    
    // Decode from BGR (OpenCV format) to ID
    return (data[idx + 2] << 16) | (data[idx + 1] << 8) | data[idx];
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
  
  private onRegionChange(id: number, screenX: number, screenY: number): void {
    this.hoveredRegion = id > 0 ? id : null;
    
    const currentRegion = document.getElementById('current-region');
    if (currentRegion) {
      currentRegion.textContent = this.hoveredRegion ? `#${this.hoveredRegion}` : '-';
    }
    
    if (this.hoveredRegion) {
      const region = this.regions.get(this.hoveredRegion);
      if (region?.project) {
        this.tooltipTitle.textContent = region.project.title;
        this.tooltipBlurb.textContent = region.project.blurb;
        this.tooltip.style.display = 'block';
        this.tooltip.style.left = `${screenX}px`;
        this.tooltip.style.top = `${screenY}px`;
      }
    } else {
      this.tooltip.style.display = 'none';
    }
    
    this.needsRedraw = true;
  }
  
  private clearHover(): void {
    if (this.hoveredRegion !== null) {
      this.hoveredRegion = null;
      this.prevId = -1;
      this.tooltip.style.display = 'none';
      const currentRegion = document.getElementById('current-region');
      if (currentRegion) currentRegion.textContent = '-';
      this.needsRedraw = true;
    }
  }
  
  private render(): void {
    if (this.baseImage) {
      this.ctx.drawImage(this.baseImage, 0, 0);
    }
    
    if (this.hoveredRegion !== null) {
      const region = this.regions.get(this.hoveredRegion);
      if (region) {
        this.ctx.save();
        
        const { x, y, width, height } = region.bbox;
        this.ctx.shadowColor = 'gold';
        this.ctx.shadowBlur = 20;
        this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, width, height);
        
        const { x: cx, y: cy } = region.centroid;
        this.ctx.shadowBlur = 5;
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        this.ctx.fillRect(cx - 25, cy - 10, 50, 20);
        
        this.ctx.shadowBlur = 0;
        this.ctx.fillStyle = 'gold';
        this.ctx.font = 'bold 11px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`#${this.hoveredRegion}`, cx, cy);
        
        this.ctx.restore();
      }
    }
  }
  
  private animate(timestamp: number): void {
    // Calculate FPS
    if (this.lastTimestamp) {
      const delta = timestamp - this.lastTimestamp;
      const fps = Math.round(1000 / delta);
      const fpsElement = document.getElementById('fps');
      if (fpsElement) fpsElement.textContent = String(fps);
    }
    this.lastTimestamp = timestamp;
    
    // Only render if needed
    if (this.needsRedraw) {
      this.render();
      this.needsRedraw = false;
    }
    
    this.animationId = requestAnimationFrame((t) => this.animate(t));
  }
  
  public destroy(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}