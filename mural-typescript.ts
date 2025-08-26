// Interfaces for 436-region crown mural
interface RegionMetadata {
  id: number;
  mask: string;
  bbox: { x: number; y: number; width: number; height: number };
  centroid: { x: number; y: number };
  project: ProjectMeta;
}

interface ProjectMeta {
  title: string;
  href: string;
  blurb: string;
  tags?: string[];
}

interface MuralMetadata {
  version: string;
  dimensions: { width: number; height: number };
  total_regions: number;
  base_image: string;
  id_map: string;
  regions: RegionMetadata[];
}

// Canvas-based controller optimized for 436 regions
export class CrownMuralController {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private idCanvas: HTMLCanvasElement;
  private idCtx: CanvasRenderingContext2D;
  
  // Data structures
  private metadata: MuralMetadata | null = null;
  private regions = new Map<number, RegionMetadata>();
  private hoveredRegion: number | null = null;
  
  // Image resources
  private baseImage: HTMLImageElement | null = null;
  private idMapImage: HTMLImageElement | null = null;
  
  // Performance
  private rafId = 0;
  private needsRedraw = false;
  private lastPointer = { x: -1, y: -1 };
  
  constructor(private dataPath = './web_data/') {
    // Get canvases
    this.canvas = document.getElementById('mural-canvas') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d')!;
    this.idCanvas = document.getElementById('id-canvas') as HTMLCanvasElement;
    this.idCtx = this.idCanvas.getContext('2d', { willReadFrequently: true })!;
    
    this.init();
  }
  
  private async init(): Promise<void> {
    try {
      // Load metadata
      await this.loadMetadata();
      
      // Load images
      await Promise.all([
        this.loadBaseImage(),
        this.loadIDMap()
      ]);
      
      // Setup interaction
      this.setupEventHandlers();
      
      // Start render loop
      this.animate();
      
      // Hide loading indicator
      const loading = document.getElementById('loading');
      if (loading) loading.style.display = 'none';
      
    } catch (err) {
      console.error('Failed to initialize mural:', err);
      this.showError('Failed to load mural data');
    }
  }
  
  private async loadMetadata(): Promise<void> {
    const response = await fetch(`${this.dataPath}metadata.json`);
    this.metadata = await response.json();
    
    // Build region map
    for (const region of this.metadata!.regions) {
      this.regions.set(region.id, region);
    }
    
    // Update stats
    const countEl = document.getElementById('region-count');
    if (countEl) countEl.textContent = String(this.regions.size);
  }
  
  private async loadBaseImage(): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.baseImage = img;
        
        // Set canvas dimensions
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        
        // Initial draw
        this.ctx.drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load base image'));
      img.src = `${this.dataPath}../coloured_regions.png`;
    });
  }
  
  private async loadIDMap(): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.idMapImage = img;
        
        // Set ID canvas dimensions
        this.idCanvas.width = img.width;
        this.idCanvas.height = img.height;
        
        // Draw ID map to hidden canvas
        this.idCtx.drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load ID map'));
      img.src = `${this.dataPath}id_map.png`;
    });
  }
  
  private setupEventHandlers(): void {
    // Pointer move with throttling
    this.canvas.addEventListener('pointermove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) * (this.canvas.width / rect.width));
      const y = Math.floor((e.clientY - rect.top) * (this.canvas.height / rect.height));
      
      // Skip if same position
      if (x === this.lastPointer.x && y === this.lastPointer.y) return;
      
      this.lastPointer = { x, y };
      
      // Throttle with rAF
      if (!this.rafId) {
        this.rafId = requestAnimationFrame(() => {
          this.rafId = 0;
          this.handlePointerMove(x, y, e.clientX, e.clientY);
        });
      }
    });
    
    // Pointer leave
    this.canvas.addEventListener('pointerleave', () => {
      this.clearHover();
    });
    
    // Click handler
    this.canvas.addEventListener('click', () => {
      if (this.hoveredRegion !== null) {
        const region = this.regions.get(this.hoveredRegion);
        if (region?.project.href) {
          console.log(`Navigate to: ${region.project.href}`);
          // window.location.href = region.project.href;
        }
      }
    });
    
    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === 'r') this.redraw();
      if (e.key === 'd') this.toggleDebug();
    });
  }
  
  private handlePointerMove(canvasX: number, canvasY: number, 
                           screenX: number, screenY: number): void {
    // O(1) region detection using ID map
    const regionId = this.getRegionAtPoint(canvasX, canvasY);
    
    if (regionId !== this.hoveredRegion) {
      this.hoveredRegion = regionId;
      this.updateTooltip(regionId, screenX, screenY);
      this.needsRedraw = true;
    }
  }
  
  private getRegionAtPoint(x: number, y: number): number | null {
    // Read pixel from ID map
    const pixel = this.idCtx.getImageData(x, y, 1, 1).data;
    
    // Decode region ID from RGB
    const id = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];
    
    // Check if valid region
    return this.regions.has(id) ? id : null;
  }
  
  private updateTooltip(regionId: number | null, x: number, y: number): void {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;
    
    if (regionId === null) {
      tooltip.style.display = 'none';
      return;
    }
    
    const region = this.regions.get(regionId);
    if (!region) return;
    
    // Update content
    const title = tooltip.querySelector('strong');
    const blurb = tooltip.querySelector('.blurb');
    if (title) title.textContent = region.project.title;
    if (blurb) blurb.textContent = region.project.blurb;
    
    // Position tooltip
    tooltip.style.display = 'block';
    tooltip.style.left = `${x + 10}px`;
    tooltip.style.top = `${y - 10}px`;
  }
  
  private clearHover(): void {
    if (this.hoveredRegion !== null) {
      this.hoveredRegion = null;
      this.updateTooltip(null, 0, 0);
      this.needsRedraw = true;
    }
  }
  
  private render(): void {
    // Redraw base image
    if (this.baseImage) {
      this.ctx.drawImage(this.baseImage, 0, 0);
    }
    
    // Highlight hovered region
    if (this.hoveredRegion !== null) {
      this.highlightRegion(this.hoveredRegion);
    }
  }
  
  private highlightRegion(regionId: number): void {
    const region = this.regions.get(regionId);
    if (!region) return;
    
    // Create highlight effect using compositing
    this.ctx.save();
    
    // Draw highlight box around region
    const { x, y, width, height } = region.bbox;
    
    // Glow effect
    this.ctx.shadowColor = 'gold';
    this.ctx.shadowBlur = 20;
    this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x, y, width, height);
    
    // Label at centroid
    this.ctx.shadowBlur = 5;
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    this.ctx.fillRect(
      region.centroid.x - 40,
      region.centroid.y - 10,
      80,
      20
    );
    
    this.ctx.shadowBlur = 0;
    this.ctx.fillStyle = 'gold';
    this.ctx.font = '12px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(
      `#${regionId}`,
      region.centroid.x,
      region.centroid.y
    );
    
    this.ctx.restore();
  }
  
  private redraw(): void {
    this.needsRedraw = true;
  }
  
  private toggleDebug(): void {
    const stats = document.getElementById('stats');
    if (stats) {
      stats.style.display = stats.style.display === 'none' ? 'block' : 'none';
    }
  }
  
  private animate(timestamp = 0): void {
    // Update FPS counter
    const fpsEl = document.getElementById('fps');
    if (fpsEl && timestamp) {
      const fps = Math.round(1000 / 16); // Approximate
      fpsEl.textContent = String(fps);
    }
    
    // Render if needed
    if (this.needsRedraw) {
      this.render();
      this.needsRedraw = false;
    }
    
    requestAnimationFrame((t) => this.animate(t));
  }
  
  private showError(message: string): void {
    const loading = document.getElementById('loading');
    if (loading) {
      loading.textContent = message;
      loading.style.color = '#ff4444';
    }
  }
  
  // Public API
  public highlightRegionById(id: number): void {
    if (this.regions.has(id)) {
      this.hoveredRegion = id;
      this.needsRedraw = true;
    }
  }
  
  public getRegionInfo(id: number): RegionMetadata | undefined {
    return this.regions.get(id);
  }
  
  public destroy(): void {
    // Clean up event handlers
    this.canvas.removeEventListener('pointermove', () => {});
    this.canvas.removeEventListener('pointerleave', () => {});
    this.canvas.removeEventListener('click', () => {});
    
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
  }
}

// Auto-initialize on DOM ready
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    const mural = new CrownMuralController();
    
    // Expose for debugging
    (window as any).mural = mural;
  });
}