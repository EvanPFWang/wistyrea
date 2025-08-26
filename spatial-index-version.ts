import RBush from 'rbush';

// Spatial index item structure
interface SpatialItem {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  id: string;
  element: SVGGraphicsElement;
}

// Enhanced mural controller with spatial indexing
export class OptimizedMuralController {
  private spatialIndex: RBush<SpatialItem>;
  private svg: SVGSVGElement;
  private svgPt: DOMPoint;
  private worker: Worker | null = null;
  
  // Cache frequently accessed elements
  private elementCache = new Map<string, {
    bbox: DOMRect;
    path: string;
  }>();

  constructor() {
    this.spatialIndex = new RBush();
    this.svg = document.getElementById("mural") as SVGSVGElement;
    this.svgPt = this.svg.createSVGPoint();
    
    this.init();
  }

  private async init(): Promise<void> {
    // Build spatial index
    await this.buildSpatialIndex();
    
    // Initialize web worker for heavy computations
    if (window.Worker) {
      this.initWorker();
    }
    
    // Set up optimized event handling
    this.setupEventHandlers();
  }

  // Build R-tree index from SVG elements
  private async buildSpatialIndex(): Promise<void> {
    const elements = this.svg.querySelectorAll('[data-id]');
    const items: SpatialItem[] = [];
    
    // Process in chunks to avoid blocking
    const chunkSize = 100;
    for (let i = 0; i < elements.length; i += chunkSize) {
      const chunk = Array.from(elements).slice(i, i + chunkSize);
      
      await new Promise(resolve => {
        requestAnimationFrame(() => {
          chunk.forEach(el => {
            const svgEl = el as SVGGraphicsElement;
            const bbox = svgEl.getBBox();
            const id = el.getAttribute('data-id')!;
            
            // Cache element data
            this.elementCache.set(id, {
              bbox: bbox,
              path: el.getAttribute('d') || ''
            });
            
            // Add to spatial index
            items.push({
              minX: bbox.x,
              minY: bbox.y,
              maxX: bbox.x + bbox.width,
              maxY: bbox.y + bbox.height,
              id: id,
              element: svgEl
            });
          });
          resolve(null);
        });
      });
    }
    
    // Bulk insert for better performance
    this.spatialIndex.load(items);
    console.log(`Indexed ${items.length} elements`);
  }

  // Convert screen coords to SVG coords
  private screenToSVG(x: number, y: number): DOMPoint {
    this.svgPt.x = x;
    this.svgPt.y = y;
    return this.svgPt.matrixTransform(
      this.svg.getScreenCTM()!.inverse()
    );
  }

  // Fast hit testing using spatial index
  private hitTest(x: number, y: number): SpatialItem | null {
    const svgCoord = this.screenToSVG(x, y);
    
    // Query spatial index for candidates (tiny bbox around cursor)
    const candidates = this.spatialIndex.search({
      minX: svgCoord.x - 1,
      minY: svgCoord.y - 1,
      maxX: svgCoord.x + 1,
      maxY: svgCoord.y + 1
    });
    
    // No candidates in area
    if (candidates.length === 0) return null;
    
    // Single candidate - quick check
    if (candidates.length === 1) {
      return this.pointInElement(svgCoord, candidates[0]) ? candidates[0] : null;
    }
    
    // Multiple candidates - find topmost
    return this.findTopmostHit(svgCoord, candidates);
  }

  // Check if point is inside element
  private pointInElement(pt: DOMPoint, item: SpatialItem): boolean {
    // Fast bbox check first
    const cache = this.elementCache.get(item.id);
    if (!cache) return false;
    
    const bbox = cache.bbox;
    if (pt.x < bbox.x || pt.x > bbox.x + bbox.width ||
        pt.y < bbox.y || pt.y > bbox.y + bbox.height) {
      return false;
    }
    
    // Precise check for complex shapes
    if (item.element instanceof SVGPathElement) {
      return item.element.isPointInFill(pt);
    }
    
    return true;
  }

  // Find topmost element when multiple candidates
  private findTopmostHit(pt: DOMPoint, candidates: SpatialItem[]): SpatialItem | null {
    // Sort by DOM order (later elements are on top)
    const hits = candidates
      .filter(item => this.pointInElement(pt, item))
      .sort((a, b) => {
        return a.element.compareDocumentPosition(b.element) & 
               Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
      });
    
    return hits[hits.length - 1] || null;
  }

  // Initialize web worker for heavy computations
  private initWorker(): void {
    const workerCode = `
      // Point-in-polygon test (ray casting)
      function pointInPolygon(x, y, points) {
        let inside = false;
        for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
          const xi = points[i][0], yi = points[i][1];
          const xj = points[j][0], yj = points[j][1];
          const intersect = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
          if (intersect) inside = !inside;
        }
        return inside;
      }
      
      // Handle messages
      self.onmessage = (e) => {
        const { type, data } = e.data;
        
        switch (type) {
          case 'hit-test':
            const result = pointInPolygon(data.x, data.y, data.points);
            self.postMessage({ type: 'hit-result', result, id: data.id });
            break;
        }
      };
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
  }

  // Optimized event handlers
  private setupEventHandlers(): void {
    let rafId = 0;
    let lastHit: string | null = null;
    
    // Use passive listeners for better scrolling performance
    this.svg.addEventListener('pointermove', (e) => {
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          const hit = this.hitTest(e.clientX, e.clientY);
          
          if (hit?.id !== lastHit) {
            lastHit = hit?.id || null;
            this.updateUI(hit, e.clientX, e.clientY);
          }
        });
      }
    }, { passive: true });
    
    this.svg.addEventListener('pointerleave', () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      lastHit = null;
      this.clearUI();
    });
  }

  // Update UI based on hit
  private updateUI(hit: SpatialItem | null, x: number, y: number): void {
    if (!hit) {
      this.clearUI();
      return;
    }
    
    // Update highlight and tooltip
    // (Implementation depends on your UI structure)
    console.log(`Hovering: ${hit.id}`);
  }

  private clearUI(): void {
    // Clear highlights and tooltips
    console.log('Clear hover');
  }

  // Public API for dynamic updates
  public addElement(el: SVGGraphicsElement, id: string): void {
    const bbox = el.getBBox();
    this.spatialIndex.insert({
      minX: bbox.x,
      minY: bbox.y,
      maxX: bbox.x + bbox.width,
      maxY: bbox.y + bbox.height,
      id: id,
      element: el
    });
  }

  public removeElement(id: string): void {
    const item = this.spatialIndex.all().find(i => i.id === id);
    if (item) {
      this.spatialIndex.remove(item);
      this.elementCache.delete(id);
    }
  }

  // Clean up
  public destroy(): void {
    if (this.worker) {
      this.worker.terminate();
    }
    this.spatialIndex.clear();
    this.elementCache.clear();
  }
}