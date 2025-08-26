// Point-in-polygon worker for offloading heavy computations
// src/workers/pip.worker.ts

interface HitTestRequest {
  type: 'hit-test';
  id: string;
  point: { x: number; y: number };
  polygon: number[][];
}

interface HitTestBatchRequest {
  type: 'hit-test-batch';
  point: { x: number; y: number };
  polygons: Array<{
    id: string;
    points: number[][];
    bbox: { minX: number; minY: number; maxX: number; maxY: number };
  }>;
}

interface HitTestResponse {
  type: 'hit-result';
  id: string;
  hit: boolean;
}

interface HitTestBatchResponse {
  type: 'hit-batch-result';
  hits: string[];
}

// Ray casting algorithm for point-in-polygon test
function pointInPolygon(px: number, py: number, polygon: number[][]): boolean {
  let inside = false;
  const n = polygon.length;
  
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    
    if (intersect) inside = !inside;
  }
  
  return inside;
}

// Winding number algorithm (alternative, more robust for complex polygons)
function windingNumber(px: number, py: number, polygon: number[][]): boolean {
  let wn = 0;
  const n = polygon.length;
  
  for (let i = 0, j = n - 1; i < n; j = i++) {
    if (polygon[j][1] <= py) {
      if (polygon[i][1] > py) {
        if (isLeft(polygon[j], polygon[i], [px, py]) > 0) {
          wn++;
        }
      }
    } else {
      if (polygon[i][1] <= py) {
        if (isLeft(polygon[j], polygon[i], [px, py]) < 0) {
          wn--;
        }
      }
    }
  }
  
  return wn !== 0;
}

// Helper for winding number algorithm
function isLeft(p0: number[], p1: number[], p2: number[]): number {
  return ((p1[0] - p0[0]) * (p2[1] - p0[1]) - 
          (p2[0] - p0[0]) * (p1[1] - p0[1]));
}

// Fast bbox check before expensive PIP test
function pointInBBox(
  px: number, 
  py: number, 
  bbox: { minX: number; minY: number; maxX: number; maxY: number }
): boolean {
  return px >= bbox.minX && px <= bbox.maxX && 
         py >= bbox.minY && py <= bbox.maxY;
}

// Process batch hit test for multiple polygons
function processBatchHitTest(
  px: number, 
  py: number, 
  polygons: Array<{
    id: string;
    points: number[][];
    bbox: { minX: number; minY: number; maxX: number; maxY: number };
  }>
): string[] {
  const hits: string[] = [];
  
  for (const poly of polygons) {
    // Quick bbox check first
    if (!pointInBBox(px, py, poly.bbox)) continue;
    
    // Detailed polygon check
    if (pointInPolygon(px, py, poly.points)) {
      hits.push(poly.id);
    }
  }
  
  return hits;
}

// Message handler
self.onmessage = (e: MessageEvent) => {
  const message = e.data;
  
  switch (message.type) {
    case 'hit-test': {
      const req = message as HitTestRequest;
      const hit = pointInPolygon(req.point.x, req.point.y, req.polygon);
      
      self.postMessage({
        type: 'hit-result',
        id: req.id,
        hit
      } as HitTestResponse);
      break;
    }
    
    case 'hit-test-batch': {
      const req = message as HitTestBatchRequest;
      const hits = processBatchHitTest(req.point.x, req.point.y, req.polygons);
      
      self.postMessage({
        type: 'hit-batch-result',
        hits
      } as HitTestBatchResponse);
      break;
    }
    
    default:
      console.warn('Unknown message type:', message.type);
  }
};

// Export for TypeScript (won't actually export in worker context)
export {};