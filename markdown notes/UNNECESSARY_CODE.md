# What's Unnecessary from CrownMuralController.ts

This document maps every section of the old controller and marks what's still needed vs. what React handles automatically.

## ❌ COMPLETELY UNNECESSARY - React Handles This

### 1. DOM Element References (Lines 120-129)
```typescript
// ❌ DELETE - React refs handle this
private canvas: HTMLCanvasElement;
private ctx: CanvasRenderingContext2D;
private tooltip: HTMLDivElement;
private tooltipTitle: HTMLElement;
private tooltipBlurb: HTMLElement;

// ✅ REPLACE WITH React refs
const canvasRef = useRef<HTMLCanvasElement>(null);
```

**Why**: React manages DOM elements through refs, no need for manual `getElementById()`.

---

### 2. State Tracking Flags (Lines 133-137)
```typescript
// ❌ DELETE - React state handles this
private hoveredRegion: number | null = null;
private prevId: number = -1;
private latestEvent: PointerEvent | null = null;
private needsRedraw: boolean = false;

// ✅ REPLACE WITH React state
const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);
```

**Why**: React's state management automatically triggers re-renders.

---

### 3. Animation Loop (Lines 161-165, 534-547)
```typescript
// ❌ DELETE - React handles rendering
private animationId: number = 0;
private lastTimestamp: number = 0;

private animate = (ts: number) => {
  this.animationId = requestAnimationFrame(this.animate);
  
  if (this.needsRedraw) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.baseImage) {
      this.ctx.drawImage(this.baseImage, 0, 0);
    }
    this.ctx.drawImage(this.overlay, 0, 0);
    this.needsRedraw = false;
  }
};

// ✅ REPLACE WITH useEffect
useEffect(() => {
  // React re-renders when hoveredRegion changes
}, [hoveredRegion]);
```

**Why**: React's declarative rendering eliminates the need for manual RAF loops.

---

### 4. FPS Counter (Lines 162-164)
```typescript
// ❌ DELETE - Optional debugging feature
private fpsElement = document.getElementById('fps');
private fpsLastUpdate = 0;

// Only needed if you want FPS display
// Can re-implement as a separate React component if desired
```

**Why**: Not core functionality. Can add back as a custom hook if needed.

---

### 5. Manual Event Listeners (Lines 487-495)
```typescript
// ❌ DELETE - React synthetic events handle this
private setupEventHandlers(): void {
  const handlePointerMove = (e: PointerEvent) => {
    this.latestEvent = e;
    requestAnimationFrame(() => this.processPointerEvents());
  };
  this.canvas.addEventListener('pointermove', handlePointerMove);
  this.canvas.addEventListener('pointerleave', () => this.clearHover());
}

// ✅ REPLACE WITH React event props
<canvas onPointerMove={handlePointerMove} onPointerLeave={clearHover} />
```

**Why**: React's synthetic event system is more efficient.

---

### 6. Tooltip DOM Manipulation (Lines 512-524)
```typescript
// ❌ DELETE - shadcn/ui Card handles this
private updateTooltip(id: number) {
  const r = this.regions.get(id);
  if (r?.project) {
    this.tooltipTitle.textContent = r.project.title;
    this.tooltipBlurb.textContent = r.project.blurb;
    this.tooltip.style.display = 'block';
    this.tooltip.style.left = this.latestEvent.clientX + 10 + 'px';
    this.tooltip.style.top = this.latestEvent.clientY + 10 + 'px';
  } else {
    this.tooltip.style.display = 'none';
  }
}

// ✅ REPLACE WITH React component
{hoveredRegion?.project && (
  <ProjectPreviewCard region={hoveredRegion} position={mousePosition} />
)}
```

**Why**: shadcn/ui components handle styling and positioning declaratively.

---

### 7. Overlay Dirty Rect (Lines 165-166, 480-484)
```typescript
// ❌ DELETE - React handles partial updates
private overlayDirty: { x: number; y: number; width: number; height: number } | null = null;

this.overlayDirty = r.bbox; // Tracking dirty region

// React's virtual DOM already optimizes updates
```

**Why**: React's reconciliation algorithm handles efficient updates.

---

## ✅ KEEP - Core Logic Still Needed

### 1. WebGPU RLE Decoder (Lines 23-57, 194-217, 342-446)
```typescript
// ✅ MOVED to src/lib/rle-decoder.ts
const RLE_DECODE_SHADER = `...`;

class RLEDecoder {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  // ... GPU initialization and decoding logic
}
```

**Why**: Core algorithm that React can't replace. Moved to separate utility.

---

### 2. Region Detection (Lines 327-340)
```typescript
// ✅ MOVED to useMuralController hook
private readIdAt(clientX: number, clientY: number): number {
  if (!this.idImageData) return 0;
  const rect = this.canvas.getBoundingClientRect();
  const x = Math.floor((clientX - rect.left) * (this.idCanvas.width / rect.width));
  const y = Math.floor((clientY - rect.top) * (this.idCanvas.height / rect.height));
  // ... pixel reading logic
}
```

**Why**: Algorithm logic, just moved into React hook.

---

### 3. Data Loading (Lines 219-290)
```typescript
// ✅ MOVED to useMuralController hook (useEffect)
private async init(): Promise<void> {
  const [metadata, rawPalette] = await Promise.all([
    fetchJSON<Metadata>(this.config.metadataPath),
    fetchJSON<unknown>(this.config.palettePath),
  ]);
  // ... initialization logic
}
```

**Why**: Still need to load data, just in React's useEffect instead.

---

### 4. Mask Fetching & Decoding (Lines 342-446)
```typescript
// ✅ MOVED to useMuralController hook
private async fetchAndDecodeMask(regionId: number): Promise<ImageData | null> {
  if (this.maskCache.has(regionId)) return this.maskCache.get(regionId)!;
  // ... fetch and decode logic
}
```

**Why**: Core business logic, moved to hook with useRef for caching.

---

### 5. Palette & Color Handling (Lines 65-76, 467)
```typescript
// ✅ MOVED to useMuralController hook
function normalizePaletteToRGB(raw: unknown): Palette {
  // ... palette normalization
}

const getRegionColor = (regionId: number): RGB => {
  return palette?.map[String(regionId)] ?? { r: 255, g: 215, b: 0 };
};
```

**Why**: Data transformation logic, moved to hook utility function.

---

## Summary Table

| Feature | Old Location | New Location | Status |
|---------|-------------|--------------|--------|
| **DOM Queries** | Lines 178-190 | React refs | ❌ DELETE |
| **State Tracking** | Lines 133-137 | useState | ❌ DELETE |
| **Animation Loop** | Lines 534-547 | useEffect | ❌ DELETE |
| **Event Handlers** | Lines 487-495 | JSX props | ❌ DELETE |
| **Tooltip DOM** | Lines 512-524 | shadcn/ui | ❌ DELETE |
| **FPS Counter** | Lines 162-164 | (Optional) | ❌ DELETE |
| **Dirty Rect** | Lines 165-166 | React VDOM | ❌ DELETE |
| **WebGPU Shader** | Lines 23-57 | lib/rle-decoder.ts | ✅ KEEP |
| **GPU Init** | Lines 194-217 | lib/rle-decoder.ts | ✅ KEEP |
| **Mask Decode** | Lines 342-446 | lib/rle-decoder.ts | ✅ KEEP |
| **Region Detection** | Lines 327-340 | useMuralController | ✅ KEEP |
| **Data Loading** | Lines 219-290 | useMuralController | ✅ KEEP |
| **Palette Utils** | Lines 65-76 | useMuralController | ✅ KEEP |

## Lines Deleted vs. Moved

**Total Lines in CrownMuralController.ts**: ~548  
**Lines Completely Deleted**: ~180 (33%)  
**Lines Moved to React**: ~368 (67%)  

### Breakdown
- **Deleted** (~180 lines): DOM manipulation, manual state, RAF loop, event setup
- **Moved** (~368 lines): 
  - RLEDecoder class: ~150 lines → `lib/rle-decoder.ts`
  - Hook logic: ~150 lines → `useMuralController.ts`
  - Constants/utils: ~68 lines → distributed across files

## Key Insight

React eliminates **all imperative UI code** (DOM queries, manual state tracking, event listeners, animation loops). The only code that survives is:

1. **Algorithms** (region detection, RLE decode)
2. **Data loading** (fetch, parse)
3. **Business logic** (mask caching, color mapping)

Everything else React does better with less code. The ~180 deleted lines are replaced by ~50 lines of declarative JSX + hooks.
