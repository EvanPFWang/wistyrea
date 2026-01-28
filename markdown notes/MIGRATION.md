# React + shadcn/ui Migration Guide

## Overview
This migration transforms your vanilla TypeScript mural controller into a modern React application with shadcn/ui components.

## What Changed

### Architecture
- **Before**: Vanilla TypeScript with manual DOM manipulation
- **After**: React components with declarative state management

### Key Improvements
1. **Component-Based**: Modular, reusable UI components
2. **State Management**: React hooks instead of class properties
3. **Declarative Rendering**: React handles DOM updates
4. **Type Safety**: Full TypeScript support maintained
5. **Modern UI**: shadcn/ui for professional, accessible components

## File Changes

### ✅ KEEP (No changes needed)
- `regions.py` - Image processing script
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime
- `wrangler.toml` - Cloudflare config (if deploying to Workers)
- `Mural_Crown_of_Italian_City.svg.png` - Source image

### 🔄 REPLACE
- `index.html` → New React-based version (minimal, just div#root)
- `src/main.ts` → `src/main.tsx` (React entry point)
- `package.json` → Updated with React dependencies
- `tsconfig.json` → Updated with React/JSX support
- `vite.config.ts` → Updated with React plugin

### 🆕 NEW FILES
- `src/App.tsx` - Main application component
- `src/components/MuralCanvas.tsx` - Canvas rendering component
- `src/components/ProjectPreviewCard.tsx` - Hover preview card
- `src/components/ProjectDemoDialog.tsx` - Full demo dialog
- `src/components/ui/*.tsx` - shadcn/ui components
- `src/hooks/useMuralController.ts` - Custom hook for mural logic
- `src/lib/rle-decoder.ts` - WebGPU RLE decoder utility
- `src/lib/utils.ts` - Utility functions
- `src/index.css` - Tailwind CSS + shadcn/ui variables
- `tailwind.config.js` - Tailwind configuration
- `postcss.config.js` - PostCSS configuration
- `tsconfig.node.json` - TypeScript config for Vite

### ❌ DELETE (No longer needed)
- `src/CrownMuralController.ts` - Replaced by React components + hooks
- `src/style.css` - Replaced by Tailwind CSS

## What Was Removed from CrownMuralController.ts

### Unnecessary Code (Now Handled by React)

1. **Manual DOM Manipulation** (Lines 178-190, 512-524)
   ```typescript
   // OLD: Manual DOM queries and updates
   this.canvas = document.getElementById('mural-canvas')
   this.tooltip = document.getElementById('tooltip')
   this.tooltipTitle.textContent = r.project.title
   
   // NEW: React state and JSX
   {hoveredRegion?.project && <ProjectPreviewCard region={hoveredRegion} />}
   ```

2. **Manual Event Handlers** (Lines 487-495)
   ```typescript
   // OLD: addEventListener with callbacks
   this.canvas.addEventListener('pointermove', handlePointerMove)
   
   // NEW: React event props
   <canvas onPointerMove={handlePointerMove} />
   ```

3. **Animation Loop** (Lines 534-547)
   ```typescript
   // OLD: requestAnimationFrame loop
   private animate = (ts: number) => {
     this.animationId = requestAnimationFrame(this.animate);
     // ...manual redraw logic
   }
   
   // NEW: React useEffect handles rendering
   useEffect(() => {
     // Declarative rendering
   }, [hoveredRegion])
   ```

4. **FPS Counter** (Lines 162-164, 537)
   ```typescript
   // OLD: Manual FPS calculation
   private fpsElement = document.getElementById('fps');
   
   // NEW: Can add with React if needed, or remove entirely
   ```

5. **Tooltip Positioning** (Lines 518-521)
   ```typescript
   // OLD: Manual style.left/top updates
   this.tooltip.style.left = this.latestEvent.clientX + 10 + 'px'
   
   // NEW: React state + CSS positioning
   <Card style={{ left: `${position.x}px` }} />
   ```

6. **needsRedraw Flag** (Lines 136, 509, 531, 539-545)
   ```typescript
   // OLD: Manual redraw tracking
   private needsRedraw: boolean = false;
   
   // NEW: React automatically re-renders on state change
   ```

### Code That Was Preserved

1. **WebGPU RLE Decoder** → Moved to `src/lib/rle-decoder.ts`
   - Lines 23-57: Shader code
   - Lines 194-217: GPU initialization
   - Lines 342-446: Mask decoding logic

2. **Region Detection** → Moved to `useMuralController` hook
   - Lines 327-340: `readIdAt` function
   - Lines 139-148: Metadata/region mapping

3. **ID Map Loading** → Moved to `useMuralController` hook
   - Lines 313-325: ID map image loading

4. **Palette Handling** → Moved to `useMuralController` hook
   - Lines 65-76: Palette normalization

## Enhanced Features

### New Capabilities
1. **Hover Previews**: shadcn/ui Card with keywords and description
2. **Full Demo Dialog**: Modal with video preview and live iframe
3. **Keyboard Accessible**: Full screen reader support via Radix UI
4. **Dark Mode Ready**: CSS variables make theming easy
5. **Responsive**: Tailwind utilities for any screen size

### Extended Types (Add to types.ts)
```typescript
export interface ProjectInfo {
  title: string;
  href: string;
  blurb: string;
  keywords?: string[];      // NEW
  videoUrl?: string;        // NEW
  demoUrl?: string;         // NEW
  thumbnailUrl?: string;    // NEW
}
```

## Installation Steps

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Process your image** (if not already done):
   ```bash
   npm run process:image
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## Deployment

### Cloudflare Pages
The existing `wrangler.toml` is configured for Cloudflare Pages. The Durable Object setup in `MuralStateObject.ts` can be used if you need server-side RLE decoding, but with WebGPU in the browser, it's optional.

To deploy:
```bash
npm run deploy
```

## Migration Checklist

- [ ] Install new dependencies (`npm install`)
- [ ] Copy all new files to your project
- [ ] Replace old files with updated versions
- [ ] Delete `src/CrownMuralController.ts`
- [ ] Delete old `src/style.css`
- [ ] Update `types.ts` with new fields (keywords, videoUrl, etc.)
- [ ] Test locally (`npm run dev`)
- [ ] Update region data with project information
- [ ] Build and deploy (`npm run build`)

## Performance Notes

### What's Faster
- **Initial Load**: React lazy loading + code splitting
- **Re-renders**: React's virtual DOM only updates changed elements
- **Event Handling**: React's synthetic events are optimized

### What's Similar
- **WebGPU Decoding**: Same performance (moved to separate module)
- **Canvas Rendering**: Same 2D context operations

### Best Practices
1. Masks are cached (same as before)
2. GPU initialization happens once
3. Overlay updates only on hover change (useEffect dependency)
4. Preview cards use CSS transforms (GPU accelerated)

## Troubleshooting

### "navigator.gpu is not defined"
- Ensure you're using a WebGPU-compatible browser (Chrome/Edge 113+)
- Check that your deployment environment supports WebGPU

### Path alias errors
- Ensure `tsconfig.json` includes the `paths` configuration
- Restart your dev server after config changes

### Tailwind styles not applying
- Check that `postcss.config.js` and `tailwind.config.js` exist
- Ensure `src/index.css` imports are at the top of `main.tsx`

## Next Steps

1. **Add Project Data**: Update your regions.py output or manually add `project` fields to metadata.json
2. **Customize Styling**: Modify CSS variables in `src/index.css`
3. **Add Features**: 
   - Video previews
   - Social sharing
   - Analytics
   - Search functionality
4. **Optimize**: Add lazy loading for demos, optimize images

## Support

The core mural detection logic is unchanged, so your existing data pipeline works as-is. The new architecture makes it easier to:
- Add new UI features
- Integrate with external APIs
- Build mobile experiences
- A/B test different layouts
