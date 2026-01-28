# Drop-in File Changes

## Quick Reference

### 🔴 DELETE These Files
```
src/CrownMuralController.ts  (replaced by React components + hooks)
src/style.css                (replaced by Tailwind CSS)
src/main.ts                  (replaced by main.tsx)
```

### 🟡 REPLACE These Files (Same Filename)
Copy the new versions to replace existing:

1. **index.html**
   - FROM: Manual canvas + tooltip + stats HTML
   - TO: Single `<div id="root"></div>` + React script

2. **package.json**
   - Added: React, shadcn/ui dependencies
   - Added: Vite React plugin, Tailwind

3. **tsconfig.json**
   - Added: `"jsx": "react-jsx"`
   - Added: Path aliases `"@/*": ["./src/*"]`
   - Added: React type support

4. **vite.config.ts**
   - Added: `@vitejs/plugin-react`
   - Added: Path alias resolution
   - Added: React-specific chunks

### 🟢 ADD These New Files

#### Configuration
```
tailwind.config.js       (Tailwind + shadcn/ui theme)
postcss.config.js        (PostCSS with Tailwind)
tsconfig.node.json       (Vite config TypeScript)
```

#### Source Files
```
src/main.tsx                              (React entry point)
src/App.tsx                               (Main app component)
src/index.css                             (Tailwind + CSS variables)

src/components/MuralCanvas.tsx            (Canvas rendering)
src/components/ProjectPreviewCard.tsx     (Hover card)
src/components/ProjectDemoDialog.tsx      (Demo modal)

src/components/ui/dialog.tsx              (shadcn Dialog)
src/components/ui/card.tsx                (shadcn Card)
src/components/ui/badge.tsx               (shadcn Badge)

src/hooks/useMuralController.ts           (Mural state hook)

src/lib/rle-decoder.ts                    (WebGPU decoder)
src/lib/utils.ts                          (Utility functions)
```

## Detailed Mapping

### CrownMuralController.ts → Multiple Files

| Old Code (Lines) | New Location | Purpose |
|------------------|--------------|---------|
| 23-57 | `src/lib/rle-decoder.ts` | WebGPU shader |
| 194-217 | `src/lib/rle-decoder.ts` | GPU initialization |
| 342-446 | `src/lib/rle-decoder.ts` | Mask decoding |
| 59-63 | `src/hooks/useMuralController.ts` | Fetch JSON |
| 65-76 | `src/hooks/useMuralController.ts` | Palette normalization |
| 327-340 | `src/hooks/useMuralController.ts` | Read ID at position |
| 219-320 | `src/hooks/useMuralController.ts` | Initialization |
| 487-532 | `src/App.tsx` + `src/components/MuralCanvas.tsx` | Event handling |
| 448-485 | `src/components/MuralCanvas.tsx` | Overlay rendering |
| ❌ 534-547 | (Removed) | Animation loop - React handles this |
| ❌ 178-190 | (Removed) | DOM queries - React refs handle this |
| ❌ 512-524 | (Removed) | Tooltip DOM manipulation - shadcn/ui handles this |

### index.html → Minimal Template

**Old** (60+ lines with inline styles):
```html
<style>/* Lots of CSS */</style>
<div id="container">
  <canvas id="mural-canvas"></canvas>
  <canvas id="id-canvas"></canvas>
</div>
<div id="tooltip">...</div>
<div id="stats">...</div>
```

**New** (7 lines):
```html
<!DOCTYPE html>
<html lang="en">
<head>...</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

All styling moved to Tailwind CSS in `src/index.css`.

## Installation Commands

```bash
# 1. Install new dependencies
npm install

# 2. If you get peer dependency warnings, use:
npm install --legacy-peer-deps

# 3. Verify installation
npm run dev
```

## Verification Steps

After copying all files:

1. ✅ **Check file structure**:
   ```bash
   ls src/components/ui/    # Should show: badge.tsx, card.tsx, dialog.tsx
   ls src/hooks/            # Should show: useMuralController.ts
   ls src/lib/              # Should show: rle-decoder.ts, utils.ts
   ```

2. ✅ **Build should succeed**:
   ```bash
   npm run build
   # Should output: dist/ folder with no errors
   ```

3. ✅ **Dev server should start**:
   ```bash
   npm run dev
   # Should open on http://localhost:3000
   ```

4. ✅ **Features should work**:
   - Mural image loads
   - Hover highlights regions
   - Click opens dialog
   - Preview card shows on hover

## Common Issues

### Import errors for `@/...`
**Problem**: `Cannot find module '@/lib/utils'`
**Fix**: Check `tsconfig.json` has:
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

And `vite.config.ts` has:
```typescript
resolve: {
  alias: {
    '@': resolve(__dirname, './src'),
  },
}
```

### CSS not loading
**Problem**: Styles don't apply
**Fix**: Ensure `src/main.tsx` imports CSS first:
```typescript
import './index.css';  // MUST be first
import { App } from './App';
```

### WebGPU errors
**Problem**: GPU initialization fails
**Fix**: Same as before - check browser support. The RLE decoder is identical to the old controller.

## What You DON'T Need to Change

✅ **Python scripts** - No changes  
✅ **Generated data** (`public/data/*`) - No changes  
✅ **Image processing** - No changes  
✅ **types.ts** - Optional enhancements only  
✅ **Cloudflare deployment** - wrangler.toml works as-is
