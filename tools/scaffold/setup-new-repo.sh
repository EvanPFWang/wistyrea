#!/bin/bash

# setup-new-repo.sh - Complete setup for new crown-mural project
# Run this in your new empty directory

echo "🎨 Setting up Crown Mural Interactive Project"
echo "=============================================="

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src scripts public/data

# Create package.json
echo "📦 Creating package.json..."
cat > package.json << 'EOF'
{
  "name": "crown-mural-interactive",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "npm run process:image && vite build",
    "build:vite-only": "vite build",
    "preview": "vite preview",
    "process:image": "python3 scripts/regions.py Mural_Crown_of_Italian_City.svg.png --blur 1 --close 1 --min_area 20 --canny_sigma 0.25 --palette_mode kbatch --sample_step 7 --output_dir public/data",
    "deploy": "npm run build"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "typescript": "^5.3.0",
    "vite": "^5.4.11"
  },
  "engines": {
    "node": ">=18.0.0"
  },
  "overrides": {
    "esbuild": "^0.24.3"
  }
}
EOF

# Create tsconfig.json
echo "⚙️ Creating tsconfig.json..."
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
EOF

# Create vite.config.ts
echo "⚡ Creating vite.config.ts..."
cat > vite.config.ts << 'EOF'
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    target: 'es2022',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log']
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'mural-controller': ['./src/CrownMuralController.ts']
        }
      }
    }
  },
  server: {
    port: 3000
  }
});
EOF

# Create wrangler.toml
echo "☁️ Creating wrangler.toml..."
cat > wrangler.toml << 'EOF'
name = "crown-mural"
compatibility_date = "2024-01-01"

[site]
bucket = "./dist"
EOF

# Create runtime.txt
echo "🐍 Creating runtime.txt..."
cat > runtime.txt << 'EOF'
python-3.9
EOF

# Create requirements.txt
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
opencv-python-headless==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.0
EOF

# Create .gitignore
echo "🚫 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Dependencies
node_modules/
__pycache__/
*.pyc

# Build outputs
dist/
.wrangler/

# Generated files (regenerated on build)
public/data/
public/coloured_regions.png

# Environment
.env
.env.local

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
npm-debug.log*
EOF

# Create index.html
echo "📄 Creating index.html..."
cat > index.html << 'EOFHTML'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crown Mural - Interactive</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0a0a0a;
      color: #fff;
      overflow: hidden;
    }
    
    #container {
      width: 100vw;
      height: 100vh;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    #mural-canvas {
      max-width: 100%;
      max-height: 100%;
      cursor: crosshair;
      image-rendering: crisp-edges;
    }
    
    #id-canvas {
      display: none;
    }
    
    #loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 18px;
      color: gold;
      animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    #tooltip {
      position: fixed;
      pointer-events: none;
      background: rgba(10, 10, 10, 0.95);
      border: 1px solid rgba(255, 215, 0, 0.3);
      border-radius: 6px;
      padding: 12px 16px;
      font-size: 14px;
      transform: translate(8px, -8px);
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      display: none;
      z-index: 1000;
      max-width: 280px;
      will-change: transform;
    }
    
    #tooltip strong {
      color: gold;
      display: block;
      margin-bottom: 4px;
      font-size: 15px;
    }
    
    #tooltip .blurb {
      color: #999;
      line-height: 1.4;
    }
    
    #stats {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      font-family: 'SF Mono', Monaco, monospace;
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="loading">Loading crown mural regions...</div>
    
    <canvas id="mural-canvas"></canvas>
    <canvas id="id-canvas"></canvas>
  </div>
  
  <div id="tooltip">
    <strong></strong>
    <div class="blurb"></div>
  </div>
  
  <div id="stats">
    <div>Regions: <span id="region-count">0</span></div>
    <div>Current: <span id="current-region">-</span></div>
    <div>FPS: <span id="fps">0</span></div>
  </div>

  <script type="module" src="/src/main.ts"></script>
</body>
</html>
EOFHTML

# Create src/style.css
echo "🎨 Creating src/style.css..."
cat > src/style.css << 'EOF'
/* All styles are currently inline in index.html */
/* This file exists to satisfy the import in main.ts */
EOF

# Create src/main.ts
echo "📝 Creating src/main.ts..."
cat > src/main.ts << 'EOF'
// src/main.ts
import './style.css';
import { CrownMuralController } from './CrownMuralController';

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new CrownMuralController();
  });
} else {
  new CrownMuralController();
}
EOF

echo ""
echo "✅ Basic structure created!"
echo ""
echo "⚠️ IMPORTANT: You still need to:"
echo "1. Copy src/types.ts from the previous artifact"
echo "2. Copy src/CrownMuralController.ts from the previous artifact"
echo "3. Copy your Mural_Crown_of_Italian_City.svg.png to this directory"
echo "4. Copy your regions.py to scripts/ and modify it for --output_dir support"
echo ""
echo "Then run:"
echo "  npm install"
echo "  npm run build"
echo "  npm run preview"