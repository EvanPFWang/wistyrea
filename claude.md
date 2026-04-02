# Crown Mural Interactive - Project Rules

## Tech Stack & Dependencies
* **Frontend**: React 18+, Vite, TypeScript, WebGPU, Tailwind CSS, Framer Motion, Zustand.
* **Backend/Edge**: Cloudflare Workers + Durable Objects (SQLite-backed).
* **Processing**: Python 3.10+ (OpenCV, NumPy, Scikit-Learn).

## Library Version Standards
* **Python**: `opencv-python-headless==4.8.1.78`, `numpy==1.24.3`, `scikit-learn==1.3.0`.
* **Node**: `>=18.0.0`.

## Architecture Rules
* **Memory Limit**: Strictly adhere to Cloudflare's 128MB limit. No `OffscreenCanvas` or `new Image()` for mask processing in the worker; keep it buffer-based.
* **WebGPU Alignment**: WGSL structs must be 16-byte aligned. Pad `RegionMetadata` to match JS `Uint32Array` buffers.
* **Performance**: Use `id_map.png` on a hidden canvas for O(1) hover detection. Avoid rendering hundreds of SVG paths.
* **RLE Encoding**: All `.bin` masks must be cropped to their `bbox` in `regions.py` to minimize data transfer.

## Deployment & Scripts
* `npm run process:image`: Runs `regions.py` to generate `.bin` and `metadata.json`.
* `npm run dev`: Starts Vite dev server.