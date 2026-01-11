// src/CrownMuralController.ts
import {RGB, Hex, Palette, Metadata, Region, ProjectType, ControllerConfig} from './types.ts';

const RAW_BASE = import.meta.env.BASE_URL ?? '/';
const BASE     = RAW_BASE.endsWith('/') ? RAW_BASE : RAW_BASE + '/';
const ABS_BASE = new URL(BASE, document.baseURI);

const ABSOLUTE_RE = /^[a-zA-Z][\w+.-]*:|^\/\//;
export const absUrl = (p: string) => {
  if (ABSOLUTE_RE.test(p)) return p;//already abs/proto-relative
  const clean = p.replace(/^\/+/, '').replace(/\\/g, '/');
  return new URL(clean, ABS_BASE).href}//fully qualified URL

export const url = absUrl;

/**WGSL Compute Shader for RLE Decompression
 * mirrors logic in MuralStateObject for client-side execution
 * 1    reads 'Start Value' (0 or 1) from first integer
 * 2    iterates through run-lengths
 * 3    writes corresponding pixel values to output buffer
 * for max 2026 performance, single-thread loop per region
 */
const RLE_DECODE_SHADER = `
@group(0) @binding(0) var<storage, read> rle_input : array<u32>;
@group(0) @binding(1) var<storage, read_write> pixel_output : array<u32>;

struct Params {
  output_size: u32,
};
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  //process single RLE stream
  var rle_index: u32 = 1u; //skip header (index 0 is start_val)
  var pixel_index: u32 = 0u;
  var current_val: u32 = rle_input[0]; //load start value (0 or 1)
  
  let total_ints = arrayLength(&rle_input);
  
  //loop through run-length integers
  for (var i = 1u; i < total_ints; i = i + 1u) {
    let run_length = rle_input[i];
    
    //write 'run_length' pixels of 'current_val'
    for (var j = 0u; j < run_length; j = j + 1u) {
      if (pixel_index < params.output_size) {
        pixel_output[pixel_index] = current_val;
        pixel_index = pixel_index + 1u;
      }
    }
    
    //flip value for next run (0 -> 1, 1 -> 0)
    current_val = 1u - current_val;
  }
}
`;

async function fetchJSON<T>(path: string): Promise<T | null> {
  const res = await fetch(url(path), { cache: 'no-cache' });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

function normalizePaletteToRGB(raw: unknown): Palette {
  const any = raw as any;
  if (any && typeof any === 'object' && 'map' in any) {
    const bg = Number(any.background_id ?? 0);
    const m = any.map as Record<string, unknown>;
    const map = Object.fromEntries(
      Object.entries(m).map(([k, v]) => [k, (v as RGB) ?? { r: 0, g: 0, b: 0 }])
    ) as Record<string, RGB>;
    return { background_id: bg, map };
  }
  return { background_id: 0, map: {} };
}

type ResolvedConfig = Required<Pick<ControllerConfig, 'dataPath'|'enableDebug'|'maxFPS'>> & {
  metadataPath: string;
  idMapPath: string;
  storagePath: string;
  baseImagePath: string;
  palettePath: string
};

const DEFAULTS: ResolvedConfig = {
  dataPath: 'data',
  enableDebug: false,
  maxFPS: 60,
  metadataPath: 'data/metadata.json',
  idMapPath: 'data/id_map.png',
  storagePath: 'data/shape_masks',
  baseImagePath: 'Mural_Crown_of_Italian_City.svg.png',
  palettePath: 'data/palette.json',
};

export class CrownMuralController {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private idCanvas: HTMLCanvasElement;
  private idCtx: CanvasRenderingContext2D;
  private overlay: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;

  private tooltip: HTMLDivElement;
  private tooltipTitle: HTMLElement;
  private tooltipBlurb: HTMLElement;

  private config: ResolvedConfig;
  private regions: Map<number, Region> = new Map();
  private hoveredRegion: number | null = null;
  private prevId: number = -1;
  private latestEvent: PointerEvent | null = null;
  private needsRedraw: boolean = false;

  //data
  private metadata: Metadata | null = null;
  private palette: Palette | null = null;
  private baseImage: HTMLImageElement | null = null;
  private idImageData: ImageData | null = null;
  private metaDir: string = '';
  private resolveFromMetaDir(p: string): string {
    const clean = p.replace(/\\/g, '/').replace(/^\/+/, '');
    if (ABSOLUTE_RE.test(clean)) return clean;
    return `${this.metaDir}/${clean}`;}



  //webgpu resources
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private shaderModule: GPUShaderModule | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private initGPUPromise: Promise<void> | null = null;

  private maskCache = new Map<number, ImageData>(); //cache decoded masks
  private regionByShapeIndex = new Map<number, Region>();

  private animationId: number = 0;
  private fpsElement = document.getElementById('fps');
  private fpsLastUpdate = 0;
  private lastTimestamp: number = 0;
  private overlayDirty: { x: number; y: number; width: number; height: number } | null = null;

  constructor(config: ControllerConfig={}) {
    const dp = config.dataPath ?? DEFAULTS.dataPath;
    this.config = {
      ...DEFAULTS,
      ...config,
      metadataPath: config.metadataPath ?? `${dp}/metadata.json`,
      idMapPath: config.idMapPath ?? `${dp}/id_map.png`,
      storagePath: config.storagePath ?? `${dp}/shape_masks`,
      palettePath: config.palettePath ?? `${dp}/palette.json`,
    };

    this.canvas = document.getElementById('mural-canvas') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d', { alpha: false, desynchronized: true })!;

    this.overlay = document.createElement('canvas');
    this.overlayCtx = this.overlay.getContext('2d', { alpha: true })!;

    this.idCanvas = document.getElementById('id-canvas') as HTMLCanvasElement;
    this.idCtx = this.idCanvas.getContext('2d', { willReadFrequently: true, alpha: false })!;

    this.tooltip = document.getElementById('tooltip') as HTMLDivElement;
    this.tooltipTitle = this.tooltip.querySelector('strong')!;
    this.tooltipBlurb = this.tooltip.querySelector('.blurb')!;

    this.init();
  }

  private async ensureGPUReady(): Promise<GPUDevice> {
    if (this.device) return this.device;

    this.initGPUPromise ??= (async () => {
      if (!navigator.gpu) {
        console.warn("WebGPU not supported, falling back to CPU decode if implemented");
        throw new Error("WebGPU not supported");
      }
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) throw new Error("No GPU adapter found");

      this.device = await this.adapter.requestDevice();

      //build pipeline once
      this.shaderModule = this.device.createShaderModule({ code: RLE_DECODE_SHADER });
      this.computePipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.shaderModule, entryPoint: "main" },
      });
    })();

    await this.initGPUPromise;
    return this.device!;
  }

  private async init(): Promise<void> {
    try {
      //initialize gpu early
      this.ensureGPUReady().catch(e => console.log("GPU init deferred/failed", e));

      const [metadata, rawPalette] = await Promise.all([
        fetchJSON<Metadata>(this.config.metadataPath),
        fetchJSON<unknown>(this.config.palettePath),
      ]);

      this.metaDir = this.config.metadataPath.replace(/\\/g, '/').split('/').slice(0, -1).join('/');
      if (!this.metaDir) this.metaDir = this.config.dataPath;


      if (!metadata) throw new Error('metadata.json missing');
      this.metadata = metadata;

      const {width, height} = this.metadata.dimensions;
      this.resizeCanvases(width, height);

      //populate regions map
      for (const region of metadata.regions) {
        this.regions.set(region.id, region);
        //parse ID from mask path in metadata if needed, or trust metadata ID
        //assuming 1:1 mapping for simplicity in this demo
        this.regionByShapeIndex.set(region.id, region);
      }

      this.palette = rawPalette ? normalizePaletteToRGB(rawPalette) : {background_id: 0, map: {}};

      this.generateProjectData();

      await Promise.all([
        this.loadBaseImage(),
        this.loadIdMap()
      ]);

      this.setupEventHandlers();
      document.getElementById('loading')?.style.setProperty('display', 'none');
      this.animate(0);
    } catch (err) {
      console.error('init Failed: ', err);
      const loading = document.getElementById('loading');
      if (loading) loading.textContent = 'Failed to load mural data';
    }
  }

  private resizeCanvases(w: number, h: number) {
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    //resize main canvas
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    //resize id canvas (1:1 pixel match required)
    this.idCanvas.width = w;
    this.idCanvas.height = h;

    //resize overlay
    this.overlay.width = w;
    this.overlay.height = h;
  }

  private generateProjectData(): void {
    if (!this.metadata) return;
    const projectTypes: ProjectType[] = [
      { prefix: 'tower-', titles: ['Defence System', 'Watch Tower'] },
      { prefix: 'arch-', titles: ['Gateway', 'Portal'] }
    ];

    for (const region of this.metadata.regions) {
      const type = projectTypes[region.id % projectTypes.length];
      const titleIdx = Math.floor(region.id / projectTypes.length) % type.titles.length;
      region.project = {
        title: `${type.titles[titleIdx]} #${region.id}`,
        href: `/projects/${type.prefix}${region.id}`,
        blurb: `Crown element ${region.id}`
      };
    }
    const rc = document.getElementById('region-count');
    if (rc) rc.textContent = String(this.regions.size);
  }

  private async loadBaseImage(): Promise<void> {
    const src = url(this.config.baseImagePath);
    const img = new Image();
    img.decoding = 'async';
    img.crossOrigin = 'anonymous';
    img.src = src;
    await img.decode();
    this.baseImage = img;
    this.needsRedraw = true;
  }

  private async loadIdMap(): Promise<void> {
    const src = url(this.config.idMapPath);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = src;
    await img.decode();

    this.idCtx.drawImage(img, 0, 0);
    this.idImageData = this.idCtx.getImageData(0, 0, this.idCanvas.width, this.idCanvas.height);
  }

  private readIdAt(clientX: number, clientY: number): number {
    if (!this.idImageData) return 0;
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left) * (this.canvas.width / rect.width));
    const y = Math.floor((clientY - rect.top) * (this.canvas.height / rect.height));

    if (x < 0 || x >= this.idCanvas.width || y < 0 || y >= this.idCanvas.height) return 0;

    const idx = (y * this.idImageData.width + x) * 4;
    const data = this.idImageData.data;
    //id encoded as R + (G<<8) + (B<<16)
    return data[idx] | (data[idx+1] << 8) | (data[idx+2] << 16);
  }

  private async fetchAndDecodeMask(regionId: number): Promise<ImageData | null> {
    if (this.maskCache.has(regionId)) return this.maskCache.get(regionId)!;

    const region = this.regions.get(regionId);
    if (!region?.mask) return null;

    //fetch binary RLE
    const res = await fetch(url(region.mask));
    if (!res.ok) return null;
    const rleBuffer = await res.arrayBuffer();

    let device: GPUDevice;
    try {
      device = await this.ensureGPUReady();
    } catch {
      return null; //fallback or fail gracefully
    }

    const outputSize = this.idCanvas.width * this.idCanvas.height;
    const outputByteSize = outputSize * 4;

    //pad input to 4 bytes
    const rleBytes = new Uint8Array(rleBuffer);
    const paddedSize = (rleBytes.byteLength + 3) & ~3;
    const rlePadded = paddedSize === rleBytes.byteLength ? rleBytes : (() => {
       const p = new Uint8Array(paddedSize);
       p.set(rleBytes);
       return p;
    })();

    //gpu buffer allocation
    const inputBuf = device.createBuffer({
      size: rlePadded.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint8Array(inputBuf.getMappedRange()).set(rlePadded);
    inputBuf.unmap();

    const outputBuf = device.createBuffer({
      size: outputByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const paramBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramBuf, 0, new Uint32Array([outputSize]));

    const stagingBuf = device.createBuffer({
      size: outputByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    //dispatch compute
    const bindGroup = device.createBindGroup({
      layout: this.computePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();

    encoder.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, outputByteSize);
    device.queue.submit([encoder.finish()]);

    //read back results
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const rawData = new Uint32Array(stagingBuf.getMappedRange());

    //convert 0/1 u32 mask to RGBA ImageData
    //optimization: direct write to clamped array
    const imgData = new ImageData(this.idCanvas.width, this.idCanvas.height);
    const px = imgData.data;
    for (let i = 0; i < outputSize; i++) {
        const val = rawData[i];
        if (val > 0) {
            const idx = i * 4;
            px[idx] = 255;   //R
            px[idx+1] = 255; //G
            px[idx+2] = 255; //B
            px[idx+3] = 255; //Alpha
        }
    }

    stagingBuf.unmap();
    inputBuf.destroy();
    outputBuf.destroy();
    paramBuf.destroy();
    stagingBuf.destroy();

    this.maskCache.set(regionId, imgData);
    return imgData;
  }

  private async drawComposite(regionId: number) {
    this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (regionId <= 0) {
      this.overlayDirty = null;
      return;
    }

    const maskData = await this.fetchAndDecodeMask(regionId);
    if (!maskData) return;

    //draw mask to temp canvas to composite
    //in a full app, you would keep mask in a GPUTexture and render via pipeline
    //but for hybrid, we putImageData
    const tmp = document.createElement('canvas');
    tmp.width = this.overlay.width;
    tmp.height = this.overlay.height;
    tmp.getContext('2d')!.putImageData(maskData, 0, 0);

    const ctx = this.overlayCtx;
    const color = this.palette?.map[String(regionId)] ?? {r:255, g:215, b:0};

    //draw solid color
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    ctx.fillRect(0, 0, this.overlay.width, this.overlay.height);

    //mask it
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(tmp, 0, 0);

    ctx.globalCompositeOperation = 'source-over';

    //set dirty rect for efficient redraw
    const r = this.regions.get(regionId);
    if (r) {
       this.overlayDirty = r.bbox;
    }
  }

  private setupEventHandlers(): void {
    const handlePointerMove = (e: PointerEvent) => {
      this.latestEvent = e;
      if (!this.animationId) return; //guard
      requestAnimationFrame(() => this.processPointerEvents());
    };
    this.canvas.addEventListener('pointermove', handlePointerMove, { passive: true });
    this.canvas.addEventListener('pointerleave', () => this.clearHover());
  }

  private processPointerEvents(): void {
    if (!this.latestEvent) return;
    const id = this.readIdAt(this.latestEvent.clientX, this.latestEvent.clientY);
    if (id !== this.prevId) {
      this.prevId = id;
      this.onRegionChange(id, this.latestEvent.clientX, this.latestEvent.clientY);
    }
  }

  private async onRegionChange(id: number, x: number, y: number) {
    this.hoveredRegion = id > 0 ? id : null;
    this.updateTooltip(id);
    await this.drawComposite(id);
    this.needsRedraw = true;
  }

  private updateTooltip(id: number) {
    const r = this.regions.get(id);
    if (r?.project) {
        this.tooltipTitle.textContent = r.project.title;
        this.tooltipBlurb.textContent = r.project.blurb;
        this.tooltip.style.display = 'block';
        if (this.latestEvent) {
            this.tooltip.style.left = this.latestEvent.clientX + 10 + 'px';
            this.tooltip.style.top = this.latestEvent.clientY + 10 + 'px';
        }
    } else {
        this.tooltip.style.display = 'none';
    }
  }

  private clearHover() {
    this.hoveredRegion = null;
    this.prevId = -1;
    this.tooltip.style.display = 'none';
    this.drawComposite(0);
    this.needsRedraw = true;
  }

  private animate = (ts: number) => {
    this.animationId = requestAnimationFrame(this.animate);

    //fps counter logic omitted for brevity

    if (this.needsRedraw) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.baseImage) {
            this.ctx.drawImage(this.baseImage, 0, 0);
        }
        this.ctx.drawImage(this.overlay, 0, 0);
        this.needsRedraw = false;
    }
  };
}