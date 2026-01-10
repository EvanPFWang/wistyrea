import { DurableObject } from "cloudflare:workers";

/**WGSL Compute Shader for RLE Decompression
 * 1    reads the 'Start Value' (0 or 1) from first integer
 * 2    iterates through run-lengths
 * 3    writes corresponding pixel values to output buffer
 * for max 2026 performance, single-thread loop per region
 *
 * formassive images, a "Prefix Sum" parallel algorithm would be
 * used
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
  //in batched scenario, global_id.x could represent the Region ID.
  //here process  single RLE stream.
  
  var rle_index: u32 = 1u; //skip header (index 0 is start_val)
  var pixel_index: u32 = 0u;
  var current_val: u32 = rle_input[0]; //load start value (0 or 1)
  
  let total_ints = arrayLength(&rle_input);
  
  //lop through run-length integers
  for (var i = 1u; i < total_ints; i = i + 1u) {
    let run_length = rle_input[i];
    
    //wite 'run_length' pixels of 'current_val'
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

export class MuralStateObject extends DurableObject {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;


  private shaderModule: GPUShaderModule | null = null;
  private computePipeline: GPUComputePipeline | null = null;

  private initGPUPromise: Promise<void> | null = null;//race safe init

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    //state init if using SQLite happens here
  }


  private async ensureGPUReady(): Promise<GPUDevice> {
    if (this.device) return this.device;

    //single init runs even if requests overlap
    this.initGPUPromise ??= (async () => {
      if (!navigator.gpu) {
        throw new Error("WebGPU not supported in this environment");
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

async fetch(request: Request): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    let device: GPUDevice;
    try {
      device = await this.ensureGPUReady();
    } catch (err: any) {
      return new Response(String(err?.message ?? err), { status: 500 });
    }

    const rleBuffer = await request.arrayBuffer();
    const outputSize = parseInt(request.headers.get("X-Output-Size") || "0", 10);

    if (rleBuffer.byteLength === 0 || outputSize === 0) {
      return new Response("Invalid input data or Missing X-Output-Size", { status: 400 });
    }

    //pad to multiple of 4 bytes (required when mappedAtCreation=true)
    const rleBytes = new Uint8Array(rleBuffer);
    const paddedSize = (rleBytes.byteLength + 3) & ~3;
    const rlePadded =
      paddedSize === rleBytes.byteLength
        ? rleBytes
        : (() => {
            const p = new Uint8Array(paddedSize);
            p.set(rleBytes);
            return p;
          })();

    const outputByteSize = outputSize * 4;

    //alloc per-request resources (wrap in try/finally so always destroy)
    const inputGPUBuffer = device.createBuffer({
      size: rlePadded.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    const outputGPUBuffer = device.createBuffer({
      size: outputByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramBuffer = device.createBuffer({
      size: 16, //16-byte aligned safe for uniforms
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const stagingBuffer = device.createBuffer({
      size: outputByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    try {
      //write input + unmap
      new Uint8Array(inputGPUBuffer.getMappedRange()).set(rlePadded);
      inputGPUBuffer.unmap();

      device.queue.writeBuffer(paramBuffer, 0, new Uint32Array([outputSize]));

      const computePipeline = this.computePipeline!;
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputGPUBuffer } },
          { binding: 1, resource: { buffer: outputGPUBuffer } },
          { binding: 2, resource: { buffer: paramBuffer } },
        ],
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(1);
      passEncoder.end();

      commandEncoder.copyBufferToBuffer(outputGPUBuffer, 0, stagingBuffer, 0, outputByteSize);
      device.queue.submit([commandEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const copyArrayBuffer = stagingBuffer.getMappedRange().slice(0);
      stagingBuffer.unmap();

      return new Response(copyArrayBuffer, {
        headers: { "Content-Type": "application/octet-stream" },
      });
    } finally {
      //explicit cleanup
      inputGPUBuffer.destroy();
      outputGPUBuffer.destroy();
      paramBuffer.destroy();
      stagingBuffer.destroy();
    }
  }
}