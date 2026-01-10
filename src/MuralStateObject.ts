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

  private initGpuPromise: Promise<void> | null = null;//race safe init

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    //state init if using SQLite happens here
  }


  private async ensureGpuReady(): Promise<GPUDevice> {
    if (this.device) return this.device;

    //single init runs even if requests overlap
    this.initGpuPromise ??= (async () => {
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

    await this.initGpuPromise;
    return this.device!;
  }
  
  async fetch(request: Request): Promise<Response> {
    //init GPU (Lazy Load)
    if (!this.device) {
      if (!navigator.gpu) {
        return new Response("WebGPU not supported in this environment", { status: 500 });
      }
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        return new Response("No GPU adapter found", { status: 500 });
      }
      this.device = await this.adapter.requestDevice();
    }
    const device = this.device!;

    //parse Request (expects RLE binary in body, OutputSize in header)
    if (request.method === "POST") {
      const rleBuffer = await request.arrayBuffer();
      const outputSize = parseInt(request.headers.get("X-Output-Size") || "0");

      if (rleBuffer.byteLength === 0 || outputSize === 0) {
        return new Response("Invalid input data or Missing X-Output-Size", { status: 400 });
      }

      //GPU resource alloc
      //
      //input Buffer (RLE data)
      const inputGPUBuffer = device.createBuffer({
        size: rleBuffer.byteLength, //must be 4-byte aligned
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint8Array(inputGPUBuffer.getMappedRange()).set(new Uint8Array(rleBuffer));
      inputGPUBuffer.unHxmap();

      //output buffer(raw pixels) - 4 bytes per pixel (u32)
      const outputByteSize = outputSize * 4;
      const outputGPUBuffer = device.createBuffer({
        size: outputByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      //uniform buffer (params)
      const paramBufferSize = 16; //4 bytes needed, but 16 byte alignment safer
      const paramBuffer = device.createBuffer({
        size: paramBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramBuffer, 0, new Uint32Array([outputSize]));

      //pipeline setup
      const shaderModule = device.createShaderModule({ code: RLE_DECODE_SHADER });
      const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: "main" }
      });

      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputGPUBuffer } },
          { binding: 1, resource: { buffer: outputGPUBuffer } },
          { binding: 2, resource: { buffer: paramBuffer } }
        ]
      });

      //execution (Dispatch)
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(1); // Single workgroup for linear RLE decode
      passEncoder.end();

      //copy result to staging buffer for reading
      const stagingBuffer = device.createBuffer({
        size: outputByteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
      commandEncoder.copyBufferToBuffer(outputGPUBuffer, 0, stagingBuffer, 0, outputByteSize);

      device.queue.submit([commandEncoder.finish()]);

      // 6. Read Back Results
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const copyArrayBuffer = stagingBuffer.getMappedRange().slice(0);
      stagingBuffer.unmap();

      //cleanup (Optional: Buffers are GC'd, but explicit destroy is good for VRAM)
      inputGPUBuffer.destroy();
      outputGPUBuffer.destroy();
      paramBuffer.destroy();
      stagingBuffer.destroy();

      //return raw pixel data (Uint32Array buffer)
      return new Response(copyArrayBuffer, {
        headers: { "Content-Type": "application/octet-stream" }
      });
    }

    return new Response("Method not allowed", { status: 405 });
  }
}