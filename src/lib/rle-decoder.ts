//src/lib/rle-decoder.ts
//WebGPU RLE Decompression - extracted from CrownMuralController
//added destroy(), device loss handling, abort signal support

const RLE_DECODE_SHADER = `
@group(0) @binding(0) var<storage, read> rle_input : array<u32>;
@group(0) @binding(1) var<storage, read_write> pixel_output : array<u32>;

struct Params {
  output_size: u32,
};
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  var rle_index: u32 = 1u;
  var pixel_index: u32 = 0u;
  var current_val: u32 = rle_input[0];
  
  let total_ints = arrayLength(&rle_input);
  
  for (var i = 1u; i < total_ints; i = i + 1u) {
    let run_length = rle_input[i];
    
    for (var j = 0u; j < run_length; j = j + 1u) {
      if (pixel_index < params.output_size) {
        pixel_output[pixel_index] = current_val;
        pixel_index = pixel_index + 1u;
      }
    }
    
    current_val = 1u - current_val;
  }
}
`;

export class RLEDecoder {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private shaderModule: GPUShaderModule | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private initPromise: Promise<void> | null = null;
  private deviceLost = false;

  async ensureReady(): Promise<GPUDevice> {
    //reset if device was lost
    if (this.device && this.deviceLost) {
      this.device = null;
      this.deviceLost = false;
      this.initPromise = null;
    }

    if (this.device) return this.device;

    this.initPromise ??= (async () => {
      if (!navigator.gpu) {
        throw new Error("WebGPU not supported");
      }
      
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) throw new Error("No GPU adapter found");

      this.device = await this.adapter.requestDevice();
      
      // FIX: Handle device loss for recovery
      this.device.lost.then((info) => {
        console.warn('WebGPU device lost:', info.message, info.reason);
        this.deviceLost = true;
      });

      this.shaderModule = this.device.createShaderModule({ code: RLE_DECODE_SHADER });
      this.computePipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.shaderModule, entryPoint: "main" },
      });
    })();

    await this.initPromise;
    return this.device!;
  }

  //added destroy method for cleanup on unmount
  destroy(): void {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.shaderModule = null;
    this.computePipeline = null;
    this.initPromise = null;
    this.deviceLost = false;
  }

  async decode(
    rleBuffer: ArrayBuffer, 
    outputSize: number, 
    width: number, 
    height: number,
    signal?: AbortSignal  //otional abort signal for cancellation
  ): Promise<ImageData> {
    //check if aborted before starting
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }

    const device = await this.ensureReady();
    const outputByteSize = outputSize * 4;

    const rleBytes = new Uint8Array(rleBuffer);
    const paddedSize = (rleBytes.byteLength + 3) & ~3;
    const rlePadded = paddedSize === rleBytes.byteLength ? rleBytes : (() => {
      const p = new Uint8Array(paddedSize);
      p.set(rleBytes);
      return p;
    })();

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

    await stagingBuf.mapAsync(GPUMapMode.READ);

    //ifmaborted after async GPU work
    if (signal?.aborted) {
      stagingBuf.unmap();
      inputBuf.destroy();
      outputBuf.destroy();
      paramBuf.destroy();
      stagingBuf.destroy();
      throw new DOMException('Aborted', 'AbortError');
    }

    const rawData = new Uint32Array(stagingBuf.getMappedRange());

    const imgData = new ImageData(width, height);
    const px = imgData.data;
    for (let i = 0; i < outputSize; i++) {
      const val = rawData[i];
      if (val > 0) {
        const idx = i * 4;
        px[idx] = 255;
        px[idx + 1] = 255;
        px[idx + 2] = 255;
        px[idx + 3] = 255;
      }
    }

    stagingBuf.unmap();
    inputBuf.destroy();
    outputBuf.destroy();
    paramBuf.destroy();
    stagingBuf.destroy();

    return imgData;
  }
}
