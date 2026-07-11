async function initWebGPU() {
  if (!navigator.gpu) throw new Error("no navigator.gpu");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no adapter");
  const device = await adapter.requestDevice();
  const canvas = document.getElementById("net");
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const U = GPUBufferUsage;
  function buf(data, usage) {
    const b = device.createBuffer({ size: data.byteLength, usage });
    device.queue.writeBuffer(b, 0, data);
    return b;
  }
  const W1 = buf(ps.W1, U.STORAGE | U.COPY_DST), b1 = buf(ps.b1, U.STORAGE | U.COPY_DST);
  const W2 = buf(ps.W2, U.STORAGE | U.COPY_DST), b2 = buf(ps.b2, U.STORAGE | U.COPY_DST);
  const W3 = buf(ps.W3, U.STORAGE | U.COPY_DST), b3 = buf(ps.b3, U.STORAGE | U.COPY_DST);
  const inBuf = device.createBuffer({ size: 784 * 4, usage: U.STORAGE | U.COPY_DST });
  const a1Buf = device.createBuffer({ size: 16 * 4, usage: U.STORAGE | U.COPY_SRC });
  const a2Buf = device.createBuffer({ size: 16 * 4, usage: U.STORAGE | U.COPY_SRC });
  const z3Buf = device.createBuffer({ size: 10 * 4, usage: U.STORAGE | U.COPY_SRC });
  const stag = device.createBuffer({ size: 42 * 4, usage: U.COPY_DST | U.MAP_READ });
  function metaBuf(inDim, outDim, act) { return buf(new Uint32Array([inDim, outDim, act, 0]), U.UNIFORM | U.COPY_DST); }
  const m1 = metaBuf(784, 16, 1), m2 = metaBuf(16, 16, 1), m3 = metaBuf(16, 10, 0);

  const compShader = device.createShaderModule({ code: `
struct Meta { inDim:u32, outDim:u32, act:u32, pad:u32 };
@group(0) @binding(0) var<storage, read> ain: array<f32>;
@group(0) @binding(1) var<storage, read_write> aout: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<uniform> meta: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= meta.outDim) { return; }
  var acc = B[j];
  let base = j * meta.inDim;
  for (var i = 0u; i < meta.inDim; i = i + 1u) { acc = acc + ain[i] * W[base + i]; }
  if (meta.act == 1u) { acc = 1.0 / (1.0 + exp(-acc)); }
  aout[j] = acc;
}` });
  const compPipe = device.createComputePipeline({ layout: "auto", compute: { module: compShader, entryPoint: "main" } });
  const cl = compPipe.getBindGroupLayout(0);
  const bg = (ain, aout, W, B, m) => device.createBindGroup({ layout: cl, entries: [
    { binding: 0, resource: { buffer: ain } }, { binding: 1, resource: { buffer: aout } },
    { binding: 2, resource: { buffer: W } }, { binding: 3, resource: { buffer: B } }, { binding: 4, resource: { buffer: m } }] });
  const bg1 = bg(inBuf, a1Buf, W1, b1, m1), bg2 = bg(a1Buf, a2Buf, W2, b2, m2), bg3 = bg(a2Buf, z3Buf, W3, b3, m3);

  const actBuf = device.createBuffer({ size: IDX.total * 4, usage: U.STORAGE | U.COPY_DST });
  const uni = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
  const neuBuf = buf(NEU, U.VERTEX | U.COPY_DST);
  const connBuf = buf(CONN.data, U.VERTEX | U.COPY_DST);

  const rShader = device.createShaderModule({ code: `
struct Uni { canvas: vec2<f32>, front: f32, pred: i32, showAll: u32, sign: u32, scale: f32, panx: f32, pany: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> act: array<f32>;
fn toClip(p: vec2<f32>) -> vec4<f32> {
  let c = U.canvas * 0.5;
  let pc = (p - c) * U.scale + c + vec2<f32>(U.panx, U.pany);
  return vec4((pc / U.canvas) * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

struct NOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) bright: f32, @location(2) win: f32 };
@vertex fn nv(@builtin(vertex_index) vi: u32, @location(0) ipos: vec2<f32>, @location(1) rad: f32, @location(2) gidxf: f32) -> NOut {
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0));
  let c = corners[vi];
  let gidx = u32(gidxf + 0.5);
  var gate = 1.0;
  if (ipos.x > U.front * U.canvas.x + 8.0) { gate = 0.0; }
  var o: NOut;
  o.pos = toClip(ipos + c * rad);
  o.uv = c;
  o.bright = act[gidx] * gate;
  var win = 0.0;
  if (gidx >= 816u && i32(gidx) - 816 == U.pred && gate > 0.5) { win = 1.0; }
  o.win = win;
  return o;
}
@fragment fn nf(o: NOut) -> @location(0) vec4<f32> {
  let d = length(o.uv);
  if (d > 1.0) { discard; }
  let edge = smoothstep(1.0, 0.82, d);
  let base = vec3<f32>(0.13, 0.15, 0.20);
  var lit = vec3<f32>(1.0, 1.0, 1.0) * clamp(o.bright, 0.0, 1.0);
  if (o.win > 0.5) { lit = mix(lit, vec3<f32>(1.0, 0.78, 0.34), 0.75) + vec3<f32>(0.15, 0.10, 0.0); }
  return vec4<f32>(base + lit, edge);
}

struct COut { @builtin(position) pos: vec4<f32>, @location(0) rgb: vec3<f32> };
@vertex fn cv(@builtin(vertex_index) vi: u32, @location(0) src: vec2<f32>, @location(1) dst: vec2<f32>, @location(2) w: f32, @location(3) sidxf: f32) -> COut {
  let sidx = u32(sidxf + 0.5);
  let a = act[sidx];
  var strength = abs(w) * a;
  if (U.showAll == 1u) { strength = max(strength, abs(w) * 0.16); }
  let frontX = U.front * U.canvas.x + 8.0;
  if (dst.x > frontX) { strength = 0.0; }
  strength = clamp(strength, 0.0, 1.2);
  let half = clamp(0.35 + strength * 2.0, 0.35, 2.8);
  let dir = normalize(dst - src);
  let nrm = vec2<f32>(-dir.y, dir.x);
  var quad = array<vec2<f32>, 6>(vec2<f32>(0.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(0.0,1.0), vec2<f32>(0.0,1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0));
  let q = quad[vi];
  let p = mix(src, dst, q.x) + nrm * (q.y * half);
  var col = vec3<f32>(0.55, 0.65, 0.9);
  if (U.sign == 1u) {
    if (w >= 0.0) { col = vec3<f32>(0.28, 0.55, 1.0); } else { col = vec3<f32>(1.0, 0.34, 0.32); }
  }
  var o: COut;
  o.pos = toClip(p);
  o.rgb = col * strength * 0.9;
  return o;
}
@fragment fn cf(o: COut) -> @location(0) vec4<f32> { return vec4<f32>(o.rgb, 1.0); }` });

  const rLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }] });
  const rPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [rLayout] });
  const rBind = device.createBindGroup({ layout: rLayout, entries: [
    { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: actBuf } }] });

  const connPipe = device.createRenderPipeline({ layout: rPipeLayout,
    vertex: { module: rShader, entryPoint: "cv", buffers: [{ arrayStride: 24, stepMode: "instance", attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x2" }, { shaderLocation: 1, offset: 8, format: "float32x2" },
      { shaderLocation: 2, offset: 16, format: "float32" }, { shaderLocation: 3, offset: 20, format: "float32" }] }] },
    fragment: { module: rShader, entryPoint: "cf", targets: [{ format, blend: {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] },
    primitive: { topology: "triangle-list" } });

  const neuPipe = device.createRenderPipeline({ layout: rPipeLayout,
    vertex: { module: rShader, entryPoint: "nv", buffers: [{ arrayStride: 16, stepMode: "instance", attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x2" }, { shaderLocation: 1, offset: 8, format: "float32" },
      { shaderLocation: 2, offset: 12, format: "float32" }] }] },
    fragment: { module: rShader, entryPoint: "nf", targets: [{ format, blend: {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] },
    primitive: { topology: "triangle-list" } });

  const uniData = new ArrayBuffer(48);
  const uf = new Float32Array(uniData), ui = new Int32Array(uniData), uu = new Uint32Array(uniData);

  async function infer(xNorm) {
    device.queue.writeBuffer(inBuf, 0, Float32Array.from(xNorm));
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(compPipe);
    p.setBindGroup(0, bg1); p.dispatchWorkgroups(1);
    p.setBindGroup(0, bg2); p.dispatchWorkgroups(1);
    p.setBindGroup(0, bg3); p.dispatchWorkgroups(1);
    p.end();
    enc.copyBufferToBuffer(a1Buf, 0, stag, 0, 64);
    enc.copyBufferToBuffer(a2Buf, 0, stag, 64, 64);
    enc.copyBufferToBuffer(z3Buf, 0, stag, 128, 40);
    device.queue.submit([enc.finish()]);
    await stag.mapAsync(GPUMapMode.READ);
    const m = new Float32Array(stag.getMappedRange().slice(0));
    stag.unmap();
    return { a1: m.subarray(0, 16), a2: m.subarray(16, 32), z3: m.subarray(32, 42) };
  }

  function writeActivations(arr) { device.queue.writeBuffer(actBuf, 0, Float32Array.from(arr)); }

  function render(state, pred) {
    uf[0] = canvas.width; uf[1] = canvas.height; uf[2] = state.front;
    ui[3] = pred; uu[4] = state.showAll ? 1 : 0; uu[5] = state.sign ? 1 : 0;
    uf[6] = state.scale; uf[7] = state.panX; uf[8] = state.panY;
    device.queue.writeBuffer(uni, 0, uniData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.043, g: 0.051, b: 0.071, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setBindGroup(0, rBind);
    pass.setPipeline(connPipe); pass.setVertexBuffer(0, connBuf); pass.draw(6, CONN.count);
    pass.setPipeline(neuPipe); pass.setVertexBuffer(0, neuBuf); pass.draw(6, NEURONS.length);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  return { infer, writeActivations, render, device };
}
