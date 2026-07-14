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
  const uni = device.createBuffer({ size: 144, usage: U.UNIFORM | U.COPY_DST });

  // instance data, both scenes. 2D positions are in pixels with z 0, 3D in world units.
  const NEU2 = new Float32Array(NEURONS.length * 6);
  for (let i = 0; i < NEURONS.length; i++) { NEU2[i*6]=NEURONS[i].x; NEU2[i*6+1]=NEURONS[i].y; NEU2[i*6+2]=0; NEU2[i*6+3]=NEURONS[i].r; NEU2[i*6+4]=i; }
  const neu2 = buf(NEU2, U.VERTEX | U.COPY_DST);
  const conn2 = buf(CONN.data, U.VERTEX | U.COPY_DST);
  const L3 = NN3D.layout3d(ps, pct);
  const neu3 = buf(L3.neu, U.VERTEX | U.COPY_DST);
  const conn3 = buf(L3.conn, U.VERTEX | U.COPY_DST);

  const depthTex = device.createTexture({ size: [canvas.width, canvas.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const depthView = depthTex.createView();

  const rShader = device.createShaderModule({ code: `
struct Uni {
  mvp: mat4x4<f32>,
  right: vec3<f32>, up: vec3<f32>, camPos: vec3<f32>,
  front: f32, mode: u32, showAll: u32, sign: u32,
  pred: i32, unit: f32, netW: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> act: array<f32>;

struct NOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) bright: f32, @location(2) win: f32 };
@vertex fn nv(@builtin(vertex_index) vi: u32, @location(0) p: vec3<f32>, @location(1) rad: f32, @location(2) gidxf: f32) -> NOut {
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0));
  let c = corners[vi];
  let gidx = u32(gidxf + 0.5);
  let world = p + U.right * (c.x * rad) + U.up * (c.y * rad);
  var gate = 1.0;
  if (U.mode == 0u && p.x > U.front * U.netW + 8.0) { gate = 0.0; }
  var o: NOut;
  o.pos = U.mvp * vec4<f32>(world, 1.0);
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
@vertex fn cv(@builtin(vertex_index) vi: u32, @location(0) src: vec3<f32>, @location(1) dst: vec3<f32>, @location(2) w: f32, @location(3) sidxf: f32) -> COut {
  let sidx = u32(sidxf + 0.5);
  let a = act[sidx];
  var strength = abs(w) * a;
  if (U.showAll == 1u) { strength = max(strength, abs(w) * 0.16); }
  if (U.mode == 0u && dst.x > U.front * U.netW + 8.0) { strength = 0.0; }
  strength = clamp(strength, 0.0, 1.2);
  let half = clamp(0.35 + strength * 2.0, 0.5, 2.8) * U.unit;
  let mid = mix(src, dst, 0.5);
  var toCam = vec3<f32>(0.0, 0.0, 1.0);
  if (U.mode == 1u) { toCam = normalize(U.camPos - mid); }
  var dir = dst - src;
  if (length(dir) < 1e-6) { dir = vec3<f32>(1.0, 0.0, 0.0); }
  dir = normalize(dir);
  var side = cross(dir, toCam);
  if (length(side) < 1e-5) { side = vec3<f32>(0.0, 1.0, 0.0); }
  side = normalize(side);
  var quad = array<vec2<f32>, 6>(vec2<f32>(0.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(0.0,1.0), vec2<f32>(0.0,1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0));
  let q = quad[vi];
  let p = mix(src, dst, q.x) + side * (q.y * half);
  var bright = 0.9;
  if (U.mode == 1u) { bright = 1.05 * clamp(3.0 / distance(U.camPos, mid), 0.5, 1.0); }
  var col = vec3<f32>(0.55, 0.65, 0.9);
  if (U.sign == 1u) {
    if (w >= 0.0) { col = vec3<f32>(0.28, 0.55, 1.0); } else { col = vec3<f32>(1.0, 0.34, 0.32); }
  }
  var o: COut;
  o.pos = U.mvp * vec4<f32>(p, 1.0);
  o.rgb = col * strength * bright;
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
    vertex: { module: rShader, entryPoint: "cv", buffers: [{ arrayStride: 32, stepMode: "instance", attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32" }, { shaderLocation: 3, offset: 28, format: "float32" }] }] },
    fragment: { module: rShader, entryPoint: "cf", targets: [{ format, blend: {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" } });

  const neuPipe = device.createRenderPipeline({ layout: rPipeLayout,
    vertex: { module: rShader, entryPoint: "nv", buffers: [{ arrayStride: 24, stepMode: "instance", attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32" },
      { shaderLocation: 2, offset: 16, format: "float32" }] }] },
    fragment: { module: rShader, entryPoint: "nf", targets: [{ format, blend: {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less-equal" } });

  const uniData = new ArrayBuffer(144);
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
    const aspect = canvas.width / canvas.height;
    const cam = state.view === "3d" ? NN3D.view3d(state, aspect) : NN3D.view2d(canvas.width, canvas.height, state);
    const mode = state.view === "3d" ? 1 : 0;
    uf.set(cam.mvp, 0);
    uf[16] = cam.right[0]; uf[17] = cam.right[1]; uf[18] = cam.right[2];
    uf[20] = cam.up[0]; uf[21] = cam.up[1]; uf[22] = cam.up[2];
    uf[24] = cam.camPos[0]; uf[25] = cam.camPos[1]; uf[26] = cam.camPos[2];
    uf[27] = state.front; uu[28] = mode; uu[29] = state.showAll ? 1 : 0; uu[30] = state.sign ? 1 : 0;
    ui[31] = pred; uf[32] = mode === 1 ? 0.007 : 1.0; uf[33] = canvas.width;
    device.queue.writeBuffer(uni, 0, uniData);
    const nBuf = mode === 1 ? neu3 : neu2, cBuf = mode === 1 ? conn3 : conn2;
    const cCount = mode === 1 ? L3.count : CONN.count;
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.043, g: 0.051, b: 0.071, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" } });
    pass.setBindGroup(0, rBind);
    pass.setPipeline(connPipe); pass.setVertexBuffer(0, cBuf); pass.draw(6, cCount);
    pass.setPipeline(neuPipe); pass.setVertexBuffer(0, nBuf); pass.draw(6, NEURONS.length);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  return { infer, writeActivations, render, device };
}
