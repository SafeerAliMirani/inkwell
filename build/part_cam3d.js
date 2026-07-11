// 3D camera maths and the two scene layouts. Column-major mat4, WebGPU depth 0..1.
const NN3D = (function () {
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f; m[10] = far * nf; m[11] = -1; m[14] = near * far * nf;
    return m;
  }
  function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
  function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
  function lookAt(eye, center, up) {
    const f = norm([center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]]);
    const s = norm(cross(f, up));
    const u = cross(s, f);
    const m = new Float32Array(16);
    m[0]=s[0]; m[4]=s[1]; m[8]=s[2];
    m[1]=u[0]; m[5]=u[1]; m[9]=u[2];
    m[2]=-f[0]; m[6]=-f[1]; m[10]=-f[2];
    m[12]=-dot(s,eye); m[13]=-dot(u,eye); m[14]=dot(f,eye); m[15]=1;
    return { view: m, right: s, up: u };
  }
  // Perspective orbit around the origin.
  function view3d(state, aspect) {
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const eye = [state.dist * cp * sy, state.dist * sp, state.dist * cp * cy];
    const la = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const proj = perspective(50 * Math.PI / 180, aspect, 0.05, 100);
    return { mvp: mul(proj, la.view), right: la.right, up: la.up, camPos: eye };
  }
  // Orthographic matrix that reproduces the flat pixel mapping with pan and zoom.
  function view2d(W, H, state) {
    const cxp = W / 2, cyp = H / 2;
    const tx = cxp * (1 - state.scale) + state.panX;
    const ty = cyp * (1 - state.scale) + state.panY;
    const m = new Float32Array(16);
    m[0] = 2 * state.scale / W; m[12] = 2 * tx / W - 1;
    m[5] = -2 * state.scale / H; m[13] = 1 - 2 * ty / H;
    m[10] = 1; m[15] = 1;
    return { mvp: m, right: [1, 0, 0], up: [0, 1, 0], camPos: [0, 0, 1] };
  }
  // 3D scene: input panel in front, hidden columns and output receding in depth.
  function layout3d(ps, pct) {
    const s1 = pct(ps.W1, 99), s2 = pct(ps.W2, 99), s3 = pct(ps.W3, 99);
    const input = [], h1 = [], h2 = [], out = [];
    const zin = 1.7, zh1 = 0.55, zh2 = -0.55, zout = -1.7;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++)
      input.push([ (c - 13.5) / 13.5 * 1.05, (13.5 - r) / 13.5 * 1.05, zin, 0.02 ]);
    const colY = (n, sp) => { const a = []; for (let i = 0; i < n; i++) a.push(n === 1 ? 0 : (sp - 2 * sp * i / (n - 1))); return a; };
    const y16 = colY(16, 1.25), y10 = colY(10, 1.05);
    for (let i = 0; i < 16; i++) h1.push([0, y16[i], zh1, 0.07]);
    for (let i = 0; i < 16; i++) h2.push([0, y16[i], zh2, 0.07]);
    for (let i = 0; i < 10; i++) out.push([0, y10[i], zout, 0.09]);
    const neurons = [...input, ...h1, ...h2, ...out];
    const neu = new Float32Array(neurons.length * 6);
    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      neu[i*6] = n[0]; neu[i*6+1] = n[1]; neu[i*6+2] = n[2]; neu[i*6+3] = n[3]; neu[i*6+4] = i; neu[i*6+5] = 0;
    }
    const inst = [];
    const add = (a, b, w, sc, sidx) => inst.push(a[0], a[1], a[2], b[0], b[1], b[2], w / sc, sidx);
    for (let j = 0; j < 16; j++) for (let i = 0; i < 784; i++) add(input[i], h1[j], ps.W1[j*784+i], s1, i);
    for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) add(h1[i], h2[j], ps.W2[j*16+i], s2, 784 + i);
    for (let j = 0; j < 10; j++) for (let i = 0; i < 16; i++) add(h2[i], out[j], ps.W3[j*16+i], s3, 800 + i);
    return { neu, conn: new Float32Array(inst), count: inst.length / 8, positions: neurons };
  }
  function project(mvp, p, W, H) {
    const x = mvp[0]*p[0] + mvp[4]*p[1] + mvp[8]*p[2] + mvp[12];
    const y = mvp[1]*p[0] + mvp[5]*p[1] + mvp[9]*p[2] + mvp[13];
    const w = mvp[3]*p[0] + mvp[7]*p[1] + mvp[11]*p[2] + mvp[15];
    if (w <= 0) return null;
    return [ (x / w * 0.5 + 0.5) * W, (0.5 - y / w * 0.5) * H ];
  }
  return { mul, perspective, lookAt, view3d, view2d, layout3d, project };
})();
if (typeof module !== "undefined") module.exports = NN3D;
