// Inkwell de-risk core: weight decoding, preprocessing, inference.
// Plain JS, no libraries. Mirrors pipeline.py exactly; keep the two in sync.
const INKWELL = (function () {
  const MEAN = 0.1307, STD = 0.3081;
  const DIMS = [784, 16, 16, 10];

  function decodeWeights(b64) {
    let bytes;
    if (typeof atob !== "undefined") {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else {
      bytes = new Uint8Array(Buffer.from(b64, "base64"));
    }
    const f = new Float32Array(bytes.buffer); // little-endian platforms
    let o = 0;
    const take = (n) => { const v = f.subarray(o, o + n); o += n; return v; };
    const ps = {
      W1: take(16 * 784), b1: take(16),
      W2: take(16 * 16),  b2: take(16),
      W3: take(10 * 16),  b3: take(10),
    };
    if (o !== f.length) throw new Error("weight blob size mismatch");
    return ps;
  }

  // box-average a square grayscale image down by integer factor
  function boxAvg(gray, size, factor) {
    const n = size / factor, out = new Float32Array(n * n), inv = 1 / (factor * factor);
    for (let by = 0; by < n; by++) for (let bx = 0; bx < n; bx++) {
      let s = 0;
      for (let dy = 0; dy < factor; dy++) for (let dx = 0; dx < factor; dx++)
        s += gray[(by * factor + dy) * size + (bx * factor + dx)];
      out[by * n + bx] = s * inv;
    }
    return out;
  }

  function bilinearResize(src, h, w, nh, nw) {
    const out = new Float32Array(nh * nw);
    for (let i = 0; i < nh; i++) {
      const sy = (i + 0.5) * h / nh - 0.5;
      let y0 = Math.floor(sy); const wy = sy - y0;
      const y0c = Math.min(Math.max(y0, 0), h - 1), y1c = Math.min(Math.max(y0 + 1, 0), h - 1);
      for (let j = 0; j < nw; j++) {
        const sx = (j + 0.5) * w / nw - 0.5;
        let x0 = Math.floor(sx); const wx = sx - x0;
        const x0c = Math.min(Math.max(x0, 0), w - 1), x1c = Math.min(Math.max(x0 + 1, 0), w - 1);
        out[i * nw + j] = src[y0c * w + x0c] * (1 - wy) * (1 - wx) + src[y0c * w + x1c] * (1 - wy) * wx
                        + src[y1c * w + x0c] * wy * (1 - wx) + src[y1c * w + x1c] * wy * wx;
      }
    }
    return out;
  }

  // MNIST-style preprocessing: threshold, bbox, longest side to 20px, paste in
  // 28x28, integer shift so the intensity centre of mass sits at (13.5, 13.5).
  function preprocess(gray, H, W) {
    let y0 = H, y1 = -1, x0 = W, x1 = -1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (gray[y * W + x] > 0.05) {
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x < x0) x0 = x; if (x > x1) x1 = x;
      }
    }
    if (y1 < 0) return null;
    const ch = y1 - y0 + 1, cw = x1 - x0 + 1;
    const crop = new Float32Array(ch * cw);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++)
      crop[y * cw + x] = gray[(y + y0) * W + (x + x0)];
    const s = 20 / Math.max(ch, cw);
    const nh = Math.max(1, Math.floor(ch * s + 0.5)), nw = Math.max(1, Math.floor(cw * s + 0.5));
    const small = bilinearResize(crop, ch, cw, nh, nw);
    const img = new Float32Array(28 * 28);
    const oy = Math.floor((28 - nh) / 2), ox = Math.floor((28 - nw) / 2);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++)
      img[(y + oy) * 28 + (x + ox)] = small[y * nw + x];
    let tot = 0, my = 0, mx = 0;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const v = img[y * 28 + x]; tot += v; my += v * y; mx += v * x;
    }
    const dy = Math.floor(13.5 - my / tot + 0.5), dx = Math.floor(13.5 - mx / tot + 0.5);
    const out = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const sy2 = y - dy, sx2 = x - dx;
      if (sy2 >= 0 && sy2 < 28 && sx2 >= 0 && sx2 < 28) out[y * 28 + x] = img[sy2 * 28 + sx2];
    }
    return out;
  }

  function normalize(img784) {
    const x = new Float64Array(784);
    for (let i = 0; i < 784; i++) x[i] = (img784[i] - MEAN) / STD;
    return x;
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, z)))); }

  // one loop per output neuron, same shape as the future WGSL kernel
  function layer(W, b, aIn, inDim, outDim, act) {
    const aOut = new Float64Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let acc = b[j];
      for (let i = 0; i < inDim; i++) acc += aIn[i] * W[j * inDim + i];
      aOut[j] = act ? sigmoid(acc) : acc;
    }
    return aOut;
  }

  function forward(ps, x784) {
    const a1 = layer(ps.W1, ps.b1, x784, 784, 16, true);
    const a2 = layer(ps.W2, ps.b2, a1, 16, 16, true);
    const z3 = layer(ps.W3, ps.b3, a2, 16, 10, false);
    let m = -Infinity;
    for (let j = 0; j < 10; j++) if (z3[j] > m) m = z3[j];
    let sum = 0; const probs = new Float64Array(10);
    for (let j = 0; j < 10; j++) { probs[j] = Math.exp(z3[j] - m); sum += probs[j]; }
    for (let j = 0; j < 10; j++) probs[j] /= sum;
    return { a1, a2, probs };
  }

  return { MEAN, STD, DIMS, decodeWeights, boxAvg, bilinearResize, preprocess, normalize, forward };
})();
if (typeof module !== "undefined") module.exports = INKWELL;
