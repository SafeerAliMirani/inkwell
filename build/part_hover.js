(function () {
  const net = document.getElementById("net");
  const hover = document.getElementById("hover");
  const hcv = document.getElementById("hoverCv");
  const hlab = document.getElementById("hoverLab");
  const hctx = hcv.getContext("2d");
  function colr(v) {
    const BG = [20, 23, 30], POS = [70, 150, 255], NEG = [240, 90, 80];
    const t = Math.min(1, Math.abs(v)), c = v >= 0 ? POS : NEG;
    return [0, 1, 2].map(i => Math.round(BG[i] + (c[i] - BG[i]) * t));
  }
  function paint(kind, weights, scale) {
    if (kind === "grid") {
      hcv.width = 28; hcv.height = 28; hcv.style.width = "112px"; hcv.style.height = "112px";
      const im = hctx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) { const [r, g, b] = colr(weights[i] / scale); im.data[i*4]=r; im.data[i*4+1]=g; im.data[i*4+2]=b; im.data[i*4+3]=255; }
      hctx.putImageData(im, 0, 0);
    } else {
      const n = weights.length; hcv.width = n; hcv.height = 1; hcv.style.width = (n * 11) + "px"; hcv.style.height = "20px";
      const im = hctx.createImageData(n, 1);
      for (let i = 0; i < n; i++) { const [r, g, b] = colr(weights[i] / scale); im.data[i*4]=r; im.data[i*4+1]=g; im.data[i*4+2]=b; im.data[i*4+3]=255; }
      hctx.putImageData(im, 0, 0);
    }
  }
  const s1 = pct(ps.W1, 99), s2 = pct(ps.W2, 99), s3 = pct(ps.W3, 99);
  net.addEventListener("pointermove", e => {
    if (state.dragging) { hover.style.display = "none"; return; }
    const r = net.getBoundingClientRect();
    const bx = (e.clientX - r.left) * (net.width / r.width), by = (e.clientY - r.top) * (net.height / r.height);
    // undo the camera so hit testing is in the network's own coordinates
    const cx = net.width / 2, cy = net.height / 2;
    const wx = (bx - cx - state.panX) / state.scale + cx, wy = (by - cy - state.panY) / state.scale + cy;
    let best = -1, bd = 1e9;
    for (let i = IDX.h1; i < NEURONS.length; i++) { const dx = NEURONS[i].x - wx, dy = NEURONS[i].y - wy, d = dx*dx + dy*dy; if (d < bd) { bd = d; best = i; } }
    if (best < 0) { hover.style.display = "none"; return; }
    const nr = NEURONS[best].r * 1.9;
    if (bd > nr * nr) { hover.style.display = "none"; return; }
    let kind, weights, scale, lab;
    if (best < IDX.h2) { const j = best - IDX.h1; kind = "grid"; weights = ps.W1.subarray(j*784, j*784+784); scale = s1; lab = "hidden 1, neuron " + (j+1) + ": its 784 input weights"; }
    else if (best < IDX.out) { const j = best - IDX.h2; kind = "strip"; weights = ps.W2.subarray(j*16, j*16+16); scale = s2; lab = "hidden 2, neuron " + (j+1) + ": 16 weights from layer 1"; }
    else { const d = best - IDX.out; kind = "strip"; weights = ps.W3.subarray(d*16, d*16+16); scale = s3; lab = "output " + d + ": 16 weights from layer 2"; }
    paint(kind, weights, scale);
    hlab.textContent = lab;
    hover.style.display = "block";
    const wrap = net.parentElement.getBoundingClientRect();
    hover.style.left = (e.clientX - wrap.left + 16) + "px";
    hover.style.top = (e.clientY - wrap.top + 16) + "px";
  });
  net.addEventListener("pointerleave", () => hover.style.display = "none");
})();
