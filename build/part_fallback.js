let fctx = null;
function initFallback() { fctx = document.getElementById("net").getContext("2d"); }
function fallbackRender() {
  if (!fctx) return;
  const cv = document.getElementById("net"), W = cv.width, H = cv.height;
  fctx.fillStyle = "#0b0d12"; fctx.fillRect(0, 0, W, H);
  const frontX = state.front * W + 8;
  const s1 = pct(ps.W1, 99), s2 = pct(ps.W2, 99), s3 = pct(ps.W3, 99);
  fctx.globalCompositeOperation = "lighter";
  function conns(srcArr, dstArr, Wt, inN, outN, scale, baseIdx) {
    for (let j = 0; j < outN; j++) {
      const dst = dstArr[j]; if (dst.x > frontX) continue;
      for (let i = 0; i < inN; i++) {
        const w = Wt[j * inN + i] / scale, a = actAll[baseIdx + i];
        let s = Math.abs(w) * a; if (state.showAll) s = Math.max(s, Math.abs(w) * 0.16);
        if (s < 0.03) continue;
        const src = srcArr[i];
        fctx.strokeStyle = !state.sign ? "rgba(150,170,230," + Math.min(1, s).toFixed(3) + ")"
          : (w >= 0 ? "rgba(70,140,255," + Math.min(1, s).toFixed(3) + ")" : "rgba(255,90,80," + Math.min(1, s).toFixed(3) + ")");
        fctx.lineWidth = Math.min(2.5, 0.4 + s * 2);
        fctx.beginPath(); fctx.moveTo(src.x, src.y); fctx.lineTo(dst.x, dst.y); fctx.stroke();
      }
    }
  }
  conns(L.input, L.h1, ps.W1, 784, 16, s1, IDX.input);
  conns(L.h1, L.h2, ps.W2, 16, 16, s2, IDX.h1);
  conns(L.h2, L.out, ps.W3, 16, 10, s3, IDX.h2);
  fctx.globalCompositeOperation = "source-over";
  for (let i = 0; i < NEURONS.length; i++) {
    const n = NEURONS[i]; if (n.x > frontX) continue;
    const a = Math.min(1, Math.max(0, actAll[i]));
    const win = i >= IDX.out && (i - IDX.out) === predicted;
    fctx.beginPath(); fctx.arc(n.x, n.y, n.r, 0, 7);
    fctx.fillStyle = win ? "rgba(255,200,90,1)" : "rgb(" + Math.round(33 + a*222) + "," + Math.round(38 + a*217) + "," + Math.round(51 + a*204) + ")";
    fctx.fill();
    fctx.lineWidth = 1; fctx.strokeStyle = "rgba(120,130,150,0.5)"; fctx.stroke();
  }
}
