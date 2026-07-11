let sampleTurn = 0;
function loadSample() {
  const b64 = SAMPLES[sampleTurn % SAMPLES.length];
  sampleTurn++;
  const bin = atob(b64), px = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) px[i] = bin.charCodeAt(i);
  resetDraw();
  const cell = 280 / 28;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const v = px[y * 28 + x] / 255;
    if (v < 0.04) continue;
    dctx.fillStyle = "rgba(255,255,255," + v.toFixed(3) + ")";
    dctx.fillRect(x * cell, y * cell, cell + 0.6, cell + 0.6);
  }
  classify(true);
}
