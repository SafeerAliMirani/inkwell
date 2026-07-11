# Canonical preprocessing pipeline. The JS in derisk.html mirrors this 1:1.
import numpy as np
MEAN, STD = 0.1307, 0.3081

def bilinear_resize(src, nh, nw):
    h, w = src.shape
    out = np.empty((nh, nw), np.float32)
    for i in range(nh):
        sy = (i + 0.5) * h / nh - 0.5
        y0 = int(np.floor(sy)); wy = sy - y0
        y0c = min(max(y0, 0), h - 1); y1c = min(max(y0 + 1, 0), h - 1)
        for j in range(nw):
            sx = (j + 0.5) * w / nw - 0.5
            x0 = int(np.floor(sx)); wx = sx - x0
            x0c = min(max(x0, 0), w - 1); x1c = min(max(x0 + 1, 0), w - 1)
            out[i, j] = (src[y0c, x0c] * (1-wy) * (1-wx) + src[y0c, x1c] * (1-wy) * wx
                       + src[y1c, x0c] * wy * (1-wx) + src[y1c, x1c] * wy * wx)
    return out

def preprocess(gray):
    """gray: float [H,W] in [0,1], ink=bright. Returns 28x28 float in [0,1] or None."""
    mask = gray > 0.05
    if not mask.any(): return None
    rows = np.where(mask.any(1))[0]; cols = np.where(mask.any(0))[0]
    crop = gray[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].astype(np.float32)
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh = max(1, int(np.floor(h * s + 0.5))); nw = max(1, int(np.floor(w * s + 0.5)))
    small = bilinear_resize(crop, nh, nw)
    img = np.zeros((28, 28), np.float32)
    oy = (28 - nh) // 2; ox = (28 - nw) // 2
    img[oy:oy+nh, ox:ox+nw] = small
    tot = img.sum()
    ys, xs = np.mgrid[0:28, 0:28]
    comy = (img * ys).sum() / tot; comx = (img * xs).sum() / tot
    dy = int(np.floor(13.5 - comy + 0.5)); dx = int(np.floor(13.5 - comx + 0.5))
    out = np.zeros((28, 28), np.float32)
    ys0, ys1 = max(0, dy), min(28, 28 + dy)
    xs0, xs1 = max(0, dx), min(28, 28 + dx)
    out[ys0:ys1, xs0:xs1] = img[ys0-dy:ys1-dy, xs0-dx:xs1-dx]
    return out

def forward_probs(ps, x784n):
    a1 = 1/(1+np.exp(-np.clip(x784n @ ps["W1"].T + ps["b1"], -60, 60)))
    a2 = 1/(1+np.exp(-np.clip(a1 @ ps["W2"].T + ps["b2"], -60, 60)))
    z3 = a2 @ ps["W3"].T + ps["b3"]; z3 -= z3.max(-1, keepdims=True)
    e = np.exp(z3); return e / e.sum(-1, keepdims=True)

def norm(x): return (x - MEAN) / STD

def load_shipped(path="weights_b64.txt"):
    """Load the shipped weights from the base64 export."""
    import base64
    blob = np.frombuffer(base64.b64decode(open(path).read().strip()), np.float32)
    o = 0; ps = {}
    for name, shape in [("W1",(16,784)),("b1",(16,)),("W2",(16,16)),("b2",(16,)),("W3",(10,16)),("b3",(10,))]:
        n = int(np.prod(shape)); ps[name] = blob[o:o+n].reshape(shape); o += n
    assert o == blob.size
    return ps
