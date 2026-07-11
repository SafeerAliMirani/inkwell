import numpy as np, time, os, sys
T_START = time.time(); BUDGET = float(os.environ.get("INKWELL_BUDGET", "1e9"))

RNG_SEED = 7
CKPT = "./ckpt.npz"
BEST = "./best.npz"
EPOCHS = 60
BATCH = 128
LR0, LR1 = 2e-3, 2e-4
MEAN, STD = 0.1307, 0.3081

d = np.load("./mnist.npz")
Xall = d["Xtr"].astype(np.float32) / 255.0
yall = d["ytr"].astype(np.int64)
Xte  = d["Xte"].astype(np.float32) / 255.0
yte  = d["yte"].astype(np.int64)
Xva, yva = Xall[-5000:], yall[-5000:]
Xtr, ytr = Xall[:-5000], yall[:-5000]

def norm(x): return (x - MEAN) / STD

def affine_batch(X, rng, chunk=8192):
    N = X.shape[0]
    out = np.empty_like(X)
    ys, xs = np.mgrid[0:28, 0:28]
    P = np.stack([ys.ravel(), xs.ravel()]).astype(np.float32)
    c = 13.5
    for s0 in range(0, N, chunk):
        xb = X[s0:s0+chunk]; n = xb.shape[0]
        ang = rng.uniform(-12, 12, n).astype(np.float32) * np.pi / 180
        sc  = rng.uniform(0.88, 1.12, n).astype(np.float32)
        ty  = rng.uniform(-2.2, 2.2, n).astype(np.float32)
        tx  = rng.uniform(-2.2, 2.2, n).astype(np.float32)
        ident = rng.random(n) < 0.12
        ang[ident] = 0; sc[ident] = 1; ty[ident] = 0; tx[ident] = 0
        cos, sin = np.cos(ang)/sc, np.sin(ang)/sc
        A = np.stack([np.stack([cos, sin], -1), np.stack([-sin, cos], -1)], 1)
        S = np.einsum("nij,jk->nik", A, P - c)
        sy = S[:, 0, :] + c - ty[:, None]
        sx = S[:, 1, :] + c - tx[:, None]
        y0 = np.floor(sy).astype(np.int32); x0 = np.floor(sx).astype(np.int32)
        wy = sy - y0; wx = sx - x0
        flat = xb.reshape(n, -1)
        def gather(yi, xi):
            valid = (yi >= 0) & (yi < 28) & (xi >= 0) & (xi < 28)
            idx = np.clip(yi, 0, 27) * 28 + np.clip(xi, 0, 27)
            return np.where(valid, np.take_along_axis(flat, idx, 1), 0.0)
        ob = (gather(y0, x0) * (1-wy) * (1-wx) + gather(y0, x0+1) * (1-wy) * wx
            + gather(y0+1, x0) * wy * (1-wx) + gather(y0+1, x0+1) * wy * wx)
        out[s0:s0+chunk] = ob.reshape(n, 28, 28)
    return out

def morph_batch(X, rng):
    r = rng.random(X.shape[0])
    def plus(A, op):
        p = np.pad(A, ((0,0),(1,1),(1,1)))
        s = [p[:,1:-1,1:-1], p[:,:-2,1:-1], p[:,2:,1:-1], p[:,1:-1,:-2], p[:,1:-1,2:]]
        return np.maximum.reduce(s) if op == "max" else np.minimum.reduce(s)
    dil = r < 0.18; ero = (r >= 0.18) & (r < 0.28)
    if dil.any(): X[dil] = plus(X[dil], "max")
    if ero.any(): X[ero] = plus(X[ero], "min")
    return X

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def init_params(rng):
    ps = {}
    for k, (fo, fi) in dict(W1=(16,784), W2=(16,16), W3=(10,16)).items():
        lim = np.sqrt(6.0 / (fi + fo))
        ps[k] = rng.uniform(-lim, lim, (fo, fi)).astype(np.float32)
    ps["b1"] = np.zeros(16, np.float32); ps["b2"] = np.zeros(16, np.float32); ps["b3"] = np.zeros(10, np.float32)
    return ps

def forward(ps, x):
    a1 = sigmoid(x @ ps["W1"].T + ps["b1"])
    a2 = sigmoid(a1 @ ps["W2"].T + ps["b2"])
    z3 = a2 @ ps["W3"].T + ps["b3"]
    z3 = z3 - z3.max(1, keepdims=True)
    e = np.exp(z3); p = e / e.sum(1, keepdims=True)
    return a1, a2, p

def accuracy(ps, X, y, bs=2000):
    hits = 0
    for i in range(0, len(X), bs):
        _, _, p = forward(ps, norm(X[i:i+bs].reshape(-1, 784)))
        hits += (p.argmax(1) == y[i:i+bs]).sum()
    return hits / len(X)

rng = np.random.default_rng(RNG_SEED)
ps = init_params(rng)
adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in ps.items()}
t_adam = 0; ep0 = 0; best_va = 0.0

if os.path.exists(CKPT):
    ck = np.load(CKPT, allow_pickle=True)
    ps = {k: ck[k] for k in ["W1","b1","W2","b2","W3","b3"]}
    adam = {k: [ck["m_"+k], ck["v_"+k]] for k in ps}
    t_adam = int(ck["t_adam"]); ep0 = int(ck["epoch"]) + 1; best_va = float(ck["best_va"])
    rng = np.random.default_rng(RNG_SEED + ep0 * 1000)
    print(f"resumed at epoch {ep0}", flush=True)

Xva_n = norm(Xva.reshape(-1, 784))

for ep in range(ep0, EPOCHS):
    t0 = time.time()
    Xa = affine_batch(Xtr, rng)
    Xa = morph_batch(Xa, rng)
    Xa = norm(Xa.reshape(-1, 784))
    order = rng.permutation(len(Xa))
    lr = LR1 + 0.5 * (LR0 - LR1) * (1 + np.cos(np.pi * ep / (EPOCHS - 1)))
    tot = 0.0
    for i in range(0, len(order) - BATCH + 1, BATCH):
        idx = order[i:i+BATCH]
        xb, yb = Xa[idx], ytr[idx]
        a1, a2, p = forward(ps, xb)
        tot += -np.log(np.maximum(p[np.arange(len(yb)), yb], 1e-12)).sum()
        d3 = p.copy(); d3[np.arange(len(yb)), yb] -= 1.0; d3 /= len(yb)
        g = {}
        g["W3"] = d3.T @ a2; g["b3"] = d3.sum(0)
        d2 = (d3 @ ps["W3"]) * a2 * (1 - a2)
        g["W2"] = d2.T @ a1; g["b2"] = d2.sum(0)
        d1 = (d2 @ ps["W2"]) * a1 * (1 - a1)
        g["W1"] = d1.T @ xb; g["b1"] = d1.sum(0)
        t_adam += 1
        for k in ps:
            m, v = adam[k]
            m[:] = 0.9 * m + 0.1 * g[k]
            v[:] = 0.999 * v + 0.001 * g[k] ** 2
            mh = m / (1 - 0.9 ** t_adam); vh = v / (1 - 0.999 ** t_adam)
            ps[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    va = accuracy(ps, Xva, yva)
    if va > best_va:
        best_va = va
        np.savez(BEST, **ps, val_acc=va, epoch=ep)
    save = dict(**ps, t_adam=t_adam, epoch=ep, best_va=best_va)
    for k in ps: save["m_" + k] = adam[k][0]; save["v_" + k] = adam[k][1]
    np.savez(CKPT, **save)
    print(f"epoch {ep:2d} loss {tot/len(order):.4f} val {va*100:.2f}% best {best_va*100:.2f}% lr {lr:.1e} {time.time()-t0:.1f}s", flush=True)
    if time.time() - T_START > BUDGET:
        print("BUDGET_STOP", flush=True); sys.exit(0)

bp = np.load(BEST)
ps_best = {k: bp[k] for k in ["W1","b1","W2","b2","W3","b3"]}
te = accuracy(ps_best, Xte, yte)
tr = accuracy(ps_best, Xall, yall)
print(f"FINAL best-val params: TEST {te*100:.2f}%  train {tr*100:.2f}%  val {float(bp['val_acc'])*100:.2f}%", flush=True)
np.savez("./final.npz", **ps_best, test_acc=te)
print("DONE", flush=True)
