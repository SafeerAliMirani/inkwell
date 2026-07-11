import numpy as np, base64, json, os
from pipeline import load_shipped, norm

# Pick one confident, correctly classified test image per digit for the
# "load a test digit" button. Writes build/samples.json.
HERE = os.path.dirname(os.path.abspath(__file__))
ps = load_shipped(os.path.join(HERE, "weights_b64.txt"))
d = np.load(os.path.join(HERE, "mnist.npz"))
Xte, yte = d["Xte"], d["yte"]

def probs(x):
    a1 = 1 / (1 + np.exp(-np.clip(x @ ps["W1"].T + ps["b1"], -60, 60)))
    a2 = 1 / (1 + np.exp(-np.clip(a1 @ ps["W2"].T + ps["b2"], -60, 60)))
    z = a2 @ ps["W3"].T + ps["b3"]
    e = np.exp(z - z.max())
    return e / e.sum()

samples = []
for c in range(10):
    best, bestp = None, 0.0
    for k in np.where(yte == c)[0][:400]:
        p = probs(norm(Xte[k].reshape(-1)))
        if p.argmax() == c and p[c] > bestp:
            bestp, best = float(p[c]), int(k)
    samples.append(base64.b64encode(Xte[best].tobytes()).decode())

out = os.path.join(HERE, "..", "build", "samples.json")
json.dump(samples, open(out, "w"))
print("wrote", out, len(samples), "samples")
