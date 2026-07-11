import numpy as np, json, base64, sys
sys.path.insert(0, ".")
from pipeline import forward_probs, norm

b = np.load("./best_ft.npz")
ps = {k: b[k].astype(np.float32) for k in ["W1","b1","W2","b2","W3","b3"]}
d = np.load("./mnist.npz")
Xte = d["Xte"].astype(np.float32)/255.0; yte = d["yte"].astype(np.int64)

# full-test accuracy with these exact float32 weights
x = norm(Xte.reshape(-1, 784))
a1 = 1/(1+np.exp(-np.clip(x @ ps["W1"].T + ps["b1"], -60, 60)))
a2 = 1/(1+np.exp(-np.clip(a1 @ ps["W2"].T + ps["b2"], -60, 60)))
z3 = a2 @ ps["W3"].T + ps["b3"]
acc = float((z3.argmax(1) == yte).mean())
print(f"shipped weights test acc: {acc*100:.2f}%")

# per-layer activation stats over the test set (for MVP colour normalisation)
stats = {
  "a1": {"p1": float(np.percentile(a1, 1)), "p50": float(np.percentile(a1, 50)), "p99": float(np.percentile(a1, 99))},
  "a2": {"p1": float(np.percentile(a2, 1)), "p50": float(np.percentile(a2, 50)), "p99": float(np.percentile(a2, 99))},
}
order = ["W1","b1","W2","b2","W3","b3"]
blob = b"".join(ps[k].ravel().tobytes() for k in order)
open("./weights.bin","wb").write(blob)
manifest = {
  "name": "inkwell-mlp",
  "architecture": [784, 16, 16, 10],
  "hidden_activation": "sigmoid",
  "output": "softmax",
  "layout": "row-major, W[j*IN_DIM+i] = weight from input i to output neuron j",
  "order": [{"name":"W1","shape":[16,784]},{"name":"b1","shape":[16]},
             {"name":"W2","shape":[16,16]},{"name":"b2","shape":[16]},
             {"name":"W3","shape":[10,16]},{"name":"b3","shape":[10]}],
  "dtype": "float32-le",
  "bytes": len(blob),
  "normalization": {"formula": "(pixel/255 - mean)/std", "mean": 0.1307, "std": 0.3081},
  "preprocessing": "threshold>0.05, crop bbox, scale longest side to 20px (bilinear, preserve aspect), paste centred in 28x28, integer shift so intensity centre of mass sits at (13.5, 13.5)",
  "mnist_test_accuracy": round(acc, 4),
  "training": "55k MNIST train (5k held out for val), numpy Adam, augmentation: rotate/scale/shift + mild dilate/erode, then low-LR finetune with milder augmentation; seed 7",
  "data_source": "MNIST idx files, canonical MD5-verified (github.com/fgnt/mnist mirror of yann.lecun.com originals)",
  "activation_stats": stats,
}
json.dump(manifest, open("./manifest.json","w"), indent=2)
open("./weights_b64.txt","w").write(base64.b64encode(blob).decode())
print("weights.bin", len(blob), "bytes; b64", len(base64.b64encode(blob)), "chars")
print(json.dumps(stats))
