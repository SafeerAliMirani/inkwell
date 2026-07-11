# Fetch canonical MNIST and build mnist.npz. Verifies the canonical MD5s.
import hashlib, gzip, subprocess, os, numpy as np
CANON = {
 "train-images-idx3-ubyte.gz":"f68b3c2dcbeaaa9fbdd348bbdeb94873",
 "train-labels-idx1-ubyte.gz":"d53e105ee54ea40749a09fcbcd1e9432",
 "t10k-images-idx3-ubyte.gz":"9fb629c4189551a2d022fa330f9573f3",
 "t10k-labels-idx1-ubyte.gz":"ec29112dd5afa0611ce80d1b7f02629c"}
if not os.path.isdir("mnist_repo"):
    subprocess.run(["git","clone","--depth","1","https://github.com/fgnt/mnist","mnist_repo"], check=True)
for f, m in CANON.items():
    h = hashlib.md5(open(os.path.join("mnist_repo", f),"rb").read()).hexdigest()
    assert h == m, f"{f}: md5 {h} does not match canonical {m}"
    print(f, "md5 ok")
def idx(f, img):
    d = gzip.open(os.path.join("mnist_repo", f)).read()
    a = np.frombuffer(d, np.uint8, offset=16 if img else 8)
    return a.reshape(-1, 28, 28) if img else a
np.savez_compressed("mnist.npz",
    Xtr=idx("train-images-idx3-ubyte.gz",1), ytr=idx("train-labels-idx1-ubyte.gz",0),
    Xte=idx("t10k-images-idx3-ubyte.gz",1),  yte=idx("t10k-labels-idx1-ubyte.gz",0))
print("mnist.npz written")
