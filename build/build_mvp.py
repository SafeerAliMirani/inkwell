import json, os

# Assemble the single self-contained index.html from the template and parts.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "training")

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

core = read(os.path.join(TRAIN, "inkwell_core.js"))
weights = read(os.path.join(TRAIN, "weights_b64.txt")).strip()
acc = json.load(open(os.path.join(TRAIN, "manifest.json")))["mnist_test_accuracy"]
cam3d = read(os.path.join(HERE, "part_cam3d.js"))
wgsl = read(os.path.join(HERE, "part_wgsl.js"))
hover = read(os.path.join(HERE, "part_hover.js"))
sample = read(os.path.join(HERE, "part_sample.js"))
fallback = read(os.path.join(HERE, "part_fallback.js"))
samples = json.load(open(os.path.join(HERE, "samples.json")))
sample_block = "const SAMPLES = " + json.dumps(samples) + ";\n" + sample

html = read(os.path.join(HERE, "index_template.html"))
html = (html
        .replace("%%CORE%%", core)
        .replace("%%WEIGHTS%%", weights)
        .replace("%%ACC%%", f"{acc*100:.1f}%")
        .replace("%%CAM3D%%", cam3d)
        .replace("%%WGSL%%", wgsl)
        .replace("%%HOVER%%", hover)
        .replace("%%SAMPLE%%", sample_block)
        .replace("%%FALLBACK%%", fallback))

out = os.path.join(ROOT, "index.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out, len(html), "bytes")
