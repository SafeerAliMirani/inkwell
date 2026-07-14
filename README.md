<div align="center">

# Inkwell

**Ink a digit and watch the exact 3Blue1Brown network recognise it, live, flat or in 3D, with the matrix maths running in raw WGSL on your GPU.**

[![Live demo](https://img.shields.io/badge/live-inkwell--3h3.pages.dev-E8B34B?style=for-the-badge)](https://inkwell-3h3.pages.dev)
&nbsp;
![WebGPU](https://img.shields.io/badge/WebGPU-raw%20WGSL-1f6feb?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/dependencies-none-2ea043?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-8957e5?style=for-the-badge)
![Model](https://img.shields.io/badge/model-784--16--16--10%20MLP-db6d28?style=for-the-badge)

</div>

Draw a digit in the box and a real neural network classifies it as you draw. It is the small network from 3Blue1Brown's "But what is a neural network?", a 784-16-16-10 perceptron of about 13,000 parameters, drawn as columns of neurons joined by weighted connections, in a flat 2D layout or a 3D view you can orbit. The matrix multiplies run in hand-written WGSL compute shaders, checked live against a plain JavaScript copy of the same network. No TensorFlow.js, no ONNX, no three.js, no d3. Both the maths and the drawing run in raw WebGPU.

## What you are looking at

- **The exact 3Blue1Brown network.** A 784-16-16-10 multilayer perceptron with sigmoid hidden layers and a softmax output, about 13,000 weights, every one of them on screen.
- **The input layer is the image.** The 28x28 grid on the left is 784 input neurons, lit by what the network actually receives after your drawing is centred and scaled the MNIST way.
- **Live signal, not decoration.** Each connection lights by its weight times the signal flowing through it, so only the paths that carry the digit light up, and they change as you draw.
- **A signal you can follow.** Every stroke runs a left to right sweep, a slowed replay of the real order the three compute passes execute in.
- **Flat or in 3D.** A toggle switches between the flat 3Blue1Brown layout and a 3D view with the layers set out in depth. Drag to pan or orbit, scroll or pinch to zoom, double click to reset.
- **Hover any neuron.** In either view, a hidden-layer-1 neuron shows its 784 weights as a 28x28 picture, the pattern it looks for. Blue is a positive weight, red is negative.

## How it works

1. `training/fetch_data.py` downloads the canonical MNIST files and verifies all four against their published checksums.
2. `training/train.py` then `training/finetune.py` train the sigmoid MLP in NumPy with light augmentation (small shifts, rotations, scales), holding out a validation split for model selection.
3. `training/export.py` writes `weights.bin` and `manifest.json`, plus a base64 copy for embedding. `training/make_samples.py` picks one clean test digit per class for the demo button.
4. `build/build_mvp.py` assembles the single `index.html` from `index_template.html` and the parts: `part_wgsl.js` (WGSL compute inference and the instanced neuron and connection renderer), `part_cam3d.js` (the flat and 3D camera maths and layout), `part_hover.js`, `part_sample.js`, and a Canvas2D fallback in `part_fallback.js`.
5. In the browser, your drawing is preprocessed by the pipeline in `inkwell_core.js` (threshold, bounding box, scale the longest side to 20px, centre by intensity), three WGSL compute dispatches produce the activations, they are read back to drive the render, and a plain JavaScript copy of the network checks the GPU result on every stroke.

## Run it

```bash
# rebuild the model (optional, the weights are committed)
cd training
python3 fetch_data.py
python3 train.py
python3 finetune.py
python3 export.py
python3 make_samples.py

# rebuild the page from the parts
cd ../build
python3 build_mvp.py

# serve it locally, needs WebGPU (current Chrome or Edge)
cd ..
python3 -m http.server 8000
```

## Honest notes

- The measured accuracy is 95.8% on the MNIST test set, on centred MNIST digits. That is not a claim about all handwriting.
- The network is small and legible on purpose. A small convolutional network passes 99.7% on MNIST, but its feature maps do not draw as individual neurons and weights. This is the network you can see all of.
- The data is real. The MNIST files are checksum verified against the original distribution. Nothing here is synthetic.
- Preprocessing is doing real work. The page recentres your drawing the way MNIST is centred, so very off-centre or tiny strokes can still be misread.
- The GPU is not needed for speed here, the network is tiny. Running it in WGSL is a deliberate choice, and a plain JavaScript copy of the same network runs alongside as a check, with no library in between.
- WebGPU is required for the live network view (current Chrome or Edge on desktop). Without it, a JavaScript fallback still classifies your drawing.

## Author

Built by **Dr. Safeer Ali Mirani**, GPU / XR / real-time visualisation engineer (PhD).

[safeer.ali.mirani@gmail.com](mailto:safeer.ali.mirani@gmail.com) · [Portfolio](https://safeeralimirani.netlify.app) · [GitHub](https://github.com/SafeerAliMirani) · [LinkedIn](https://www.linkedin.com/in/safeeralimirani)

## License

MIT. Data: the MNIST database of handwritten digits (Yann LeCun, Corinna Cortes, Christopher Burges).
