// Pure functions shared by the browser page and the Node test (no DOM, no onnxruntime here).

/** RGBA pixels (already resized to size x size) -> Float32Array CHW normalised like torchvision. */
export function preprocess(rgba, size, mean, std) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * n + i] = (rgba[4 * i + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

/** Min-max normalise a class-activation map to [0, 1]. */
export function normalise(cam) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of cam) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo > 1e-12 ? hi - lo : 1;
  return Float32Array.from(cam, (v) => (v - lo) / span);
}

/** Matplotlib-like "jet" colormap, t in [0, 1] -> [r, g, b] in 0..255. */
export function jet(t) {
  const f = (x) => Math.round(255 * Math.min(1, Math.max(0, x)));
  return [f(1.5 - Math.abs(4 * t - 3)), f(1.5 - Math.abs(4 * t - 2)), f(1.5 - Math.abs(4 * t - 1))];
}

/** Normalised h x w map -> RGBA heat image (alpha grows with the value). */
export function camToRGBA(cam01, maxAlpha = 0.6) {
  const out = new Uint8ClampedArray(4 * cam01.length);
  cam01.forEach((v, i) => {
    const [r, g, b] = jet(v);
    out.set([r, g, b, Math.round(255 * maxAlpha * (0.25 + 0.75 * v))], 4 * i);
  });
  return out;
}

export function decision(prob, threshold) {
  return prob >= threshold ? "Pneumonie suspectée" : "Pas de pneumonie détectée";
}
