// Diverging colour map for vorticity: cobalt (clockwise) - paper white - carmine
// (anticlockwise). The same stops drive the WebGL shader and the CSS legend.

export const STOPS: [number, [number, number, number]][] = [
  [0.0, [0x1d, 0x34, 0x77]],
  [0.25, [0x2b, 0x4f, 0xa6]],
  [0.5, [0xf7, 0xf8, 0xf6]],
  [0.75, [0xb3, 0x26, 0x3a]],
  [1.0, [0x6e, 0x10, 0x22]],
];

/** RGB in [0, 255] for t in [0, 1]. */
export function colorAt(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (x <= t1) {
      const f = (x - t0) / (t1 - t0);
      return [0, 1, 2].map((k) => Math.round(c0[k] + f * (c1[k] - c0[k]))) as [number, number, number];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export const cssGradient = (): string =>
  `linear-gradient(90deg, ${STOPS.map(([t, c]) => `rgb(${c.join(",")}) ${t * 100}%`).join(", ")})`;

const vec3 = (c: [number, number, number]) => `vec3(${c.map((v) => (v / 255).toFixed(4)).join(", ")})`;

/** GLSL function mapping t in [0, 1] to the same colours. */
export const glslColormap = (): string => {
  let src = "vec3 colormap(float t) {\n  t = clamp(t, 0.0, 1.0);\n";
  for (let i = 1; i < STOPS.length; i++) {
    const [t0, c0] = STOPS[i - 1];
    const [t1, c1] = STOPS[i];
    src += `  if (t <= ${t1.toFixed(4)}) return mix(${vec3(c0)}, ${vec3(c1)}, (t - ${t0.toFixed(4)}) / ${(t1 - t0).toFixed(4)});\n`;
  }
  return src + `  return ${vec3(STOPS[STOPS.length - 1][1])};\n}\n`;
};

/** Symmetric colour scale: a high percentile of |values| so a few extremes don't wash it out. */
export function robustScale(values: Float32Array, q = 0.995): number {
  const abs = Float32Array.from(values, (v) => (Number.isFinite(v) ? Math.abs(v) : 0)).sort();
  const v = abs[Math.min(abs.length - 1, Math.floor(q * abs.length))];
  return v > 1e-9 ? v : 1;
}
