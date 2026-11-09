// Messages exchanged with the Python server (see src/nfds/app/server.py).

export interface ModelInfo {
  id: string;
  label: string;
  kind: "model" | "baseline";
  description: string;
  resolution: number;
  n_in: number;
  nu_min: number;
  nu_max: number;
  physics: { drag: number; forcing_wavenumber: number; save_dt: number };
  physics_source: string;
  extra: Record<string, unknown>;
}

export interface Hello {
  type: "hello";
  models: ModelInfo[];
  presets: Record<string, string>;
  ref_n: number;
  nu_limits: [number, number];
  ref_resolution_warning: number;
}

export interface InitialCondition {
  kind: string;
  seed: number;
  k_peak: number;
}

export interface SessionState {
  type: "state";
  playing: boolean;
  fps: number;
  model: string | null;
  nu: number | null;
  forcing: boolean | null;
  ic: InitialCondition;
  t: number;
}

export interface Metrics {
  t: number;
  rel_l2: number | null;
  correlation: number | null;
  energy_ref: number;
  energy_model: number | null;
  diverged: boolean;
  ref_resolution_ratio: number;
}

export interface Frame {
  n: number;
  step: number;
  metrics: Metrics;
  spectra: { ref: number[]; model: number[] } | null;
  reference: Float32Array;
  prediction: Float32Array;
}

export type ClientMessage =
  | { type: "reset"; model?: string; ic?: Partial<InitialCondition>; nu?: number; forcing?: boolean }
  | { type: "play" }
  | { type: "pause" }
  | { type: "step" }
  | { type: "resync" }
  | { type: "set_viscosity"; nu: number }
  | { type: "set_forcing"; on: boolean }
  | { type: "set_model"; model: string }
  | { type: "set_fps"; fps: number }
  | { type: "brush"; x: number; y: number; radius: number; strength: number };

/** IEEE 754 half precision -> number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

const HALF_TABLE = (() => {
  const t = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) t[i] = halfToFloat(i);
  return t;
})();

/** Decode a binary frame: uint32 header length, JSON header, then two float16 n*n arrays. */
export function decodeFrame(buf: ArrayBuffer): Frame {
  const view = new DataView(buf);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLength)));
  const n: number = header.n;
  const offset = 4 + headerLength;
  // the payload may not be 2-byte aligned within the buffer, so copy it
  const halves = new Uint16Array(buf.slice(offset, offset + 4 * n * n));
  if (halves.length !== 2 * n * n) throw new Error(`frame payload has ${halves.length} values, expected ${2 * n * n}`);
  const reference = new Float32Array(n * n);
  const prediction = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    reference[i] = HALF_TABLE[halves[i]];
    prediction[i] = HALF_TABLE[halves[n * n + i]];
  }
  return { n, step: header.step, metrics: header.metrics, spectra: header.spectra, reference, prediction };
}

export const reynolds = (nu: number) => 1 / nu;
