import { describe, expect, it } from "vitest";
import { colorAt, glslColormap, robustScale } from "./colormap";
import { decodeFrame, halfToFloat } from "./protocol";

// Encode a number as IEEE half precision (normal range only), for building test frames.
function floatToHalf(v: number): number {
  if (v === 0) return 0;
  const sign = v < 0 ? 0x8000 : 0;
  const a = Math.abs(v);
  const exp = Math.floor(Math.log2(a));
  const frac = Math.round((a / 2 ** exp - 1) * 1024);
  return sign | ((exp + 15) << 10) | frac;
}

describe("half precision", () => {
  it("decodes known values", () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 4);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
    expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 9);
  });
});

describe("decodeFrame", () => {
  it("splits header, reference and prediction", () => {
    const n = 2;
    const header = new TextEncoder().encode(
      JSON.stringify({ type: "frame", n, step: 7, metrics: { t: 1.5 }, spectra: null }),
    );
    // odd header length makes the payload unaligned, as it can be from the server
    const padded = header.length % 2 === 0 ? new Uint8Array([...header, 0x20]) : header;
    const values = [1, -2, 0.5, 4, 3, 2, 1, -0.25];
    const buf = new ArrayBuffer(4 + padded.length + 2 * values.length);
    const view = new DataView(buf);
    view.setUint32(0, padded.length, true);
    new Uint8Array(buf, 4, padded.length).set(padded);
    values.forEach((v, i) => view.setUint16(4 + padded.length + 2 * i, floatToHalf(v), true));
    const f = decodeFrame(buf);
    expect(f.n).toBe(2);
    expect(f.step).toBe(7);
    expect(Array.from(f.reference)).toEqual([1, -2, 0.5, 4]);
    expect(Array.from(f.prediction)).toEqual([3, 2, 1, -0.25]);
  });
});

describe("colour map", () => {
  it("is white at zero and saturated at the ends", () => {
    expect(colorAt(0.5)).toEqual([0xf7, 0xf8, 0xf6]);
    expect(colorAt(0)).toEqual([0x1d, 0x34, 0x77]);
    expect(colorAt(2)).toEqual([0x6e, 0x10, 0x22]);
    expect(glslColormap()).toContain("vec3 colormap(float t)");
  });
  it("uses a robust symmetric scale", () => {
    const v = new Float32Array(1000).map((_, i) => (i % 2 ? 1 : -1) * (i / 1000));
    v[0] = 1e6; // one extreme value does not set the scale
    expect(robustScale(v)).toBeLessThan(1.01);
  });
});
