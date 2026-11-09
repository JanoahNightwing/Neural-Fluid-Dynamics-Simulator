import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

const MUTED = "#6b7a88";
const GRID = "#dde4ea";
export const COLORS = { reference: "#1c2b3a", network: "#2b4fa6", error: "#b3263a", warn: "#a87410" };

function axis(label: string, extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
  return {
    label,
    stroke: MUTED,
    labelFont: "500 12px 'Barlow Semi Condensed', sans-serif",
    font: "12px 'Barlow Semi Condensed', sans-serif",
    grid: { stroke: GRID, width: 1 },
    ticks: { stroke: GRID, width: 1 },
    ...extra,
  };
}

interface ChartProps {
  title: string;
  data: uPlot.AlignedData;
  series: uPlot.Series[];
  axes: uPlot.Axis[];
  scales?: uPlot.Scales;
  empty: string;
}

function Chart({ title, data, series, axes, scales, empty }: ChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    const el = host.current!;
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height: 170,
      series: [{}, ...series],
      axes,
      scales: scales ?? { x: { time: false } },
      legend: { show: true, live: false },
      cursor: { drag: { x: false, y: false } },
    };
    plot.current = new uPlot(opts, data, el);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: 170 }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
    // the chart is rebuilt only when its structure changes; data updates use setData
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.length]);

  useEffect(() => {
    plot.current?.setData(data);
  }, [data]);

  const hasData = data[0].length > 0;
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <div ref={host} className="chart-host" style={{ opacity: hasData ? 1 : 0.35 }} />
      {!hasData && <p className="chart-empty">{empty}</p>}
    </figure>
  );
}

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const superscriptPower = (v: number) => `10${String(Math.round(Math.log10(v))).replace(/[-0-9]/g, (c) => SUP[c])}`;

export interface History {
  t: number[];
  correlation: (number | null)[];
  relL2: (number | null)[];
  energyRef: number[];
  energyModel: (number | null)[];
}

export function AgreementChart({ h }: { h: History }) {
  return (
    <Chart
      title="Agreement with the solver"
      empty="Press Play to start the comparison."
      data={[h.t, h.correlation, h.relL2]}
      series={[
        { label: "correlation", stroke: COLORS.network, width: 2, scale: "c" },
        { label: "relative error", stroke: COLORS.error, width: 1.5, dash: [5, 4], scale: "e" },
      ]}
      scales={{ x: { time: false }, c: { range: [-0.1, 1.02] }, e: { range: [0, 1.6] } }}
      axes={[axis("time"), axis("correlation", { scale: "c" }), axis("relative error", { scale: "e", side: 1, grid: { show: false } })]}
    />
  );
}

export function EnergyChart({ h }: { h: History }) {
  return (
    <Chart
      title="Kinetic energy"
      empty="Energy appears once the flow is running."
      data={[h.t, h.energyRef, h.energyModel]}
      series={[
        { label: "solver", stroke: COLORS.reference, width: 2 },
        { label: "network", stroke: COLORS.network, width: 2 },
      ]}
      axes={[axis("time"), axis("energy")]}
    />
  );
}

export function SpectrumChart({ spectra }: { spectra: { ref: number[]; model: number[] } | null }) {
  const k = spectra ? spectra.ref.map((_, i) => i).slice(1) : [];
  const kMax = k.length ? k[k.length - 1] : 32;
  const clip = (a: number[]) => a.slice(1).map((v) => (v > 1e-14 && Number.isFinite(v) ? v : null));
  return (
    <Chart
      title="Energy by length scale"
      empty="The spectrum appears with the first frame."
      data={[k, spectra ? clip(spectra.ref) : [], spectra ? clip(spectra.model) : []]}
      series={[
        { label: "solver", stroke: COLORS.reference, width: 2 },
        { label: "network", stroke: COLORS.network, width: 2 },
      ]}
      scales={{ x: { time: false, distr: 3, range: [1, kMax] }, y: { distr: 3 } }}
      axes={[
        axis("wavenumber k (small k = large eddies)"),
        axis("E(k)", {
          // at most ~5 evenly spaced decades, labelled 10⁻³ rather than 0.001
          splits: (u) => {
            const lo = Math.floor(Math.log10(u.scales.y.min ?? 1e-8));
            const hi = Math.ceil(Math.log10(u.scales.y.max ?? 1));
            const step = Math.max(1, Math.ceil((hi - lo) / 5));
            const out: number[] = [];
            for (let e = hi; e >= lo; e -= step) out.unshift(10 ** e);
            return out;
          },
          values: (_u, vals) => vals.map((v) => (v == null ? "" : superscriptPower(v))),
        }),
      ]}
    />
  );
}
