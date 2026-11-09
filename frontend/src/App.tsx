import { useEffect, useMemo, useRef, useState } from "react";
import { AgreementChart, EnergyChart, SpectrumChart } from "./Charts";
import { cssGradient, robustScale } from "./colormap";
import { FieldView, type BrushEvent } from "./FieldView";
import { useSession } from "./useSession";
import { ViscositySlider } from "./ViscositySlider";

const DEFAULT_NU = 3e-3;

export default function App() {
  const s = useSession();
  const { hello, state, frame, send } = s;
  const [modelId, setModelId] = useState<string>("");
  const [preset, setPreset] = useState("random");
  const [seed, setSeed] = useState(0);
  const [kPeak, setKPeak] = useState(4);
  const [nu, setNu] = useState(DEFAULT_NU);
  const [forcing, setForcing] = useState(true);
  const [speed, setSpeed] = useState(15);
  const [strength, setStrength] = useState(8);
  const [radius, setRadius] = useState(0.05);
  const scale = useRef(1);

  const models = hello?.models ?? [];
  const model = models.find((m) => m.id === modelId) ?? models[0];

  // first contact: pick the first trained network and start a flow
  useEffect(() => {
    if (hello && hello.models.length && !modelId) {
      const first = hello.models[0];
      const start = Math.sqrt(first.nu_min * first.nu_max); // middle of the training range
      setModelId(first.id);
      setNu(start);
      send({ type: "reset", model: first.id, ic: { kind: "random", seed: 0, k_peak: 4 }, nu: start, forcing: true });
    }
  }, [hello, modelId, send]);

  const playing = state?.playing ?? false;
  const startOver = (overrides: { kind?: string; seed?: number; k_peak?: number } = {}) =>
    send({ type: "reset", model: model?.id, ic: { kind: preset, seed, k_peak: kPeak, ...overrides }, nu, forcing });

  // keyboard: space = play/pause, s = step, r = restart the network from the solver
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName)) return;
      if (e.key === " ") {
        e.preventDefault();
        send({ type: playing ? "pause" : "play" });
      } else if (e.key === "s") send({ type: "step" });
      else if (e.key === "r") send({ type: "resync" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, send]);

  // one colour scale for all three panels, smoothed so it does not flicker
  const difference = useMemo(() => {
    if (!frame) return null;
    const d = new Float32Array(frame.n * frame.n);
    for (let i = 0; i < d.length; i++) d[i] = frame.prediction[i] - frame.reference[i];
    const target = robustScale(frame.reference);
    scale.current = frame.step === 0 ? target : 0.85 * scale.current + 0.15 * target;
    return d;
  }, [frame]);

  const onBrush = (b: BrushEvent) =>
    send({ type: "brush", x: b.x, y: b.y, radius, strength: b.invert ? -strength : strength });

  const m = frame?.metrics;
  const diverged = m?.diverged ?? false;
  const refUnresolved = !!(hello && m && m.ref_resolution_ratio > hello.ref_resolution_warning);
  const ready = s.connection === "open" && models.length > 0;

  return (
    <div className="app">
      <header className="top">
        <div className="title">
          <h1>Turbulence, two ways</h1>
          <p>
            A reference solver and a trained network start from the same flow. Paint vortices, change the
            viscosity, and watch where they part.
          </p>
        </div>
        <dl className="readout" aria-live="polite">
          <div>
            <dt>Time</dt>
            <dd>{(m?.t ?? 0).toFixed(2)}</dd>
          </div>
          <div>
            <dt>Correlation</dt>
            <dd className={m?.correlation != null && m.correlation < 0.8 ? "low" : ""}>
              {m?.correlation == null ? "–" : m.correlation.toFixed(3)}
            </dd>
          </div>
          <div>
            <dt>Frames per second</dt>
            <dd>{playing ? s.fps.toFixed(0) : "–"}</dd>
          </div>
        </dl>
      </header>

      {s.connection !== "open" && (
        <div className="banner" role="alert">
          {s.connection === "connecting" ? (
            "Connecting to the simulation server…"
          ) : (
            <>
              Lost the connection to the simulation server. Check that <code>python -m nfds.app</code> is running.
              <button onClick={s.reconnect}>Reconnect</button>
            </>
          )}
        </div>
      )}
      {s.error && (
        <div className="banner banner-error" role="alert">
          {s.error}
          <button onClick={s.clearError}>Dismiss</button>
        </div>
      )}

      <aside className="rail" aria-label="Controls">
        <section>
          <h2>Starting flow</h2>
          <label>
            Pattern
            <select value={preset} onChange={(e) => { setPreset(e.target.value); startOver({ kind: e.target.value }); }} disabled={!ready}>
              {Object.entries(hello?.presets ?? {}).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
          </label>
          {preset === "random" && (
            <label>
              Eddy size <span className="value">{kPeak <= 3 ? "large" : kPeak >= 7 ? "small" : "medium"}</span>
              <input type="range" min={2} max={8} step={1} value={kPeak} onChange={(e) => setKPeak(+e.target.value)}
                onPointerUp={() => startOver()} onKeyUp={() => startOver()} disabled={!ready} />
            </label>
          )}
          <div className="row">
            <button onClick={() => startOver()} disabled={!ready}>Start over</button>
            <button onClick={() => { const n = seed + 1; setSeed(n); startOver({ seed: n }); }} disabled={!ready}>New random flow</button>
          </div>
          <label className="check">
            <input type="checkbox" checked={forcing} disabled={!ready}
              onChange={(e) => { setForcing(e.target.checked); send({ type: "set_forcing", on: e.target.checked }); }} />
            Keep stirring the flow (forcing)
          </label>
        </section>

        <section>
          <h2>Viscosity</h2>
          {model && hello && (
            <ViscositySlider
              nu={nu}
              limits={hello.nu_limits}
              trained={[model.nu_min, model.nu_max]}
              onChange={setNu}
              onCommit={(v) => send({ type: "set_viscosity", nu: v })}
            />
          )}
        </section>

        <section>
          <h2>Network</h2>
          <label>
            Predictor
            <select value={model?.id ?? ""} disabled={!ready}
              onChange={(e) => { setModelId(e.target.value); send({ type: "set_model", model: e.target.value }); }}>
              <optgroup label="Trained networks">
                {models.filter((x) => x.kind === "model").map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
              </optgroup>
              <optgroup label="Baselines">
                {models.filter((x) => x.kind === "baseline").map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
              </optgroup>
            </select>
          </label>
          {model && <p className="hint">{model.description}. Grid {model.resolution}², reference solver {hello?.ref_n}².</p>}
        </section>

        <section>
          <h2>Playback</h2>
          <div className="row">
            <button className="primary" onClick={() => send({ type: playing ? "pause" : "play" })} disabled={!ready} aria-keyshortcuts="Space">
              {playing ? "Pause" : "Play"}
            </button>
            <button onClick={() => send({ type: "step" })} disabled={!ready || playing} aria-keyshortcuts="S">Step</button>
          </div>
          <label>
            Speed <span className="value">{speed} frames/s</span>
            <input type="range" min={1} max={60} value={speed} disabled={!ready}
              onChange={(e) => { setSpeed(+e.target.value); send({ type: "set_fps", fps: +e.target.value }); }} />
          </label>
          <button onClick={() => send({ type: "resync" })} disabled={!ready} aria-keyshortcuts="R">
            Restart the network from the solver
          </button>
        </section>

        <section>
          <h2>Paint vortices</h2>
          <p className="hint">Drag on any panel to add a spinning eddy. Hold Shift for the opposite spin.</p>
          <label>
            Strength <span className="value">{strength}</span>
            <input type="range" min={1} max={30} value={strength} onChange={(e) => setStrength(+e.target.value)} />
          </label>
          <label>
            Size <span className="value">{radius < 0.04 ? "small" : radius > 0.07 ? "large" : "medium"}</span>
            <input type="range" min={0.02} max={0.1} step={0.005} value={radius} onChange={(e) => setRadius(+e.target.value)} />
          </label>
        </section>
        <p className="keys">Keys: Space play or pause, S step, R restart the network.</p>
      </aside>

      <main className="stage">
        <div className="fields">
          <figure className="panel">
            <figcaption>Reference solver</figcaption>
            <FieldView data={frame?.reference ?? null} n={frame?.n ?? 64} scale={scale.current} label="Vorticity from the reference solver" onBrush={onBrush} />
            {refUnresolved && (
              <div className="panel-alert panel-alert-warn" role="status">
                At this Reynolds number the {hello?.ref_n}² reference grid is too coarse: its finest
                detail is unreliable. Lower the Reynolds number for a trustworthy comparison.
              </div>
            )}
          </figure>
          <figure className="panel">
            <figcaption>{model?.kind === "baseline" ? model.label : "Network prediction"}</figcaption>
            <FieldView data={frame?.prediction ?? null} n={frame?.n ?? 64} scale={scale.current} label="Vorticity predicted by the network" onBrush={onBrush} />
            {diverged && (
              <div className="panel-alert" role="alert">
                The prediction blew up.
                <button onClick={() => send({ type: "resync" })}>Restart it from the solver</button>
              </div>
            )}
          </figure>
          <figure className="panel">
            <figcaption>Difference (network − solver)</figcaption>
            <FieldView data={difference} n={frame?.n ?? 64} scale={scale.current} label="Difference between prediction and solver" onBrush={onBrush} />
          </figure>
        </div>
        <div className="legend" aria-hidden="true">
          <span>clockwise</span>
          <span className="legend-bar" style={{ background: cssGradient() }} />
          <span>anticlockwise</span>
          <span className="legend-scale">colour limit ±{scale.current.toFixed(1)}</span>
        </div>
        <div className="charts">
          <AgreementChart h={s.history} />
          <EnergyChart h={s.history} />
          <SpectrumChart spectra={s.spectra} />
        </div>
      </main>
    </div>
  );
}
