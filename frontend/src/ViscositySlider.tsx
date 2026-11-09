// The one bespoke control: a Reynolds-number slider on a log scale whose track shows the
// band of Reynolds numbers the selected network was trained on.

interface Props {
  nu: number;
  limits: [number, number]; // viscosity limits accepted by the server
  trained: [number, number]; // viscosity range of the training data
  onChange: (nu: number) => void;
  onCommit: (nu: number) => void;
}

const STEPS = 1000;

export function formatNu(nu: number): string {
  const e = Math.floor(Math.log10(nu));
  const m = nu / 10 ** e;
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]);
  return `${m.toFixed(1)} × 10${sup}`;
}

export function ViscositySlider({ nu, limits, trained, onChange, onCommit }: Props) {
  // position 0 = lowest Reynolds number (most viscous), 1 = highest
  const lo = Math.log(1 / limits[1]);
  const hi = Math.log(1 / limits[0]);
  const pos = (v: number) => (Math.log(1 / v) - lo) / (hi - lo);
  const fromStep = (s: number) => 1 / Math.exp(lo + (s / STEPS) * (hi - lo));
  const bandStart = pos(trained[1]);
  const bandEnd = pos(trained[0]);
  const inside = nu >= trained[0] * 0.999 && nu <= trained[1] * 1.001;
  const re = Math.round(1 / nu);

  return (
    <div className="visc">
      <div className="visc-readout">
        <span className="visc-re">Re {re.toLocaleString()}</span>
        <span className="visc-nu">ν = {formatNu(nu)}</span>
      </div>
      <div className="visc-track-wrap">
        <div
          className="visc-band"
          style={{ left: `${bandStart * 100}%`, width: `${(bandEnd - bandStart) * 100}%` }}
          aria-hidden="true"
        />
        <input
          type="range"
          min={0}
          max={STEPS}
          value={Math.round(pos(nu) * STEPS)}
          aria-label="Reynolds number"
          aria-valuetext={`Reynolds number ${re}`}
          onChange={(e) => onChange(fromStep(+e.target.value))}
          onPointerUp={(e) => onCommit(fromStep(+(e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onCommit(fromStep(+(e.target as HTMLInputElement).value))}
        />
      </div>
      <div className="visc-scale" aria-hidden="true">
        <span>smoother</span>
        <span>more turbulent</span>
      </div>
      <p className={inside ? "visc-note" : "visc-note visc-note-out"} role="status">
        {inside
          ? `Inside the training range (Re ${Math.round(1 / trained[1])}–${Math.round(1 / trained[0])}).`
          : `Outside the training range (Re ${Math.round(1 / trained[1])}–${Math.round(1 / trained[0])}): the network is extrapolating.`}
      </p>
    </div>
  );
}
