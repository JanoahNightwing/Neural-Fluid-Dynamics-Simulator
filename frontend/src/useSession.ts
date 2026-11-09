import { useCallback, useEffect, useRef, useState } from "react";
import type { History } from "./Charts";
import { decodeFrame, type ClientMessage, type Frame, type Hello, type SessionState } from "./protocol";

export type Connection = "connecting" | "open" | "closed";

const MAX_POINTS = 4000;
const emptyHistory = (): History => ({ t: [], correlation: [], relL2: [], energyRef: [], energyModel: [] });

export function useSession() {
  const ws = useRef<WebSocket | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [hello, setHello] = useState<Hello | null>(null);
  const [state, setState] = useState<SessionState | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [spectra, setSpectra] = useState<Frame["spectra"]>(null);
  const [history, setHistory] = useState<History>(emptyHistory);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const frameTimes = useRef<number[]>([]);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    setConnection("connecting");
    sock.onopen = () => setConnection("open");
    sock.onclose = () => setConnection("closed");
    sock.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "hello") setHello(msg);
        else if (msg.type === "state") setState(msg);
        else if (msg.type === "error") setError(msg.message);
        return;
      }
      const f = decodeFrame(ev.data as ArrayBuffer);
      setFrame(f);
      if (f.spectra) setSpectra(f.spectra);
      const now = performance.now();
      frameTimes.current = [...frameTimes.current.filter((t) => now - t < 2000), now];
      setFps(frameTimes.current.length / 2);
      setHistory((h) => {
        const m = f.metrics;
        const last = h.t.length ? h.t[h.t.length - 1] : -Infinity;
        // Time going backwards means a new flow: start a fresh history. A frame at the same
        // time (sent after a settings change or a brush stroke) replaces the last point.
        let base = h;
        if (m.t < last) base = emptyHistory();
        else if (m.t === last) {
          base = {
            t: h.t.slice(0, -1), correlation: h.correlation.slice(0, -1), relL2: h.relL2.slice(0, -1),
            energyRef: h.energyRef.slice(0, -1), energyModel: h.energyModel.slice(0, -1),
          };
        }
        const cap = <T,>(a: T[], v: T) => [...a.slice(-(MAX_POINTS - 1)), v];
        return {
          t: cap(base.t, m.t),
          correlation: cap(base.correlation, m.correlation),
          relL2: cap(base.relL2, m.rel_l2),
          energyRef: cap(base.energyRef, m.energy_ref),
          energyModel: cap(base.energyModel, m.diverged ? null : m.energy_model),
        };
      });
    };
    return () => sock.close();
  }, [attempt]);

  const send = useCallback((msg: ClientMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(msg));
  }, []);

  const reconnect = useCallback(() => setAttempt((a) => a + 1), []);
  const clearError = useCallback(() => setError(null), []);

  return { connection, hello, state, frame, spectra, history, error, clearError, fps, send, reconnect };
}
