import { useEffect, useRef } from "react";
import { colorAt, glslColormap } from "./colormap";

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_field;
uniform float u_scale;
out vec4 color;
${glslColormap()}
void main() {
  float v = texture(u_field, v_uv).r;
  color = vec4(colormap(0.5 + 0.5 * v / u_scale), 1.0);
}`;

interface GLState {
  gl: WebGL2RenderingContext;
  tex: WebGLTexture;
  scaleLoc: WebGLUniformLocation | null;
}

function initGL(canvas: HTMLCanvasElement): GLState | null {
  const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // half-float textures are filterable in WebGL2: smooth interpolation between grid points
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); // the domain is periodic
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  return { gl, tex, scaleLoc: gl.getUniformLocation(prog, "u_scale") };
}

/** Canvas 2D fallback when WebGL2 is unavailable (nearest-neighbour pixels). */
function draw2D(canvas: HTMLCanvasElement, data: Float32Array, n: number, scale: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(n, n);
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const [r, g, b] = colorAt(0.5 + (0.5 * data[iy * n + ix]) / scale);
      const o = 4 * ((n - 1 - iy) * n + ix); // row 0 is the bottom of the domain
      img.data.set([r, g, b, 255], o);
    }
  }
  const off = new OffscreenCanvas(n, n);
  off.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

export interface BrushEvent {
  x: number; // [0, 1) left to right
  y: number; // [0, 1) bottom to top
  invert: boolean;
}

interface Props {
  data: Float32Array | null;
  n: number;
  scale: number;
  label: string;
  onBrush?: (e: BrushEvent) => void;
}

export function FieldView({ data, n, scale, label, onBrush }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<GLState | null | undefined>(undefined);
  const lastBrush = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const size = Math.max(1, Math.round(canvas.clientWidth * dpr));
      if (canvas.width !== size) {
        canvas.width = size;
        canvas.height = size;
      }
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current!;
    if (!data) return;
    if (glRef.current === undefined) {
      try {
        glRef.current = initGL(canvas);
      } catch {
        glRef.current = null;
      }
    }
    const st = glRef.current;
    if (!st) {
      draw2D(canvas, data, n, scale);
      return;
    }
    const { gl, tex, scaleLoc } = st;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, n, n, 0, gl.RED, gl.FLOAT, data);
    gl.uniform1f(scaleLoc, scale);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [data, n, scale]);

  const brush = (e: React.PointerEvent<HTMLCanvasElement>, force = false) => {
    if (!onBrush || (!force && e.buttons === 0)) return;
    const now = performance.now();
    if (!force && now - lastBrush.current < 45) return;
    lastBrush.current = now;
    const r = e.currentTarget.getBoundingClientRect();
    onBrush({
      x: (e.clientX - r.left) / r.width,
      y: 1 - (e.clientY - r.top) / r.height,
      invert: e.shiftKey || e.button === 2 || (e.buttons & 2) === 2,
    });
  };

  return (
    <canvas
      ref={canvasRef}
      className="field"
      role="img"
      aria-label={label}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        brush(e, true);
      }}
      onPointerMove={(e) => brush(e)}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
