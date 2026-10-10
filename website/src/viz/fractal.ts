// Fault activation across a quasi-static loading history, as a scrubbable
// sequence of equilibria.
//
// THIS IS NOT A TIME SERIES. The slider is the LOAD PARAMETER. Every frame is
// an equilibrium in which every element is locked with its shear below its
// static strength; moving the slider moves along the load axis. Nothing
// propagates and nothing has a velocity, so the controls say "load" and never
// "time", and the play button is a convenience for scrubbing rather than a
// claim about rupture.
//
// Geometry is NON-INDEXED on purpose, as in fault.ts: each triangle needs its
// own three vertices to carry a flat colour. Unlike fault.ts there is no
// picking here, so the only thing the non-indexed layout buys is the flat
// colour -- but that is the whole picture.
//
// TWO THINGS THAT WOULD QUIETLY RUIN THE ANIMATION, both avoided here.
//
// 1. A PER-FRAME COLOUR SCALE. fault.ts recomputes its maximum on every
//    select() because each of its arrays is a different physical quantity.
//    Doing that per frame would rescale the colours as the network activates,
//    so the growth being animated would be normalised away and every frame
//    would look equally active. The scale here is GLOBAL: the payload encodes
//    the whole (frames, elements) array in one pass, so the byte value is
//    already on one common log ramp and the lookup table can be indexed
//    directly with no per-frame arithmetic at all.
// 2. INDEXING. fault.ts reads a square matrix source-major, `source * n + i`.
//    A frame array is (frames, elements), so the offset is `frame * n + i`.
//    The two look identical and mean different things.
//
// Byte 0 means "has never slipped" and is drawn in grey rather than at the
// bottom of the colour ramp, because a locked element is categorically
// different from one that slipped a little -- and a sequential colour map's
// dark end would read as "barely moved" instead of "did not move".

import {
  BufferGeometry, Color, DoubleSide, Float32BufferAttribute, LineBasicMaterial,
  LineSegments, Mesh, MeshBasicMaterial, PerspectiveCamera, Scene, Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { lut, cssGradient, type MapName } from "./colormaps";

const MAP: MapName = "plasma";
const LOCKED = [0.84, 0.84, 0.86];      // grey: never slipped
const FPS = 12;                          // frames per second when playing

export type Mode = "slip" | "incr" | "activation";

interface ArrayMeta {
  file: string; bytes: number; frames: number; n: number;
  log_min: number; log_max: number; min: number; max: number;
  measured_rel_err: number; bound_rel_err: number;
}

export interface FractalPayload {
  schema: number;
  config: string;
  unit: string;
  n_elements: number;
  n_faults: number;
  frames: number;
  lam: number[];
  geometry: { file: string; shape: [number, number, number] };
  arrays: Record<string, ArrayMeta>;
  frame_stats: {
    lam: number; events: number; events_cumulative: number;
    moment_cumulative_Nm: number; n_ever: number; slipped_this_frame: number;
  }[];
  d_geom: { known: number; calibrated: number; window_km: number[] } | null;
  network: Record<string, any>;
  friction: Record<string, any>;
}

function centre(pos: Float32Array): Vector3 {
  let x = 0, y = 0, z = 0;
  const n = pos.length / 3;
  for (let i = 0; i < pos.length; i += 3) { x += pos[i]; y += pos[i + 1]; z += pos[i + 2]; }
  return new Vector3(x / n, y / n, z / n);
}

function freeSurface(half: number): LineSegments {
  const v: number[] = [];
  const step = (2 * half) / 10;
  for (let i = 0; i <= 10; i++) {
    const t = -half + i * step;
    v.push(-half, t, 0, half, t, 0, t, -half, 0, t, half, 0);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(v, 3));
  return new LineSegments(g, new LineBasicMaterial({
    color: 0xbbbbbb, transparent: true, opacity: 0.6 }));
}

export class FractalScene {
  private el: HTMLElement;
  private scene = new Scene();
  private camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private colours: Float32BufferAttribute;
  private table = lut(MAP);
  private n: number;
  private frames: number;
  private data: Record<string, Uint8Array> = {};
  private activation: Uint8Array;        // first frame with slip, 255 = never
  private mode: Mode = "incr";
  private frame = 0;
  private playing = false;
  private raf = 0;
  private last = 0;

  constructor(el: HTMLElement, p: FractalPayload, pos: Float32Array) {
    this.el = el;
    this.n = p.n_elements;
    this.frames = p.frames;
    this.activation = new Uint8Array(this.n).fill(255);

    this.scene.background = new Color(0xffffff);
    this.camera = new PerspectiveCamera(36, 1, 0.5, 5e3);
    this.camera.up.set(0, 0, 1);         // the model's vertical is +z
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);

    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(pos, 3));
    this.colours = new Float32BufferAttribute(new Float32Array(pos.length), 3);
    geo.setAttribute("color", this.colours);
    this.scene.add(new Mesh(geo, new MeshBasicMaterial({
      vertexColors: true, side: DoubleSide })));

    const half = Math.max(
      ...Array.from({ length: pos.length / 3 }, (_, i) =>
        Math.max(Math.abs(pos[i * 3]), Math.abs(pos[i * 3 + 1]))));
    this.scene.add(freeSurface(half * 1.05));

    const c = centre(pos);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(c);
    this.camera.position.set(c.x + half * 1.9, c.y - half * 2.2, c.z + half * 1.5);
    this.controls.update();              // orient now, or frame one draws blank

    this.resize();
    addEventListener("resize", () => this.resize());
    this.controls.addEventListener("change", () => this.render());
  }

  setArray(name: string, bytes: Uint8Array) {
    this.data[name] = bytes;
    if (name === "incr") {
      // Activation load: the first frame in which an element slipped at all.
      // Derived here rather than shipped, because it is one pass over an array
      // the page already has.
      for (let f = 0; f < this.frames; f++) {
        const off = f * this.n;
        for (let i = 0; i < this.n; i++) {
          if (this.activation[i] === 255 && bytes[off + i] > 0) {
            this.activation[i] = Math.min(254, f);
          }
        }
      }
    }
  }

  setMode(m: Mode) { this.mode = m; this.paint(); }
  setFrame(f: number) {
    this.frame = Math.max(0, Math.min(this.frames - 1, f | 0));
    this.paint();
  }
  get currentFrame() { return this.frame; }

  /** Per-face colour for the current frame. One LUT lookup per element. */
  paint() {
    const col = this.colours.array as Float32Array;
    const t = this.table;
    const n = this.n;
    const put = (i: number, r: number, g: number, b: number) => {
      const o = i * 9;
      for (let k = 0; k < 3; k++) {
        col[o + k * 3] = r; col[o + k * 3 + 1] = g; col[o + k * 3 + 2] = b;
      }
    };
    if (this.mode === "activation") {
      const span = Math.max(1, this.frames - 1);
      for (let i = 0; i < n; i++) {
        const a = this.activation[i];
        if (a === 255 || a > this.frame) { put(i, LOCKED[0], LOCKED[1], LOCKED[2]); continue; }
        const q = 1 + Math.round((a / span) * 254);
        put(i, t[q * 4] / 255, t[q * 4 + 1] / 255, t[q * 4 + 2] / 255);
      }
    } else {
      const arr = this.data[this.mode];
      if (!arr) return;
      const off = this.frame * n;          // (frames, elements), NOT source-major
      for (let i = 0; i < n; i++) {
        const q = arr[off + i];
        if (q === 0) { put(i, LOCKED[0], LOCKED[1], LOCKED[2]); continue; }
        put(i, t[q * 4] / 255, t[q * 4 + 1] / 255, t[q * 4 + 2] / 255);
      }
    }
    this.colours.needsUpdate = true;
    this.render();
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this.last = performance.now();
    const step = (now: number) => {
      if (!this.playing) return;
      if (now - this.last >= 1000 / FPS) {
        this.last = now;
        this.setFrame((this.frame + 1) % this.frames);
        this.onTick?.(this.frame);
      }
      this.controls.update();
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  pause() { this.playing = false; cancelAnimationFrame(this.raf); }
  get isPlaying() { return this.playing; }
  onTick?: (f: number) => void;

  resize() {
    const w = this.el.clientWidth || 800;
    const h = Math.max(420, Math.round(w * 0.62));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() { this.renderer.render(this.scene, this.camera); }
}

const MODE_LABEL: Record<Mode, string> = {
  incr: "slip in this increment",
  slip: "cumulative slip",
  activation: "load at first slip",
};

export async function startFractal(root: HTMLElement, baseUrl: string) {
  const base = baseUrl.replace(/\/?$/, "/") + "data/";
  const host = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;
  const bar = root.querySelector<HTMLElement>("[data-bar]")!;

  let p: FractalPayload;
  try {
    const r = await fetch(base + "fractal.json");
    if (!r.ok) throw new Error(`fractal.json: ${r.status}`);
    p = await r.json();
  } catch (e) {
    status.textContent = `could not load the loading history (${e}).`;
    return;
  }

  const geomBuf = await (await fetch(base + p.geometry.file)).arrayBuffer();
  const pos = new Float32Array(geomBuf);
  if (pos.length !== p.n_elements * 9) {
    status.textContent = `geometry is ${pos.length} floats for `
      + `${p.n_elements} elements; expected ${p.n_elements * 9}.`;
    return;
  }

  const scene = new FractalScene(host, p, pos);
  for (const name of ["incr", "slip"]) {
    const meta = p.arrays[name];
    if (!meta) continue;
    const buf = await (await fetch(base + meta.file)).arrayBuffer();
    const bytes = new Uint8Array(buf);
    if (bytes.length !== p.frames * p.n_elements) {
      status.textContent = `${name} is ${bytes.length} bytes for `
        + `${p.frames} x ${p.n_elements}.`;
      return;
    }
    scene.setArray(name, bytes);
  }

  // Controls. Built here rather than in the page so the markup stays a few
  // empty hooks and the wiring lives next to the thing it drives.
  const slot = (key: string) => root.querySelector<HTMLElement>(`[data-ctl="${key}"]`)!;

  const sel = document.createElement("select");
  for (const m of ["incr", "slip", "activation"] as Mode[]) {
    const o = document.createElement("option");
    o.value = m; o.textContent = MODE_LABEL[m];
    sel.appendChild(o);
  }
  sel.value = "incr";
  slot("mode").appendChild(sel);

  const play = document.createElement("button");
  play.textContent = "play";
  slot("play").appendChild(play);

  const rng = document.createElement("input");
  rng.type = "range"; rng.min = "0"; rng.max = String(p.frames - 1);
  rng.step = "1"; rng.value = "0";
  const val = document.createElement("span");
  val.className = "val";
  slot("frame").appendChild(rng);
  slot("frame").appendChild(val);

  const strip = document.createElement("div");
  strip.className = "strip";
  strip.style.background = cssGradient(MAP);
  const ends = document.createElement("div");
  ends.className = "ends";
  bar.appendChild(strip);
  bar.appendChild(ends);

  const fmt = (v: number) => (v >= 1e4 || (v !== 0 && v < 1e-2))
    ? v.toExponential(2) : v.toFixed(3);

  const describe = (f: number) => {
    const st = p.frame_stats[f];
    rng.value = String(f);
    val.textContent = `λ = ${st.lam.toFixed(4)}`;
    const meta = p.arrays[scene_mode()];
    const unit = scene_mode() === "activation" ? "" : " m of slip";
    const lo = meta ? meta.min * 1e3 : 0;
    const hi = meta ? meta.max * 1e3 : 0;
    ends.innerHTML = scene_mode() === "activation"
      ? `<span>first to slip</span><span class="mid">load at first slip</span>`
        + `<span>last to slip</span>`
      : `<span>${fmt(lo)}${unit}</span>`
        + `<span class="mid">log scale, grey = never slipped</span>`
        + `<span>${fmt(hi)}${unit}</span>`;
    status.textContent =
      `frame ${f + 1} of ${p.frames} · ${st.n_ever} of ${p.n_elements} elements`
      + ` have slipped (${(100 * st.n_ever / p.n_elements).toFixed(1)} %)`
      + ` · ${st.events_cumulative} events · ${st.slipped_this_frame} slipped in`
      + ` this increment · cumulative M₀ ${st.moment_cumulative_Nm.toExponential(2)} N m`;
  };
  const scene_mode = () => sel.value as Mode;

  sel.addEventListener("change", () => {
    scene.setMode(scene_mode());
    describe(scene.currentFrame);
  });
  rng.addEventListener("input", () => {
    scene.pause(); play.textContent = "play";
    scene.setFrame(Number(rng.value));
    describe(scene.currentFrame);
  });
  play.addEventListener("click", () => {
    if (scene.isPlaying) { scene.pause(); play.textContent = "play"; }
    else { scene.play(); play.textContent = "pause"; }
  });
  scene.onTick = (f) => describe(f);

  scene.setMode("incr");
  scene.setFrame(0);
  describe(0);
}
