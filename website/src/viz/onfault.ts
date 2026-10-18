// On-fault stress on a system of three surface-breaking rough faults: one
// scalar per element, three kernels, one window.
//
// COLOUR IS DATA HERE, so this scene is unlit and flat: no light, no relief
// shading, nothing that would modulate a Coulomb excess by where a lamp is.
// Every facet is painted with the one value sampled at its centroid, the
// collocation point -- the value IS the sample, nothing is interpolated across
// a facet, and the saturated area in the legend is a count of facets weighted
// by area. (The payload still describes each facet as k^2 cells; at k = 1 the
// cell is the facet, and the loader below is written for any k.)
//
// THE COLOUR IS LOOKED UP IN THE SHADER from a window and a 256-entry table,
// the way the slice viewer does it, and not written into a vertex-colour
// attribute: a vertex colour is interpolated as a COLOUR and is taken as
// linear, so the two viewers that do that render lighter than their own bars.
// Here the value rides as a float attribute, the window is a uniform, and the
// table's bytes are written straight to the fragment.
//
// ONE WINDOW FOR EVERY KERNEL, by construction. The whole point is that the
// maps are the same quantity at the same scale; an array windowed over its own
// range would hide any difference by rescaling.

import {
  Box3, BufferGeometry, DataTexture, DoubleSide, Float32BufferAttribute, GLSL3,
  Group, LinearFilter, LineBasicMaterial, LineLoop, LineSegments, Mesh,
  MeshBasicMaterial, PerspectiveCamera, RGBAFormat, Scene, ShaderMaterial,
  Vector2, Vector3, WebGLRenderer, Color,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { lut, type MapName } from "./colormaps";
import {
  buildFurniture, fitCamera, meshEdges, onDemand, presetDir, screenshot,
  type Preset,
} from "./stage";

export interface OnFaultManifest {
  schema: number; units: string; git: string | null;
  k: number; cells_per_facet: number; n_facets: number; n_cells: number;
  eps_km: number; h_km: number; nearest_edge_eps: number; edge_eps_step: number;
  lam_GPa: number; lam0_GPa: number; lam_over_lam0: number;
  friction: { mu: number; sigma_eff_GPa: number; regime: string;
              mean_stress_ratio: number; coupled: boolean };
  faults: { name: string; a_km: number; dip_deg: number; trace_km: number[];
            trace_length_km: number; attach: string | null; n: number;
            remote_shear_MPa: number; remote_sigma_n_MPa: number }[];
  system: { clearance_km: number; contact: Record<string, { host: string; min_distance_km: number }> };
  window: Record<string, number | string>;
  stats: Record<string, Record<string, { saturated_area_fraction: number;
                                         nonfinite: number }>>;
  arrays: Record<string, { dtype: string; shape: number[]; bytes: number }>;
  defaults: { kernel: string; metric: string };
}

interface FaultGeom {
  name: string; vertices: number[][]; triangles: number[][]; outline: number[][];
}

const VERT = `
attribute float value;
varying float vValue;
void main() {
  vValue = value;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// Own output declared, for the reason scene.ts gives: with glslVersion GLSL3
// three does not inject gl_FragColor.
const FRAG = `
precision highp float;
layout(location = 0) out vec4 fragColor;
varying float vValue;
uniform sampler2D uLut;
uniform vec2  uWin;        // lo, hi in GPa
uniform bool  uMark;       // paint out-of-window cells flat rather than saturated
void main() {
  if (isnan(vValue)) { fragColor = vec4(1.0, 0.0, 1.0, 1.0); return; }
  float t = (vValue - uWin.x) / max(uWin.y - uWin.x, 1e-30);
  if (uMark && (t < 0.0 || t > 1.0)) { fragColor = vec4(0.13, 0.13, 0.14, 1.0); return; }
  fragColor = vec4(texture(uLut, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb, 1.0);
}`;

/** The metrics as the manifest names them, with what the page calls them. */
export const METRICS: Record<string, { label: string; signed: boolean; map: MapName }> = {
  S: { label: "Coulomb excess τₛ − μσₙ′", signed: true, map: "rdbu_r" },
  dsn: { label: "normal-stress change Δσₙ (compression +)", signed: true, map: "rdbu_r" },
  dts: { label: "shear change Δτₛ along the slip direction", signed: true, map: "rdbu_r" },
  slip: { label: "slip, per element", signed: false, map: "viridis" },
};

export const KERNELS: Record<string, string> = {
  mollified: "mollified P0 (mhs)", classical: "classical TDE (cutde)",
  continuous: "mollified, continuous P2 (mhs)",
};

export class OnFaultScene {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private content = new Group();
  private furniture = new Group();
  private wire = new Group();
  private mat: ShaderMaterial;
  private mesh: Mesh | null = null;
  private valueAttr: Float32BufferAttribute | null = null;
  private el: HTMLElement;
  private stop: (() => void) | null = null;
  private dir = new Vector3(1.0, -1.35, 0.85);
  private normal: number[] = [0, 0, 1];
  barKm = 0;

  constructor(el: HTMLElement) {
    this.el = el;
    this.camera = new PerspectiveCamera(36, 1, 0.01, 1e4);
    this.camera.up.set(0, 0, 1);
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.scene.background = new Color(0xfafaf9);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    const lutTex = new DataTexture(lut("rdbu_r"), 256, 1, RGBAFormat);
    lutTex.minFilter = lutTex.magFilter = LinearFilter;
    lutTex.needsUpdate = true;
    this.mat = new ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, side: DoubleSide,
      glslVersion: GLSL3,
      uniforms: {
        uLut: { value: lutTex }, uWin: { value: new Vector2(-1, 1) },
        uMark: { value: false },
      },
    });
    this.content.add(this.wire);
    this.scene.add(this.content, this.furniture);
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  /** The sub-mesh: every facet's k^2 cells as flat triangles. `tris` is
   * (n, 3, 3) and `corners` (k^2, 3, 3) barycentric; a cell corner is
   * `corners[c][j] . tris[f]`, which is all the page needs to know about the
   * lattice; at k = 1 the one cell is the facet. */
  build(tris: Float32Array, corners: Float32Array, geom: FaultGeom[],
        normal: number[]) {
    const n = tris.length / 9, kk = corners.length / 9;
    const pos = new Float32Array(n * kk * 9);
    let o = 0;
    for (let f = 0; f < n; f++) {
      for (let c = 0; c < kk; c++) {
        for (let j = 0; j < 3; j++) {
          const b0 = corners[(c * 3 + j) * 3], b1 = corners[(c * 3 + j) * 3 + 1],
                b2 = corners[(c * 3 + j) * 3 + 2];
          for (let x = 0; x < 3; x++) {
            pos[o++] = b0 * tris[f * 9 + x] + b1 * tris[f * 9 + 3 + x]
                     + b2 * tris[f * 9 + 6 + x];
          }
        }
      }
    }
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(pos, 3));
    this.valueAttr = new Float32BufferAttribute(new Float32Array(n * kk * 3), 1);
    g.setAttribute("value", this.valueAttr);
    this.mesh = new Mesh(g, this.mat);
    this.content.add(this.mesh);

    // The welded edge set of each fault, once, and its rim in the accent. A
    // rim includes the trace along the surface and, for the splay, the
    // junction chord it shares with its host.
    for (const fg of geom) {
      const lines = new LineSegments(meshEdges(fg.vertices, fg.triangles),
        new LineBasicMaterial({ color: 0x3c4046, transparent: true,
                                opacity: 0.35, depthWrite: false }));
      lines.renderOrder = 1;
      this.wire.add(lines);
      const rg = new BufferGeometry();
      rg.setAttribute("position", new Float32BufferAttribute(
        fg.outline.flat(), 3));
      const rim = new LineLoop(rg, new LineBasicMaterial({ color: 0xb5367a }));
      rim.renderOrder = 2;
      this.content.add(rim);
    }
    this.normal = normal;
    const box = new Box3().setFromObject(this.content);

    // The free surface, as a faint sheet at z = 0 with its outline: the
    // faults break it, and without it a trace reads as just another rim.
    const pad = 0.08 * Math.max(box.max.x - box.min.x, box.max.y - box.min.y);
    const x0 = box.min.x - pad, x1 = box.max.x + pad;
    const y0 = box.min.y - pad, y1 = box.max.y + pad;
    const sg = new BufferGeometry();
    sg.setAttribute("position", new Float32BufferAttribute(
      [x0, y0, 0, x1, y0, 0, x1, y1, 0, x0, y1, 0], 3));
    sg.setIndex([0, 1, 2, 0, 2, 3]);
    const sheet = new Mesh(sg, new MeshBasicMaterial({
      color: 0x8a8f96, transparent: true, opacity: 0.07, depthWrite: false,
      side: DoubleSide }));
    sheet.renderOrder = 0;
    const edge = new LineLoop(sg.clone(), new LineBasicMaterial({
      color: 0x9a9ea4, transparent: true, opacity: 0.6, depthWrite: false }));
    this.content.add(sheet, edge);

    this.barKm = buildFurniture(this.furniture, box, 0x44484c);
    this.preset("oblique");
  }

  /** One value per CELL, replicated onto the cell's three vertices. */
  setValues(perCell: Float32Array) {
    if (!this.valueAttr) return;
    const a = this.valueAttr.array as Float32Array;
    for (let c = 0; c < perCell.length; c++) {
      a[c * 3] = a[c * 3 + 1] = a[c * 3 + 2] = perCell[c];
    }
    this.valueAttr.needsUpdate = true;
    this.render();
  }

  setMap(name: MapName) {
    const t = this.mat.uniforms.uLut.value as DataTexture;
    (t.image.data as Uint8Array).set(lut(name));
    t.needsUpdate = true;
    this.render();
  }

  setWindow(lo: number, hi: number) {
    (this.mat.uniforms.uWin.value as Vector2).set(lo, hi);
    this.render();
  }

  setMark(on: boolean) { this.mat.uniforms.uMark.value = on; this.render(); }
  setWire(on: boolean) { this.wire.visible = on; this.render(); }

  preset(name: Preset) {
    this.dir = presetDir(name, this.normal);
    const box = new Box3().setFromObject(this.content);
    if (box.isEmpty()) return;
    fitCamera(this.camera, this.controls, box, this.dir);
    this.render();
  }

  screenshot(): string { return screenshot(this.renderer, this.scene, this.camera); }

  resize() {
    const w = this.el.clientWidth || 640;
    const h = this.el.clientHeight || Math.round(w * 0.62);
    this.renderer.setSize(w, h, true);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  start() { this.stop = onDemand(this.controls, () => this.render()); }
  render() { this.renderer.render(this.scene, this.camera); }
  dispose() { this.stop?.(); this.controls.dispose(); this.renderer.dispose(); }
}

async function bin(url: string, bytes: number): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split("/").pop()}: ${r.status}`);
  const b = await r.arrayBuffer();
  if (b.byteLength !== bytes) {
    throw new Error(`${url.split("/").pop()} is ${b.byteLength} bytes, expected ${bytes}`);
  }
  return b;
}

const fmt = (v: number, d = 1) => {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(d) : v.toPrecision(2);
};

export async function startOnFault(root: HTMLElement, baseUrl: string,
                                   manifestName = "onfault_manifest.json") {
  const canvasEl = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;
  const legend = root.querySelector<HTMLElement>("[data-legend]");
  const base = baseUrl.replace(/\/?$/, "/") + "data/";

  let man: OnFaultManifest;
  let geom: { faults: FaultGeom[] };
  try {
    man = await (await fetch(base + manifestName)).json();
    geom = await (await fetch(base + "onfault_geometry.json")).json();
  } catch (e) {
    status.textContent = `Could not load the on-fault data: ${e}`;
    return;
  }
  const A = man.arrays;
  const load = (key: string) => bin(`${base}${key}.bin`, A[key].bytes);

  const view = new OnFaultScene(canvasEl);
  view.start();
  try {
    const [t, c, ar] = await Promise.all([
      load("onfault.tris"), load("onfault.corners"), load("onfault.area")]);
    const dipA = man.faults[0].dip_deg * Math.PI / 180;
    view.build(new Float32Array(t), new Float32Array(c), geom.faults,
               [Math.sin(dipA), 0, Math.cos(dipA)]);
    var facetArea = new Float32Array(ar);
  } catch (err) {
    status.textContent = `failed to build the mesh: ${err}`;
    return;
  }
  const kk = man.cells_per_facet, n = man.n_facets;
  const cellArea = (c: number) => facetArea[Math.floor(c / kk)] / kk;

  let kernel = man.defaults.kernel;
  let metric = man.defaults.metric;
  let winMPa = Number(man.window[metric]) * 1e3;     // half-width, MPa
  let mark = false;

  const cache = new Map<string, Float32Array>();
  async function values(kern: string, met: string): Promise<Float32Array> {
    const key = met === "slip" ? `onfault.${kern}.slip` : `onfault.${kern}.${met}`;
    const hit = cache.get(key);
    if (hit) return hit;
    let v = new Float32Array(await load(key));
    if (met === "slip") {              // per element -> per cell, km -> m
      const out = new Float32Array(n * kk);
      for (let f = 0; f < n; f++) for (let c = 0; c < kk; c++) out[f * kk + c] = v[f] * 1e3;
      v = out;
    }
    cache.set(key, v);
    return v;
  }

  const sel = (name: string, opts: [string, string][], value: string,
               onChange: (v: string) => void) => {
    const wrap = root.querySelector<HTMLElement>(`[data-ctl="${name}"]`);
    if (!wrap) return null;
    const s = document.createElement("select");
    for (const [v, label] of opts) {
      const o = document.createElement("option");
      o.value = v; o.textContent = label; o.selected = v === value;
      s.appendChild(o);
    }
    s.addEventListener("change", () => onChange(s.value));
    wrap.appendChild(s);
    return s;
  };
  const box = (name: string, checked: boolean, onChange: (on: boolean) => void) => {
    const wrap = root.querySelector<HTMLElement>(`[data-ctl="${name}"]`);
    if (!wrap) return null;
    const b = document.createElement("input");
    b.type = "checkbox"; b.checked = checked;
    b.addEventListener("change", () => onChange(b.checked));
    wrap.appendChild(b);
    return b;
  };

  sel("kernel", Object.keys(KERNELS).map((k) => [k, KERNELS[k]]), kernel,
      (v) => { kernel = v; void refresh(); });
  sel("metric", Object.keys(METRICS).map((k) => [k, METRICS[k].label]), metric,
      (v) => { metric = v; winMPa = defaultWindow(v); void refresh(); });

  function defaultWindow(met: string): number {
    if (met === "slip") return 0;        // set from the data when loaded
    return Number(man.window[met]) * 1e3;
  }

  // The window slider is log-spaced over 0.1..100 MPa: the maps span decades
  // and a linear slider would spend most of its travel above anything the
  // cores ever reach.
  const winWrap = root.querySelector<HTMLElement>(`[data-ctl="window"]`);
  let winIn: HTMLInputElement | null = null, winOut: HTMLElement | null = null;
  if (winWrap) {
    winIn = document.createElement("input");
    winIn.type = "range"; winIn.min = "-1"; winIn.max = "2"; winIn.step = "0.05";
    winOut = document.createElement("span"); winOut.className = "val";
    winIn.addEventListener("input", () => {
      winMPa = 10 ** Number(winIn!.value); applyWindow();
    });
    winWrap.append(winIn, winOut);
  }
  box("mark", false, (on) => { mark = on; view.setMark(on); });
  box("wire", true, (on) => view.setWire(on));

  const viewWrap = root.querySelector<HTMLElement>(`[data-ctl="view"]`);
  if (viewWrap) {
    for (const p of ["oblique", "strike", "cross", "top"] as Preset[]) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = p;
      b.addEventListener("click", () => view.preset(p));
      viewWrap.appendChild(b);
    }
  }
  const shotWrap = root.querySelector<HTMLElement>(`[data-ctl="shot"]`);
  if (shotWrap) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = "save PNG";
    b.addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = view.screenshot();
      a.download = `mhs-onfault-${kernel}-${metric}.png`;
      a.click();
    });
    shotWrap.appendChild(b);
  }
  box("publication", false, (on) => {
    root.classList.toggle("publication", on);
    requestAnimationFrame(() => view.resize());
  });

  let current: Record<string, Float32Array> = {};

  function applyWindow() {
    const m = METRICS[metric];
    if (m.signed) view.setWindow(-winMPa * 1e-3, winMPa * 1e-3);
    else view.setWindow(0, winMPa);          // slip is already in metres
    if (winIn) winIn.value = String(Math.log10(Math.max(winMPa, 0.1)));
    if (winOut) winOut.textContent = m.signed ? `±${fmt(winMPa)} MPa`
                                              : `0 – ${fmt(winMPa)} m`;
    paintLegend();
  }

  /** Area-weighted fraction of cells outside the window. */
  function saturated(v: Float32Array): number {
    const m = METRICS[metric];
    let sat = 0, tot = 0;
    const lo = m.signed ? -winMPa * 1e-3 : 0, hi = m.signed ? winMPa * 1e-3 : winMPa;
    for (let c = 0; c < v.length; c++) {
      const a = cellArea(c);
      tot += a;
      const x = v[c];
      if (x < lo || x > hi) sat += a;
    }
    return sat / tot;
  }

  function paintLegend() {
    if (!legend) return;
    const m = METRICS[metric];
    const fa = man.faults[0];
    const bits: string[] = [];
    bits.push(`${m.label}${m.signed ? " (MPa)" : " (m)"}`);
    bits.push(m.signed ? `window ±${fmt(winMPa)} MPa` : `window 0–${fmt(winMPa)} m`);
    const sats = Object.keys(KERNELS)
      .filter((k) => current[k])
      .map((k) => `${k} ${(100 * saturated(current[k])).toFixed(1)} %`);
    if (sats.length) bits.push(`saturated: ${sats.join(", ")} of area`);
    bits.push(`remote on A: τ ${fmt(fa.remote_shear_MPa)} MPa, `
      + `σₙ′ ${fmt(fa.remote_sigma_n_MPa)} MPa`);
    bits.push(kk === 1 ? `one value per element, sampled at its centroid`
                       : `${kk} cells per facet`);
    bits.push(`${man.n_facets} elements, h ${man.h_km} km, eps ${man.eps_km} km`);
    bits.push(`scale bar ${view.barKm} km`);
    bits.push(`axes x red, y green, z blue (z up, depth negative); the grey sheet is z = 0`);
    legend.textContent = bits.join(" · ");
  }

  // Latest call wins, as on the isosurface page: two controls in one tick
  // start two of these, and the slower must not paint over the faster.
  let seq = 0;
  async function refresh() {
    const mine = ++seq;
    status.textContent = "loading…";
    try {
      // Every kernel of the chosen metric, so the legend can state all the
      // saturated fractions without a toggle.
      const names = Object.keys(KERNELS);
      const loaded = await Promise.all(names.map((k) => values(k, metric)));
      if (mine !== seq) return;
      current = {};
      names.forEach((k, i) => { current[k] = loaded[i]; });
      const m = METRICS[metric];
      if (metric === "slip" && winMPa === 0) {
        let mx = 0;
        for (const v of loaded) for (let i = 0; i < v.length; i++) mx = Math.max(mx, v[i]);
        winMPa = mx;
      }
      view.setMap(m.map);
      view.setValues(current[kernel]);
      applyWindow();
      status.textContent = `${KERNELS[kernel]} — ${m.label}`;
      root.dataset.shown = `${kernel}/${metric}`;
    } catch (e) {
      status.textContent = `failed: ${e}`;
    }
  }

  await refresh();
  return view;
}
