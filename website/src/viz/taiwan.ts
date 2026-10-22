// The Longitudinal Valley Fault under a sweep of regional stress tensors: the
// static Coulomb LCP's slip and activation maps on the two TEM segments, one
// value per element, selected by the tensor's azimuth and regime, the
// effective-stress model, the kernel and the load. The scene, the shader and
// the colour lookup are the on-fault page's (`onfault.ts`): colour is data,
// unlit, one value per facet sampled at its centroid; only the controls and
// the manifest differ.

import { OnFaultScene } from "./onfault";
import type { MapName } from "./colormaps";
import type { Preset } from "./stage";

interface Case {
  lam0_MPa: number; lam_half_MPa: number | null; never_fail: number;
  never_fail_area: number; active_fraction: number[]; moment_Nm: number[];
  max_slip_m: number[]; rake_mean_deg: number; min_eig_sym_negP: number;
  branching_ratio: number; newton_solves: number; lam_tensile_over_lam0: number;
}

export interface TaiwanManifest {
  schema: number; units: string; git: string; n_elements: number;
  h_km: number; eps_km: number; frame: string;
  origin: { lon: number; lat: number };
  faults: { name: string; n: number; trace_km: number; depth_km: number;
            dip_deg: number[]; strike_azimuth_deg: number; area_km2: number;
            edge_km: number[] }[];
  data_source: string;
  sweep: { thetas: number[]; rs: number[]; models: string[]; kernels: string[];
           loads: number[]; ship_loads: number[]; quoted_load: number };
  friction: { mu: number; mean_stress_ratio: number; pore_ratio: number;
              sigma_eff_uniform_GPa: number };
  labels: { models: Record<string, string>; kernels: Record<string, string> };
  cases: Record<string, Case>;
  arrays: Record<string, { dtype: string; shape: number[]; bytes: number }>;
  defaults: { kernel: string; model: string; theta: number; r: number;
              load: number; metric: string };
}

interface FaultGeom {
  name: string; vertices: number[][]; triangles: number[][]; outline: number[][];
}

const METRICS: Record<string, { label: string; map: MapName; unit: string }> = {
  slip: { label: "slip at the chosen load", map: "viridis", unit: "m" },
  act: { label: "activation load λ/λ₀ (grey: not by 6 λ₀)", map: "viridis", unit: "λ₀" },
  rake: { label: "rake of the slip direction (0 sinistral, 90 reverse, 180 dextral)",
          map: "rdbu_r", unit: "deg" },
  d: { label: "Coulomb loading rate per unit λ (≤ 0: can never fail)",
       map: "rdbu_r", unit: "" },
};

const REGIME = (r: number) =>
  r === 0 ? "0: thrust (Sv least)" : r === 1 ? "1: strike-slip (Sv intermediate)"
  : r === 0.5 ? "0.5: axial compression (Sv = Shmin)" : `${r}`;

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
const az3 = (t: number) => String(Math.round(t)).padStart(3, "0");
const r2 = (r: number) => r.toFixed(2);

export async function startTaiwan(root: HTMLElement, baseUrl: string,
                                  manifestName = "taiwan_manifest.json") {
  const canvasEl = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;
  const legend = root.querySelector<HTMLElement>("[data-legend]");
  const base = baseUrl.replace(/\/?$/, "/") + "data/";

  let man: TaiwanManifest;
  let geom: { faults: FaultGeom[] };
  try {
    man = await (await fetch(base + manifestName)).json();
    geom = await (await fetch(base + "taiwan_geometry.json")).json();
  } catch (e) {
    status.textContent = `Could not load the Taiwan data: ${e}`;
    return;
  }
  const A = man.arrays;
  const load = (key: string) => bin(`${base}${key}.bin`, A[key].bytes);

  const view = new OnFaultScene(canvasEl);
  view.start();
  let area: Float32Array;
  try {
    const [t, c, ar] = await Promise.all([
      load("taiwan.tris"), load("taiwan.corners"), load("taiwan.area")]);
    // The fault's mean upward normal, for the strike and cross views.
    const f0 = man.faults[0];
    const dip = f0.dip_deg[1] * Math.PI / 180;
    const dipAz = (f0.strike_azimuth_deg + 90) * Math.PI / 180;
    const normal = [Math.sin(dip) * Math.sin(dipAz), Math.sin(dip) * Math.cos(dipAz),
                    Math.cos(dip)];
    view.build(new Float32Array(t), new Float32Array(c), geom.faults, normal);
    area = new Float32Array(ar);
  } catch (err) {
    status.textContent = `failed to build the mesh: ${err}`;
    return;
  }
  const S = man.sweep;
  let kernel = man.defaults.kernel;
  let model = man.defaults.model;
  let theta = man.defaults.theta;
  let r = man.defaults.r;
  let loadF = S.ship_loads.includes(man.defaults.load) ? man.defaults.load : S.ship_loads[0];
  let metric = man.defaults.metric;
  let win = 0;              // upper bound of the window in the metric's unit
  let mark = false;

  const caseKey = () => `${kernel}.${model}.az${az3(theta)}.r${r2(r)}`;
  const arrayKey = () => {
    if (metric === "slip") return `taiwan.${caseKey()}.slip${loadF.toFixed(2)}`;
    if (metric === "act") return `taiwan.${caseKey()}.act`;
    return `taiwan.az${az3(theta)}.r${r2(r)}.${metric}`;
  };

  const cache = new Map<string, Float32Array>();
  async function values(): Promise<Float32Array> {
    const key = arrayKey();
    const hit = cache.get(key);
    if (hit) return hit;
    if (!A[key]) throw new Error(`${key} is not in the manifest`);
    let v = new Float32Array(await load(key));
    if (metric === "slip") {                   // km -> m
      const out = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) out[i] = v[i] * 1e3;
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

  sel("theta", S.thetas.map((t) => [String(t), `${t}°`]), String(theta),
      (v) => { theta = Number(v); win = 0; void refresh(); });
  sel("r", S.rs.map((x) => [String(x), REGIME(x)]), String(r),
      (v) => { r = Number(v); win = 0; void refresh(); });
  sel("model", S.models.map((m) => [m, man.labels.models[m]]), model,
      (v) => { model = v; win = 0; void refresh(); });
  sel("kernel", S.kernels.map((k) => [k, man.labels.kernels[k]]), kernel,
      (v) => { kernel = v; void refresh(); });
  const loadSel = sel("load", S.ship_loads.map((f) => [String(f), `${f} λ₀`]),
                      String(loadF), (v) => { loadF = Number(v); win = 0; void refresh(); });
  sel("metric", Object.keys(METRICS).map((k) => [k, METRICS[k].label]), metric,
      (v) => { metric = v; win = 0; if (loadSel) loadSel.disabled = v !== "slip"; void refresh(); });

  const winWrap = root.querySelector<HTMLElement>(`[data-ctl="window"]`);
  let winIn: HTMLInputElement | null = null, winOut: HTMLElement | null = null;
  if (winWrap) {
    winIn = document.createElement("input");
    winIn.type = "range"; winIn.min = "-1"; winIn.max = "2"; winIn.step = "0.05";
    winOut = document.createElement("span"); winOut.className = "val";
    winIn.addEventListener("input", () => { win = 10 ** Number(winIn!.value); applyWindow(); });
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
      a.download = `mhs-taiwan-${caseKey()}-${metric}${metric === "slip" ? "-" + loadF : ""}.png`;
      a.click();
    });
    shotWrap.appendChild(b);
  }
  box("publication", false, (on) => {
    root.classList.toggle("publication", on);
    requestAnimationFrame(() => view.resize());
  });

  let current: Float32Array | null = null;

  /** Fixed windows where the quantity has a natural range; the slider only
   * drives slip and the Coulomb rate. */
  function applyWindow() {
    if (metric === "act") view.setWindow(1, S.loads[S.loads.length - 1]);
    else if (metric === "rake") view.setWindow(0, 180);
    else if (metric === "d") view.setWindow(-win, win);
    else view.setWindow(0, win);
    if (winIn) {
      winIn.disabled = metric === "act" || metric === "rake";
      winIn.value = String(Math.log10(Math.max(win, 0.1)));
    }
    if (winOut) {
      winOut.textContent = metric === "act" ? `1 – ${S.loads[S.loads.length - 1]} λ₀`
        : metric === "rake" ? "0 – 180°"
        : metric === "d" ? `±${fmt(win)}` : `0 – ${fmt(win)} m`;
    }
    paintLegend();
    paintStatus();
  }

  const describe = () => `S_Hmax ${theta}°, r ${r} (${REGIME(r).split(":")[1].trim()}), `
    + `${man.labels.models[model]}, ${man.labels.kernels[kernel]}`;

  function paintStatus() {
    const m = METRICS[metric];
    const what = metric === "slip" ? `slip at ${loadF} λ₀` : m.label;
    status.textContent = `${describe()} — ${what} — ${winOut?.textContent ?? ""}`;
  }

  /** Area-weighted fraction outside the window (slip and d only). */
  function saturated(v: Float32Array): number {
    let sat = 0, tot = 0;
    const lo = metric === "d" ? -win : 0;
    const hi = metric === "act" ? S.loads[S.loads.length - 1] : metric === "rake" ? 180 : win;
    for (let i = 0; i < v.length; i++) {
      if (Number.isNaN(v[i])) continue;
      tot += area[i];
      if (v[i] < lo || v[i] > hi) sat += area[i];
    }
    return tot > 0 ? sat / tot : 0;
  }

  function paintLegend() {
    if (!legend) return;
    const c = man.cases[caseKey()];
    const bits: string[] = [];
    bits.push(describe());
    if (c) {
      const il = S.loads.indexOf(loadF);
      bits.push(`λ₀ ${fmt(c.lam0_MPa)} MPa of maximum shear (uniform model's first failure)`);
      bits.push(c.lam_half_MPa === null ? `half the area is not active by 6 λ₀`
        : `half the area active at ${fmt(c.lam_half_MPa)} MPa = ${fmt(c.lam_half_MPa / c.lam0_MPa)} λ₀`);
      if (il >= 0) {
        bits.push(`at ${loadF} λ₀: ${(100 * c.active_fraction[il]).toFixed(1)} % of the area active, `
          + `moment ${c.moment_Nm[il].toExponential(2)} N m, max slip ${fmt(c.max_slip_m[il])} m`);
      }
      bits.push(`${(100 * c.never_fail_area).toFixed(1)} % of the area can never fail under this tensor`);
      bits.push(`area-mean rake ${c.rake_mean_deg.toFixed(0)}°`);
      bits.push(`min eig sym(−P) ${c.min_eig_sym_negP.toFixed(1)}, branching ratio ${c.branching_ratio.toFixed(3)}`
        + (c.newton_solves ? `, ${c.newton_solves} of ${S.loads.length} loads by the active-set solver` : ""));
    }
    if (current && (metric === "slip" || metric === "d")) {
      bits.push(`saturated ${(100 * saturated(current)).toFixed(1)} % of the area`);
    }
    bits.push(`${man.n_elements} elements, h ${man.h_km.toFixed(2)} km, eps ${man.eps_km.toFixed(2)} km`);
    bits.push(`scale bar ${view.barKm} km; axes x east (red), y north (green), z up (blue); the grey sheet is the surface`);
    legend.textContent = bits.join(" · ");
  }

  let seq = 0;
  async function refresh() {
    const mine = ++seq;
    status.textContent = "loading…";
    try {
      const v = await values();
      if (mine !== seq) return;
      current = v;
      if (win === 0) {
        let mx = 0;
        for (let i = 0; i < v.length; i++) if (!Number.isNaN(v[i])) mx = Math.max(mx, Math.abs(v[i]));
        win = mx > 0 ? mx : 1;
      }
      view.setMap(METRICS[metric].map);
      view.setValues(v);
      applyWindow();
      root.dataset.shown = `${caseKey()}/${metric}`;
    } catch (e) {
      status.textContent = `failed: ${e}`;
    }
  }

  if (loadSel) loadSel.disabled = metric !== "slip";
  await refresh();
  return view;
}
