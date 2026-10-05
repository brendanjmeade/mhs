// The home page figure: one row of the interaction matrix, chosen by clicking.
//
// This is NOT the volume viewer. At 256 triangles the quantity is a single
// number per element, so it is a vertex-colour attribute recoloured on the CPU
// in well under a frame -- a 3-D texture and a fragment shader would be the
// wrong tool, and a sampler cannot be raycast against anyway. The volume
// machinery in scene.ts is reused unchanged on the Examples page, where the
// field really is three-dimensional.
//
// Geometry is NON-INDEXED on purpose: each triangle needs its own three
// vertices to carry a flat colour, and it makes `faceIndex` from the raycaster
// equal to the source index with no lookup table to get wrong.
//
// COLOUR SCALE. The values span about -14.5 to +2.9 MPa, and almost all of the
// area is within a few hundredths of zero, so a linear ramp would show one red
// patch on a white sheet. The mapping is therefore symmetric-log: zero sits at
// the centre of a diverging map, and each decade either side gets equal width.
// A diverging map with its neutral colour anywhere but zero would be a lie
// about the sign, so the centre is pinned rather than fitted.

import {
  BufferGeometry, Color, DoubleSide, Float32BufferAttribute, LineBasicMaterial,
  LineSegments, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene,
  Vector2, Vector3, WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { lut, cssGradient, type MapName } from "./colormaps";

const MAP: MapName = "rdbu_r";
const V0 = 0.02;              // MPa: the half-width of the scale's linear core

export interface FaultPayload {
  schema: number;
  unit: string;
  meta: Record<string, any>;
  triangles: number[][][];
  centroids: number[][];
  arrays: Record<string, { bytes: number; n: number; min: number; max: number }>;
}

export type Component = "cfs" | "shear" | "normal";

export const COMPONENT_LABEL: Record<Component, string> = {
  cfs: "Coulomb",
  shear: "shear",
  normal: "normal",
};

/** Symmetric-log into [0, 1], with 0 mapped exactly to the centre. */
function norm(v: number, vmax: number): number {
  const d = Math.log10(1 + vmax / V0);
  const t = Math.log10(1 + Math.abs(v) / V0) / (d || 1);
  return 0.5 + 0.5 * Math.sign(v) * Math.min(t, 1);
}

export class FaultScene {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private mesh: Mesh;
  private picked: LineSegments;
  private el: HTMLElement;
  private raf = 0;
  private colours: Float32BufferAttribute;
  private tris: number[][][];
  private table: ReturnType<typeof lut>;
  private data: Record<string, Float32Array> = {};
  private comp: Component = "cfs";
  private source = 0;
  private vmax = 1;
  private n: number;
  private onPick?: (i: number, value: number, comp: Component) => void;

  constructor(el: HTMLElement, p: FaultPayload) {
    this.el = el;
    this.tris = p.triangles;
    this.n = p.triangles.length;
    this.table = lut(MAP);

    this.scene.background = new Color(0xffffff);
    this.camera = new PerspectiveCamera(36, 1, 0.5, 5e3);
    this.camera.up.set(0, 0, 1);     // the model's vertical is +z, as in scene.ts
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);

    const pos: number[] = [];
    for (const t of this.tris) for (const v of t) pos.push(v[0], v[1], v[2]);
    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(pos, 3));
    this.colours = new Float32BufferAttribute(new Float32Array(pos.length), 3);
    geo.setAttribute("color", this.colours);
    this.mesh = new Mesh(geo, new MeshBasicMaterial({
      vertexColors: true, side: DoubleSide }));
    this.scene.add(this.mesh);

    // The fault's own edges, so individual patches stay legible when the
    // colours are nearly equal.
    const wire: number[] = [];
    for (const t of this.tris) {
      for (let i = 0; i < 3; i++) {
        const a = t[i], b = t[(i + 1) % 3];
        wire.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    }
    const wg = new BufferGeometry();
    wg.setAttribute("position", new Float32BufferAttribute(wire, 3));
    this.scene.add(new LineSegments(wg, new LineBasicMaterial({
      color: 0x888888, transparent: true, opacity: 0.35 })));

    // The selected source, outlined. Drawn as its own object so changing the
    // source is three number writes rather than a geometry rebuild.
    const pg = new BufferGeometry();
    pg.setAttribute("position", new Float32BufferAttribute(new Float32Array(18), 3));
    this.picked = new LineSegments(pg, new LineBasicMaterial({ color: 0x111111 }));
    this.scene.add(this.picked);

    this.scene.add(freeSurface());

    const c = centre(this.tris);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(c);
    this.camera.position.set(c.x + 44, c.y - 52, c.z + 34);
    this.controls.update();      // orient now, or the first frame draws blank

    this.renderer.domElement.addEventListener("click", (e) => this.click(e));
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  setData(name: Component, values: Float32Array) {
    this.data[name] = values;
  }

  /** Which component to show, and the source patch to show it for. */
  select(comp: Component, source = this.source) {
    this.comp = comp;
    this.source = Math.max(0, Math.min(this.n - 1, source));
    const a = this.data[comp];
    if (!a) return;
    // One scale for the whole component, not per source: a per-source rescale
    // would make every patch look equally consequential.
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
    this.vmax = m || 1;

    const col = this.colours.array as Float32Array;
    const off = this.source * this.n;
    for (let i = 0; i < this.n; i++) {
      const t = norm(a[off + i], this.vmax);
      const k = Math.min(255, Math.max(0, Math.round(t * 255))) * 4;
      const r = this.table[k] / 255, g = this.table[k + 1] / 255,
            b = this.table[k + 2] / 255;
      for (let v = 0; v < 3; v++) {
        const j = (i * 3 + v) * 3;
        col[j] = r; col[j + 1] = g; col[j + 2] = b;
      }
    }
    this.colours.needsUpdate = true;

    const tri = this.tris[this.source];
    const pp = (this.picked.geometry.getAttribute("position").array as Float32Array);
    for (let i = 0; i < 3; i++) {
      const p0 = tri[i], p1 = tri[(i + 1) % 3];
      pp[i * 6] = p0[0]; pp[i * 6 + 1] = p0[1]; pp[i * 6 + 2] = p0[2];
      pp[i * 6 + 3] = p1[0]; pp[i * 6 + 4] = p1[1]; pp[i * 6 + 5] = p1[2];
    }
    this.picked.geometry.getAttribute("position").needsUpdate = true;
    this.render();
    this.onPick?.(this.source, a[off + this.source], comp);
  }

  get currentRange(): [number, number] { return [-this.vmax, this.vmax]; }
  get currentSource(): number { return this.source; }

  onSelect(fn: (i: number, value: number, comp: Component) => void) {
    this.onPick = fn;
  }

  private click(e: MouseEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1);
    const ray = new Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hit = ray.intersectObject(this.mesh, false)[0];
    // A miss keeps the current source: clicking the background to clear the
    // figure would leave nothing to look at.
    if (hit?.faceIndex != null) this.select(this.comp, hit.faceIndex);
  }

  resize() {
    const w = this.el.clientWidth || 640;
    const h = this.el.clientHeight || Math.round(w * 0.58);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  /** A slow idle orbit that hands control over for good on first interaction.
   *
   * After the visitor takes over, the loop is CANCELLED and runs again only
   * between OrbitControls' start and end events -- the damping needs frames
   * while a drag is live and nothing needs them the rest of the time. Leaving
   * the orbit loop running would keep a phone's GPU awake on a static page.
   */
  start(idle = true) {
    let t = 0;
    const c = this.controls.target;
    const radius = 70, height = 34;
    let spinning = idle;

    const damped = () => {
      this.raf = requestAnimationFrame(damped);
      if (this.controls.update()) this.renderer.render(this.scene, this.camera);
    };
    const orbit = () => {
      if (!spinning) return;
      this.raf = requestAnimationFrame(orbit);
      t += 0.0011;
      this.camera.position.set(c.x + radius * Math.cos(t - 0.9),
                               c.y + radius * Math.sin(t - 0.9),
                               c.z + height);
      this.camera.lookAt(c);
      this.renderer.render(this.scene, this.camera);
    };
    const stop = () => {
      if (!spinning) return;
      spinning = false;
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.controls.update();
      this.render();
    };
    for (const ev of ["pointerdown", "wheel"] as const) {
      this.renderer.domElement.addEventListener(ev, stop, { once: true });
    }
    this.controls.addEventListener("start", () => { if (!this.raf) damped(); });
    this.controls.addEventListener("end", () => {
      cancelAnimationFrame(this.raf); this.raf = 0; this.render();
    });
    if (spinning) orbit(); else this.render();
  }

  render() { this.renderer.render(this.scene, this.camera); }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

function centre(tris: number[][][]): Vector3 {
  let x = 0, y = 0, z = 0, n = 0;
  for (const t of tris) for (const v of t) { x += v[0]; y += v[1]; z += v[2]; n++; }
  return new Vector3(x / n, y / n, z / n);
}

/** The free surface at z = 0, so the fault is visibly in a half space. */
function freeSurface(x0 = -34, y0 = -26, w = 68, h = 56, n = 14): LineSegments {
  const pos: number[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    pos.push(x0, y0 + t * h, 0, x0 + w, y0 + t * h, 0);
    pos.push(x0 + t * w, y0, 0, x0 + t * w, y0 + h, 0);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  return new LineSegments(g, new LineBasicMaterial({
    color: 0x999999, transparent: true, opacity: 0.4 }));
}

/** Load the payload and wire the figure into `root`. */
export async function startFault(root: HTMLElement, baseUrl: string) {
  const base = baseUrl.replace(/\/?$/, "/") + "data/";
  const status = root.querySelector<HTMLElement>("[data-status]");
  const canvas = root.querySelector<HTMLElement>("[data-canvas]") ?? root;

  let p: FaultPayload;
  try {
    p = await (await fetch(base + "fault.json")).json();
  } catch (e) {
    if (status) status.textContent = `Could not load the fault data: ${e}`;
    return;
  }

  const scene = new FaultScene(canvas, p);
  const want = async (name: Component) => {
    const r = await fetch(`${base}${name}.bin`);
    if (!r.ok) throw new Error(`${name}.bin: ${r.status}`);
    const a = new Float32Array(await r.arrayBuffer());
    const n = p.triangles.length;
    if (a.length !== n * n) {
      throw new Error(`${name}.bin holds ${a.length} floats, expected ${n * n}`);
    }
    scene.setData(name, a);
    return a;
  };

  await want("cfs");
  // Start from a patch partway down the fault rather than index 0, which is a
  // corner and the least representative element there is.
  const start = Math.floor(p.triangles.length * 0.42);
  scene.select("cfs", start);
  scene.start();

  const bar = root.querySelector<HTMLElement>("[data-bar]");
  const paintBar = () => {
    if (!bar) return;
    const [lo, hi] = scene.currentRange;
    bar.innerHTML = "";
    const strip = document.createElement("div");
    strip.className = "strip";
    strip.style.background = cssGradient(MAP);
    const ends = document.createElement("div");
    ends.className = "ends";
    ends.innerHTML = `<span>${fmt(lo)}</span>` +
      `<span class="mid">MPa, symmetric log about zero</span>` +
      `<span>+${fmt(hi)}</span>`;
    bar.append(strip, ends);
  };

  scene.onSelect((i, v, comp) => {
    if (!status) return;
    const c = p.centroids[i];
    // The shear and Coulomb rows have a negative diagonal as a matter of
    // physics -- a patch that slips relieves its own driving stress -- so that
    // is worth saying. The normal row's diagonal carries no such guarantee, so
    // it is reported without the explanation.
    const own = comp === "normal"
      ? `the normal traction on the patch itself is <b>${v.toFixed(2)} MPa</b>`
      : `it unloads itself by <b>${Math.abs(v).toFixed(2)} MPa</b>`;
    status.innerHTML =
      `${COMPONENT_LABEL[comp]} &middot; source patch <b>${i}</b>, centroid ` +
      `${c[0].toFixed(1)}, ${c[1].toFixed(1)}, ${c[2].toFixed(1)} km ` +
      `&mdash; ${own}. Click any patch to move the source.`;
  });

  for (const b of root.querySelectorAll<HTMLButtonElement>("[data-comp]")) {
    b.addEventListener("click", async () => {
      const name = b.dataset.comp as Component;
      try {
        await want(name);
      } catch (e) {
        if (status) status.textContent = `failed: ${e}`;
        return;
      }
      for (const o of root.querySelectorAll<HTMLButtonElement>("[data-comp]")) {
        o.classList.toggle("ghost", o !== b);
      }
      scene.select(name);
      paintBar();
    });
  }
  paintBar();
  return scene;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-2 || a >= 1e4) return v.toExponential(1);
  return v.toPrecision(3).replace(/\.?0+$/, "");
}
