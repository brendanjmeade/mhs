// Opaque stress isosurfaces, lit: the one scene on this site that is shaded by
// light rather than by arithmetic.
//
// WHY THIS ONE IS DIFFERENT. Every other scene here fakes its shading on the
// CPU and uses MeshBasicMaterial, because what it is drawing is a FIELD and the
// colour IS the quantity -- a light would corrupt the reading. Here the
// quantity is the SURFACE: a level set carries one value everywhere by
// definition, so there is nothing for a colour map to say, and the only things
// that can carry the shape are silhouette, curvature and light. Hence PBR
// materials, an environment and a three-light rig, and hence opaque: a stack of
// translucent shells would hide exactly the topology this page is about.
//
// THE PAYLOAD IS glTF, loaded with three's own GLTFLoader. One format serves
// both consumers -- this page and Blender -- so there is no second exporter to
// keep honest. The files carry km, z up, no transform node (glTF declares y up,
// so a Blender import needs one rotation); our camera is z up already, which is
// why nothing here rotates anything.
//
// RENDER ON DEMAND, like the rest of the site: a lighting rig does not need an
// idle loop, and fault.ts:206 records why leaving one running is a bad idea on
// a phone.

import {
  ACESFilmicToneMapping, AmbientLight, Box3, BufferGeometry, Color,
  DirectionalLight, Float32BufferAttribute, Group, LineBasicMaterial,
  LineSegments, Mesh, MeshStandardMaterial, PerspectiveCamera, PMREMGenerator,
  Scene, SRGBColorSpace, Vector3, WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export interface IsoSurface {
  scene: string; kernel: string; scalar: string; level_tag: string;
  level_GPa: number; level_MPa: number; file: string;
  triangles: number; vertices: number; triangles_before_decimation: number;
  decimation_deviation_km: number; level_rel_error: number;
  bytes: number; sha256: string;
}

export interface IsoManifest {
  schema: number; axis_up: string; units: string; eps_km: number;
  surfaces: IsoSurface[];
  scenes: Record<string, {
    grid: { dims: [number, number, number]; spacing: number;
            origin: number[]; rotation: number[][] };
    spacing_over_eps: number; crack: string; n_facets: number;
    centre: number[]; max_triangles: number; level_percentiles: number[];
  }>;
  receiver: { normal: number[]; slip_direction: number[]; friction: number;
              sign: string };
}

export type Look = "cinematic" | "neutral";
export type Preset = "oblique" | "strike" | "cross" | "top";

/** Art direction, and the neutral view that exists to check it.
 *
 * The guide this page follows asks for a companion scientific view, and the
 * reason is not politeness: a grazing key light and a warm rim are chosen to
 * make form readable, and the same choices can invent a crease or hide one. The
 * neutral look is the same geometry under flat white light with no rim, so the
 * reader can see what the lighting is doing. */
const LOOKS: Record<Look, {
  background: number; env: boolean; ambient: number;
  key: number; fill: number; rim: number;
  iso?: number; crack: number; lines: number; furniture: number;
  exposure: number;
}> = {
  // `lines` is the crack's own wireframe, which lies ON a lit surface and
  // should stay subordinate to it. `furniture` is the scale bar, which is
  // annotation and has to read against the BACKGROUND -- at `lines` on the
  // dark look it is about 2.5:1 against #0b0d12 and effectively invisible.
  cinematic: { background: 0x0b0d12, env: true, ambient: 0.25, key: 2.1,
               fill: 0.75, rim: 2.4, crack: 0x1a1c20, lines: 0x4a5058,
               furniture: 0x9aa2aa, exposure: 1.15 },
  neutral: { background: 0xf7f7f6, env: false, ambient: 1.4, key: 1.6,
             fill: 0.9, rim: 0.0, iso: 0xb9bcc0, crack: 0x6f7378,
             lines: 0x9aa0a6, furniture: 0x44484c, exposure: 1.0 },
};

export class IsoScene {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private loader = new GLTFLoader();
  private cache = new Map<string, Group>();
  private content = new Group();      // everything the camera should frame
  private isos = new Group();
  private crack = new Group();
  private furniture = new Group();    // axes and scale bar
  private lights: Record<string, DirectionalLight | AmbientLight> = {};
  private env: ReturnType<PMREMGenerator["fromScene"]> | null = null;
  private look: Look = "cinematic";
  private base: string;
  private el: HTMLElement;
  private raf = 0;
  private dir = new Vector3(1.0, -1.35, 0.85);
  /** Scale-bar length in km, so the page can say what the bar means. */
  barKm = 0;

  constructor(el: HTMLElement, readonly manifest: IsoManifest, baseUrl: string) {
    this.el = el;
    this.base = baseUrl.replace(/\/?$/, "/") + "data/";
    this.camera = new PerspectiveCamera(36, 1, 0.01, 1e4);
    this.camera.up.set(0, 0, 1);         // the model's vertical is +z

    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.toneMapping = ACESFilmicToneMapping;
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    const pmrem = new PMREMGenerator(this.renderer);
    this.env = pmrem.fromScene(new RoomEnvironment(), 0.04);

    this.lights.ambient = new AmbientLight(0xffffff, 1);
    // Key high and to one side, fill cool from the other, rim low and behind:
    // the rim is what separates an isosurface's silhouette from the crack
    // behind it, which on a dark background is the whole composition.
    this.lights.key = new DirectionalLight(0xfff3e2, 1);
    this.lights.key.position.set(0.6, -1.0, 1.2);
    this.lights.fill = new DirectionalLight(0xcfe0ff, 1);
    this.lights.fill.position.set(-1.0, 0.5, 0.25);
    this.lights.rim = new DirectionalLight(0xffd9a8, 1);
    this.lights.rim.position.set(-0.4, 1.0, -0.8);
    for (const l of Object.values(this.lights)) this.scene.add(l);

    this.content.add(this.isos, this.crack);
    this.scene.add(this.content, this.furniture);
    this.setLook("cinematic");
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  // --- content -------------------------------------------------------------
  private async load(file: string): Promise<Group> {
    const hit = this.cache.get(file);
    if (hit) return hit;
    const gltf = await this.loader.loadAsync(this.base + file);
    const g = gltf.scene;
    g.traverse((o) => {
      const m = (o as Mesh).material as MeshStandardMaterial | undefined;
      if (m) {
        m.envMapIntensity = 0.55;
        (o as Mesh).userData.baseColor = m.color.clone();
      }
    });
    this.cache.set(file, g);
    return g;
  }

  /** Show exactly these surface files, plus this scene's crack. */
  async show(files: string[], crackFile: string | null): Promise<void> {
    const want = await Promise.all(files.map((f) => this.load(f)));
    this.isos.clear();
    for (const g of want) this.isos.add(g);
    this.crack.clear();
    if (crackFile) this.crack.add(await this.load(crackFile));
    this.applyLook();
    this.buildFurniture();
    this.render();
  }

  setCrack(on: boolean) { this.crack.visible = on; this.render(); }

  // --- look ----------------------------------------------------------------
  setLook(look: Look) {
    this.look = look;
    this.applyLook();
    this.render();
  }

  private applyLook() {
    const L = LOOKS[this.look];
    this.scene.background = new Color(L.background);
    this.scene.environment = L.env && this.env ? this.env.texture : null;
    this.renderer.toneMappingExposure = L.exposure;
    (this.lights.ambient as AmbientLight).intensity = L.ambient;
    (this.lights.key as DirectionalLight).intensity = L.key;
    (this.lights.fill as DirectionalLight).intensity = L.fill;
    (this.lights.rim as DirectionalLight).intensity = L.rim;
    const paint = (g: Group, neutral: number | undefined) => {
      g.traverse((o) => {
        const m = (o as Mesh).material as MeshStandardMaterial | undefined;
        if (!m) return;
        const base = (o as Mesh).userData.baseColor as Color | undefined;
        if (neutral === undefined) {
          if (base) m.color.copy(base);
        } else {
          m.color.setHex(neutral, SRGBColorSpace);
        }
        m.envMapIntensity = L.env ? 0.55 : 0.0;
      });
    };
    paint(this.isos, L.iso);
    paint(this.crack, L.crack);
    for (const o of this.furniture.children) {
      const m = (o as LineSegments).material as LineBasicMaterial;
      if (m && !(o as LineSegments).userData.keepColor)
        m.color.setHex(L.furniture);
    }
  }

  // --- axes and scale bar --------------------------------------------------
  /** A triad and a round scale bar, parked at the content's lower corner.
   *
   * No 3-D text: a label in the scene would need a font or a canvas texture,
   * and the page can simply say how long the bar is. The triad keeps its own
   * colours in both looks, because an axis that changes colour with the
   * lighting is not an axis. */
  private buildFurniture() {
    this.furniture.clear();
    const box = new Box3().setFromObject(this.content);
    if (box.isEmpty()) return;
    const size = box.getSize(new Vector3());
    const span = Math.max(size.x, size.y, size.z);
    const raw = span / 4;
    const p = 10 ** Math.floor(Math.log10(raw));
    this.barKm = [1, 2, 5, 10].map((m) => m * p)
      .reduce((a, b) => (Math.abs(b - raw) < Math.abs(a - raw) ? b : a));
    const o = new Vector3(box.min.x, box.min.y, box.min.z);
    const seg = (pts: number[], color: number, keep = false) => {
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(pts, 3));
      const l = new LineSegments(g, new LineBasicMaterial({ color }));
      l.userData.keepColor = keep;
      this.furniture.add(l);
    };
    const a = 0.12 * span;
    seg([o.x, o.y, o.z, o.x + a, o.y, o.z], 0xd4573f, true);
    seg([o.x, o.y, o.z, o.x, o.y + a, o.z], 0x55a35a, true);
    seg([o.x, o.y, o.z, o.x, o.y, o.z + a], 0x4a7fc1, true);
    const y = o.y - 0.04 * span;
    seg([o.x, y, o.z, o.x + this.barKm, y, o.z,
         o.x, y - 0.01 * span, o.z, o.x, y + 0.01 * span, o.z,
         o.x + this.barKm, y - 0.01 * span, o.z,
         o.x + this.barKm, y + 0.01 * span, o.z],
        LOOKS[this.look].furniture);
  }

  // --- camera --------------------------------------------------------------
  preset(name: Preset) {
    const n = this.manifest.receiver.normal;
    // Horizontal strike of the receiver plane, which is the crack's mean plane:
    // the same (-n_y, n_x, 0) the solver's own frame uses.
    const strike = new Vector3(-n[1], n[0], 0).normalize();
    const cross = new Vector3().crossVectors(new Vector3(0, 0, 1), strike);
    this.dir = {
      oblique: new Vector3(1.0, -1.35, 0.85),
      strike: strike.clone(),
      cross: cross.clone(),
      top: new Vector3(0, 0, 1),
    }[name].clone().normalize();
    this.fit();
  }

  /** Frame the content by pushing the camera back until every corner clears.
   *
   * Corner by corner rather than by bounding sphere: a sphere around a slab
   * that is 1.6 km across and 0.6 km thick wastes most of the frame on empty
   * half space, which is exactly the complaint fractal.ts records. */
  fit() {
    const box = new Box3().setFromObject(this.content);
    if (box.isEmpty()) return;
    const c = box.getCenter(new Vector3());
    const right = new Vector3().crossVectors(new Vector3(0, 0, 1), this.dir);
    if (right.lengthSq() < 1e-9) right.set(1, 0, 0);
    right.normalize();
    const up = new Vector3().crossVectors(this.dir, right).normalize();
    const vt = Math.tan((this.camera.fov * Math.PI) / 360);
    const ht = vt * Math.max(this.camera.aspect, 1e-6);
    let d = 0, r = 0;
    for (const sx of [box.min.x, box.max.x]) {
      for (const sy of [box.min.y, box.max.y]) {
        for (const sz of [box.min.z, box.max.z]) {
          const v = new Vector3(sx, sy, sz).sub(c);
          r = Math.max(r, v.length());
          d = Math.max(d, v.dot(this.dir) + Math.abs(v.dot(up)) / vt,
                       v.dot(this.dir) + Math.abs(v.dot(right)) / ht);
        }
      }
    }
    d *= 1.08;
    this.camera.position.copy(c).addScaledVector(this.dir, d);
    this.controls.target.copy(c);
    this.camera.near = Math.max(0.01, d - 3 * r);
    this.camera.far = d + 3 * r;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }

  /** A PNG of exactly what is on screen, rendered in this tick.
   *
   * Rendering immediately before the read is what makes `preserveDrawingBuffer`
   * unnecessary -- the buffer is only guaranteed between a draw and the end of
   * the frame, and asking for it to persist costs every frame to serve one. */
  screenshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  resize() {
    const w = this.el.clientWidth || 640;
    const h = this.el.clientHeight || Math.round(w * 0.62);
    this.renderer.setSize(w, h, true);   // updateStyle true; see scene.ts:391
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  start() {
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      if (this.controls.update()) this.renderer.render(this.scene, this.camera);
    };
    this.controls.addEventListener("start", () => { if (!this.raf) tick(); });
    this.controls.addEventListener("end", () => {
      cancelAnimationFrame(this.raf); this.raf = 0; this.render();
    });
    this.render();
  }

  render() { this.renderer.render(this.scene, this.camera); }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.controls.dispose();
    this.env?.texture.dispose();
    this.renderer.dispose();
  }
}

/** Labels, kept beside the manifest's own keys rather than in the markup. */
export const SCALARS: Record<string, { label: string; unit: string }> = {
  von_mises: { label: "von Mises", unit: "MPa" },
  diff_stress: { label: "differential, σ₁ − σ₃", unit: "MPa" },
  dcfs: { label: "ΔCFS on the mean plane", unit: "MPa" },
};

const KERNELS: Record<string, string> = {
  mollified: "mollified (mhs)", classical: "classical TDE (cutde)",
};

export async function startIso(root: HTMLElement, baseUrl: string,
                               manifestName = "iso_manifest.json") {
  const canvasEl = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;
  const legend = root.querySelector<HTMLElement>("[data-legend]");
  const base = baseUrl.replace(/\/?$/, "/") + "data/";

  let man: IsoManifest;
  try {
    man = await (await fetch(base + manifestName)).json();
  } catch (e) {
    status.textContent = `Could not load the surfaces: ${e}`;
    return;
  }

  const scenes = Object.keys(man.scenes);
  let scene = scenes.includes("envelope") ? "envelope" : scenes[0];
  let scalar = "von_mises";
  let kernel = "mollified";
  let level = "";
  let crackOn = true;

  const view = new IsoScene(canvasEl, man, baseUrl);
  view.start();

  const pick = () => man.surfaces.filter(
    (s) => s.scene === scene && s.scalar === scalar && s.kernel === kernel);

  const sel = (name: string, opts: [string, string][], value: string,
               onChange: (v: string) => void) => {
    const wrap = root.querySelector<HTMLElement>(`[data-ctl="${name}"]`);
    if (!wrap) return null;
    wrap.querySelector("select")?.remove();
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

  function levelOptions(): [string, string][] {
    const seen = new Map<string, string>();
    for (const s of man.surfaces.filter(
      (q) => q.scene === scene && q.scalar === scalar)) {
      const u = SCALARS[s.scalar]?.unit ?? "";
      seen.set(s.level_tag, `${s.level_MPa.toPrecision(3)} ${u}`);
    }
    return [...seen.entries()];
  }

  sel("scene", scenes.map((s) => [s, s]), scene, (v) => {
    scene = v; rebuildLevels(); void refresh(true);
  });
  sel("scalar", Object.keys(SCALARS).map((k) => [k, SCALARS[k].label]), scalar,
      (v) => { scalar = v; rebuildLevels(); void refresh(false); });
  sel("kernel", Object.keys(KERNELS).map((k) => [k, KERNELS[k]]), kernel,
      (v) => { kernel = v; void refresh(false); });

  function rebuildLevels() {
    const opts = levelOptions();
    if (!opts.some(([v]) => v === level)) level = opts[0]?.[0] ?? "";
    sel("level", opts, level, (v) => { level = v; void refresh(false); });
  }
  rebuildLevels();

  const crackWrap = root.querySelector<HTMLElement>(`[data-ctl="crack"]`);
  if (crackWrap) {
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = true;
    box.addEventListener("change", () => {
      crackOn = box.checked; view.setCrack(crackOn);
    });
    crackWrap.append(box);
  }

  sel("look", [["cinematic", "cinematic"], ["neutral", "neutral (checking)"]],
      "cinematic", (v) => view.setLook(v as Look));

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
      a.download = `mhs-${scene}-${scalar}-${kernel}-${level}.png`;
      a.click();
    });
    shotWrap.appendChild(b);
  }

  const pubWrap = root.querySelector<HTMLElement>(`[data-ctl="publication"]`);
  if (pubWrap) {
    const box = document.createElement("input");
    box.type = "checkbox";
    box.addEventListener("change", () => {
      root.classList.toggle("publication", box.checked);
      requestAnimationFrame(() => view.resize());
    });
    pubWrap.append(box);
  }

  async function refresh(reframe: boolean) {
    const want = pick().filter((s) => s.level_tag === level);
    if (!want.length) { status.textContent = "no surface for that choice"; return; }
    status.textContent = "loading…";
    try {
      await view.show(want.map((s) => s.file), man.scenes[scene].crack);
      view.setCrack(crackOn);
      if (reframe) view.preset("oblique");
      const s = want[0];
      const sc = man.scenes[scene];
      status.textContent =
        `${KERNELS[kernel]} — ${SCALARS[scalar].label} at ` +
        `${s.level_MPa.toPrecision(3)} MPa`;
      if (legend) {
        const bits = [
          `level set of ${SCALARS[scalar].label} at ` +
          `${s.level_MPa.toPrecision(3)} MPa`,
          `${s.triangles.toLocaleString()} triangles`,
          `grid ${sc.grid.spacing} km = ${sc.spacing_over_eps.toFixed(1)} eps`,
          `scale bar ${view.barKm} km`,
          `axes x red, y green, z blue (z up, depth negative)`,
        ];
        if (scalar === "dcfs") {
          bits.splice(1, 0, `receiver: the crack's mean plane, friction ` +
            `${man.receiver.friction}, ${man.receiver.sign}`);
        }
        legend.textContent = bits.join(" · ");
      }
    } catch (e) {
      status.textContent = `failed: ${e}`;
    }
  }

  await refresh(true);
  return view;
}
