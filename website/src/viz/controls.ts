// Wiring for the Examples viewer: state, field, colour map, slices.
//
// EVERY point in the half space is drawn, always -- including those nearest the
// fault, where the field is largest and a quadrature scheme would be most
// strained. Nothing is masked: the states differ only in whether the free
// surface is present, so a hidden region would hide exactly the comparison the
// page is for.

import { cssGradient, type MapName } from "./colormaps";
import { VolumeScene } from "./scene";
import { FIELDS, Payload, stateLabel } from "./volume";

export async function startViewer(root: HTMLElement, baseUrl: string) {
  const canvasEl = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;

  const p = new Payload(baseUrl);
  try {
    await p.load();
  } catch (e) {
    status.textContent = `Could not load the volume data: ${e}`;
    return;
  }

  const scene = new VolumeScene(canvasEl, p);
  scene.start();

  const states = Object.keys(p.manifest.states);
  const fields = Object.keys(p.manifest.states[states[0]].fields);
  let state = states.includes("half") ? "half"
    : (states.find((s) => !s.startsWith("diff_")) ?? states[0]);
  let field = fields.includes("von_mises") ? "von_mises" : fields[0];
  let map: MapName = "viridis";

  // --- controls -----------------------------------------------------------
  const sel = (name: string, opts: [string, string][], value: string,
               onChange: (v: string) => void) => {
    const wrap = root.querySelector<HTMLElement>(`[data-ctl="${name}"]`);
    if (!wrap) return;
    const s = document.createElement("select");
    for (const [v, label] of opts) {
      const o = document.createElement("option");
      o.value = v; o.textContent = label; o.selected = v === value;
      s.appendChild(o);
    }
    s.addEventListener("change", () => onChange(s.value));
    wrap.appendChild(s);
  };

  sel("state", states.map((s) => [s, stateLabel(s)]), state, (v) => {
    state = v; void refresh();
  });
  sel("field", fields.map((f) => [f, FIELDS[f]?.label ?? f]), field, (v) => {
    field = v; void refresh();
  });
  sel("map", [["viridis", "viridis"], ["plasma", "plasma"],
              ["rdbu_r", "red-blue"]], map, (v) => {
    map = v as MapName; scene.setColormap(map); paintBar();
  });

  const slices = { x: 0.5, y: 0.5, z: 0.72 };
  for (const axis of ["x", "y", "z"] as const) {
    const wrap = root.querySelector<HTMLElement>(`[data-ctl="slice-${axis}"]`);
    if (!wrap) continue;
    const r = document.createElement("input");
    r.type = "range"; r.min = "0"; r.max = "1"; r.step = "0.01";
    r.value = String(slices[axis]);
    const out = document.createElement("span");
    out.className = "val";
    const show = () => {
      const i = "xyz".indexOf(axis);
      const km = p.origin[i] + slices[axis] * p.extent[i];
      out.textContent = `${km.toFixed(0)} km`;
    };
    r.addEventListener("input", () => {
      slices[axis] = Number(r.value); scene.setSlices(slices); show();
    });
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = true;
    box.addEventListener("change", () =>
      scene.setVisible("xyz".indexOf(axis) as 0 | 1 | 2, box.checked));
    wrap.append(box, r, out);
    show();
  }

  // --- colour bar ---------------------------------------------------------
  const bar = root.querySelector<HTMLElement>("[data-bar]");
  function paintBar() {
    if (!bar) return;
    const key = p.manifest.states[state].fields[field];
    const [lo, hi] = p.range(key);
    const f = FIELDS[field] ?? { label: field, unit: "", scale: 1 };
    bar.innerHTML = "";
    const strip = document.createElement("div");
    strip.className = "strip";
    strip.style.background = cssGradient(map);
    const labels = document.createElement("div");
    labels.className = "ends";
    labels.innerHTML =
      `<span>${fmt(lo * f.scale)}</span>` +
      `<span class="mid">${f.label}${f.unit ? ` (${f.unit})` : ""}` +
      `, log scale</span>` +
      `<span>${fmt(hi * f.scale)}</span>`;
    bar.append(strip, labels);
  }

  // --- load ---------------------------------------------------------------
  async function refresh() {
    const st = p.manifest.states[state];
    const key = st.fields[field];
    if (!key) { status.textContent = `no ${field} for ${state}`; return; }
    status.textContent = "loading…";
    try {
      const [f, r] = await Promise.all([p.array(key), p.array(st.region)]);
      scene.setTextures(f, r);
      paintBar();
      const kind = st.kind === "difference"
        ? "what the free surface is worth" : "one evaluation";
      status.textContent = `${stateLabel(state)} — ${kind}`;
    } catch (e) {
      status.textContent = `failed: ${e}`;
    }
  }

  await refresh();
  return scene;
}

function fmt(v: number): string {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-2 || a >= 1e4) return v.toExponential(1);
  return v.toPrecision(3).replace(/\.?0+$/, "");
}
