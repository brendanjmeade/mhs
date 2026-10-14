// Wiring for the Examples viewer: state, field, colour map, slices.
//
// EVERY point in the half space is drawn, always -- including those nearest the
// fault, where the field is largest and a quadrature scheme would be most
// strained. Nothing is masked: the states differ only in whether the free
// surface is present, so a hidden region would hide exactly the comparison the
// page is for.

import { cssGradient, type MapName } from "./colormaps";
import { VolumeScene } from "./scene";
import { decadeTicks, FIELDS, Payload, stateLabel } from "./volume";

export async function startViewer(root: HTMLElement, baseUrl: string,
                                  manifestName = "manifest.json") {
  const canvasEl = root.querySelector<HTMLElement>("[data-canvas]")!;
  const status = root.querySelector<HTMLElement>("[data-status]")!;

  const p = new Payload(baseUrl, manifestName);
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
  const want = p.manifest.defaults ?? {};
  let state = want.state && states.includes(want.state) ? want.state
    : states.includes("half") ? "half"
    : (states.find((s) => !s.startsWith("diff_")) ?? states[0]);
  let field = want.field && fields.includes(want.field) ? want.field
    : fields.includes("von_mises") ? "von_mises" : fields[0];
  let map: MapName = "viridis";
  // Decades of the colour ramp to show, counted down from the array's own
  // maximum. 0 is all of it, and stays 0 rather than becoming a number so that
  // "show everything" survives a switch to an array with a different span.
  let decades = 0;

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

  // --- the displayed window -----------------------------------------------
  const decWrap = root.querySelector<HTMLElement>(`[data-ctl="decades"]`);
  let decIn: HTMLInputElement | null = null;
  let decOut: HTMLElement | null = null;
  if (decWrap) {
    decIn = document.createElement("input");
    decIn.type = "range"; decIn.min = "0.4"; decIn.step = "0.1";
    decOut = document.createElement("span");
    decOut.className = "val";
    decIn.addEventListener("input", () => {
      const v = Number(decIn!.value);
      // The top of the slider is "all of it", kept as 0 rather than as that
      // array's span, so switching to an array with more decades still shows
      // all of them instead of silently clipping to the previous one's.
      decades = v >= Number(decIn!.max) - 1e-9 ? 0 : v;
      applyWindow();
    });
    decWrap.append(decIn, decOut);
  }

  /** The window, in decades, for whatever array is loaded now. */
  function shown(key: string): number {
    const sp = p.decades(key);
    return decades > 0 ? Math.min(decades, sp) : sp;
  }

  function applyWindow() {
    const key = p.manifest.states[state].fields[field];
    const sp = p.decades(key), d = shown(key);
    if (decIn) {
      // The range input snaps to min + k*step, so a max of the array's exact
      // span (3.04) would make the largest REACHABLE value 3.0 and the full
      // span unselectable. Round the top up onto the grid instead; `shown`
      // clamps it back to the span.
      decIn.max = (Math.ceil(sp * 10) / 10).toFixed(1);   // before the value
      decIn.value = decades > 0 ? String(Math.min(decades, Number(decIn.max)))
                                : decIn.max;
      decIn.disabled = !(sp > 0);
    }
    if (decOut) {
      decOut.textContent = !(sp > 0) ? "linear"
        : decades > 0 ? `${d.toFixed(1)} of ${sp.toFixed(1)} dec`
        : `all ${sp.toFixed(1)} dec`;
    }
    scene.setWindow(p.window(key, d));
    paintBar();
  }

  const crackWrap = root.querySelector<HTMLElement>(`[data-ctl="crack"]`);
  if (crackWrap && scene.hasCrack) {
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = true;
    box.addEventListener("change", () => scene.setCrack(box.checked));
    crackWrap.append(box);
  }

  const slices = { x: 0.5, y: 0.5, z: 0.72, ...(want.slices ?? {}) };
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
    const w = p.window(key, shown(key));
    const f = FIELDS[field] ?? { label: field, unit: "", scale: 1 };
    // In the units the bar LABELS, so the decade ticks are round numbers of
    // MPa rather than of the solve's GPa.
    const lo = w.loValue * f.scale, hi = w.hiValue * f.scale;
    bar.innerHTML = "";
    const strip = document.createElement("div");
    strip.className = "strip";
    strip.style.background = cssGradient(map);
    // Ticks at whole powers of ten: the one thing that makes a log axis
    // readable as a log axis, rather than as two end labels and a promise.
    const ticks = document.createElement("div");
    ticks.className = "ticks";
    for (const { t, exp } of decadeTicks(lo, hi)) {
      const s = document.createElement("span");
      s.style.left = `${(t * 100).toFixed(2)}%`;
      s.innerHTML = exp === 0 ? "1"
        : `10<sup>${exp < 0 ? "\u2212" : ""}${Math.abs(exp)}</sup>`;
      ticks.appendChild(s);
    }
    const d = shown(key);
    const labels = document.createElement("div");
    labels.className = "ends";
    labels.innerHTML =
      `<span>${fmt(lo)}</span>` +
      `<span class="mid">log<sub>10</sub> ${f.label}` +
      `${f.unit ? ` (${f.unit})` : ""}` +
      `${d > 0 ? `, ${d.toFixed(1)} decades` : ""}</span>` +
      `<span>${fmt(hi)}</span>`;
    bar.append(strip, ticks, labels);
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
      applyWindow();
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
