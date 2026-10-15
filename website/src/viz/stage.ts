// The stage every 3-D scene on this site stands on: framing, camera presets,
// the axes-and-scale-bar furniture, on-demand rendering and screenshots.
//
// Lifted out of iso.ts when a second lit-free data scene (the on-fault page)
// needed the same five things. One copy, because the repo already carried two
// of the on-demand loop (scene.ts, iso.ts) and three of a free-surface grid, and
// a convention written twice is a bug waiting for the second copy to drift.
// Nothing here knows what it is framing: a caller passes a Box3 and a direction.

import {
  Box3, BufferGeometry, Float32BufferAttribute, Group, LineBasicMaterial,
  LineSegments, PerspectiveCamera, Scene, Vector3, WebGLRenderer,
} from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type Preset = "oblique" | "strike" | "cross" | "top";

/** The viewing direction of a named preset, for a reference plane with this
 * (upward) normal. Strike is the horizontal trace of that plane, the same
 * (-n_y, n_x, 0) the solver's own frame uses; cross is perpendicular to it. */
export function presetDir(name: Preset, normal: ArrayLike<number>): Vector3 {
  const strike = new Vector3(-normal[1], normal[0], 0).normalize();
  const cross = new Vector3().crossVectors(new Vector3(0, 0, 1), strike);
  return {
    oblique: new Vector3(1.0, -1.35, 0.85),
    strike, cross,
    top: new Vector3(0, 0, 1),
  }[name].clone().normalize();
}

/** Frame `box` by pushing the camera back along `dir` until every corner
 * clears the frustum, then point the controls at its centre.
 *
 * Corner by corner rather than by bounding sphere: a sphere around a slab that
 * is 1.6 km across and 0.6 km thick wastes most of the frame on empty half
 * space. Does not render; the caller decides when. */
export function fitCamera(camera: PerspectiveCamera, controls: OrbitControls,
                          box: Box3, dir: Vector3, pad = 1.08): void {
  const c = box.getCenter(new Vector3());
  const right = new Vector3().crossVectors(new Vector3(0, 0, 1), dir);
  if (right.lengthSq() < 1e-9) right.set(1, 0, 0);
  right.normalize();
  const up = new Vector3().crossVectors(dir, right).normalize();
  const vt = Math.tan((camera.fov * Math.PI) / 360);
  const ht = vt * Math.max(camera.aspect, 1e-6);
  let d = 0, r = 0;
  for (const sx of [box.min.x, box.max.x]) {
    for (const sy of [box.min.y, box.max.y]) {
      for (const sz of [box.min.z, box.max.z]) {
        const v = new Vector3(sx, sy, sz).sub(c);
        r = Math.max(r, v.length());
        d = Math.max(d, v.dot(dir) + Math.abs(v.dot(up)) / vt,
                     v.dot(dir) + Math.abs(v.dot(right)) / ht);
      }
    }
  }
  d *= pad;
  camera.position.copy(c).addScaledVector(dir, d);
  controls.target.copy(c);
  camera.near = Math.max(0.01, d - 3 * r);
  camera.far = d + 3 * r;
  camera.updateProjectionMatrix();
  controls.update();
}

/** Fill `into` with a triad and a round scale bar parked at the box's lower
 * corner, and return the bar's length in km so the page can say what it is.
 *
 * No 3-D text: a label in the scene would need a font or a canvas texture, and
 * the page can simply state the length. The triad keeps its own colours
 * (`userData.keepColor`) whatever the look, because an axis that changes colour
 * with the lighting is not an axis; the bar takes `colour`, which has to read
 * against the BACKGROUND rather than against any surface. */
export function buildFurniture(into: Group, box: Box3, colour: number): number {
  into.clear();
  const size = box.getSize(new Vector3());
  const span = Math.max(size.x, size.y, size.z);
  const raw = span / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const barKm = [1, 2, 5, 10].map((m) => m * p)
    .reduce((a, b) => (Math.abs(b - raw) < Math.abs(a - raw) ? b : a));
  const o = new Vector3(box.min.x, box.min.y, box.min.z);
  const seg = (pts: number[], color: number, keep = false) => {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(pts, 3));
    const l = new LineSegments(g, new LineBasicMaterial({ color }));
    l.userData.keepColor = keep;
    into.add(l);
  };
  const a = 0.12 * span;
  seg([o.x, o.y, o.z, o.x + a, o.y, o.z], 0xd4573f, true);
  seg([o.x, o.y, o.z, o.x, o.y + a, o.z], 0x55a35a, true);
  seg([o.x, o.y, o.z, o.x, o.y, o.z + a], 0x4a7fc1, true);
  const y = o.y - 0.04 * span;
  seg([o.x, y, o.z, o.x + barKm, y, o.z,
       o.x, y - 0.01 * span, o.z, o.x, y + 0.01 * span, o.z,
       o.x + barKm, y - 0.01 * span, o.z,
       o.x + barKm, y + 0.01 * span, o.z], colour);
  return barKm;
}

/** Recolour the furniture's bar (and anything else not marked keepColor). */
export function recolourFurniture(group: Group, colour: number): void {
  for (const o of group.children) {
    const m = (o as LineSegments).material as LineBasicMaterial;
    if (m && !(o as LineSegments).userData.keepColor) m.color.setHex(colour);
  }
}

/** Render on demand: a damped orbit needs a loop while the pointer is down,
 * and nothing otherwise. fault.ts:206 records why an idle loop is a bad idea
 * on a phone. Returns the function that stops the loop. */
export function onDemand(controls: OrbitControls, render: () => void): () => void {
  let raf = 0;
  const tick = () => {
    raf = requestAnimationFrame(tick);
    if (controls.update()) render();
  };
  controls.addEventListener("start", () => { if (!raf) tick(); });
  controls.addEventListener("end", () => {
    cancelAnimationFrame(raf); raf = 0; render();
  });
  render();
  return () => { cancelAnimationFrame(raf); raf = 0; };
}

/** A PNG of exactly what is on screen, rendered in this tick.
 *
 * Rendering immediately before the read is what makes `preserveDrawingBuffer`
 * unnecessary -- the buffer is only guaranteed between a draw and the end of
 * the frame, and asking for it to persist costs every frame to serve one. */
export function screenshot(renderer: WebGLRenderer, scene: Scene,
                           camera: PerspectiveCamera): string {
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL("image/png");
}

/** Every edge of a welded triangle mesh once, as line-segment positions. */
export function meshEdges(V: number[][], T: number[][]): BufferGeometry {
  const seen = new Set<number>();
  const lp: number[] = [];
  for (const t of T) {
    for (const [p, q] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
      const key = Math.min(p, q) * V.length + Math.max(p, q);
      if (seen.has(key)) continue;
      seen.add(key);
      lp.push(...V[p], ...V[q]);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(lp, 3));
  return g;
}
