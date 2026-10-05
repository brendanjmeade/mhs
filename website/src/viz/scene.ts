// Three orthogonal slices through the volume, in a rotatable 3-D scene.
//
// Each slice is one quad whose fragment shader samples a 3-D texture at the
// fragment's own position in the box. That is why the slice planes and the
// colour map are instant: they are uniforms, and nothing is recomputed on the
// CPU when a slider moves. Rebuilding a 2-D texture per slice would also have
// worked at this grid size, but it puts the controls on the wrong side of a
// 61x61 loop.
//
// Two textures, not one packed RGBA: `region` is a CATEGORICAL code and must
// use a NEAREST sampler, because linearly interpolating it invents regions that
// do not exist between the real ones. Filtering is per-sampler, not per-channel,
// so it cannot share a texture with the fields.
//
// uint8 everywhere and never FloatType: OES_texture_float_linear is at 54 % on
// iOS, and a float texture with LinearFilter there is texture-incomplete and
// samples as black with no error raised.

import {
  BufferGeometry, Color, Data3DTexture, DataTexture, DoubleSide,
  Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh,
  NearestFilter, LinearFilter, PerspectiveCamera, PlaneGeometry, RedFormat,
  RGBAFormat, Scene, ShaderMaterial, UnsignedByteType, Vector3, WebGLRenderer,
  ClampToEdgeWrapping, GLSL3,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { lut, type MapName } from "./colormaps";
import type { Payload } from "./volume";

const VERT = `
varying vec3 vBox;
uniform vec3 uOrigin, uExtent;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vBox = (world.xyz - uOrigin) / uExtent;        // [0,1]^3 in the sampled box
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

// Declaring our own output is REQUIRED, not stylistic. three.js compiles every
// ShaderMaterial as "#version 300 es", but it injects
// `layout(location=0) out highp vec4 pc_fragColor` and the `gl_FragColor` alias
// only when glslVersion !== GLSL3 -- setting GLSL3 explicitly is the signal that
// the shader declares its own. Writing gl_FragColor here fails to compile with
// "cannot convert from 'highp 4-component vector of float' to 'const highp
// float'", which no type check or build step sees.
const FRAG = `
precision highp float;
precision highp sampler3D;
layout(location = 0) out vec4 fragColor;
varying vec3 vBox;
uniform sampler3D uField, uRegion;
uniform sampler2D uLut;
uniform vec3  uDims;           // (nx, ny, nz), for the texel-centre mapping
uniform bool  uShowOutside;
void main() {
  if (any(lessThan(vBox, vec3(0.0))) || any(greaterThan(vBox, vec3(1.0)))) discard;
  // vBox puts GRID POINT i at i/(n-1); a TEXEL CENTRE is at (i+0.5)/n. Sampling
  // at vBox directly stretches the volume by (n-1)/n -- zero error in the
  // middle, half a texel at each end, which is why the artefact appeared at the
  // extremes of every axis and nowhere else.
  vec3 uvw = (vBox * (uDims - 1.0) + 0.5) / uDims;
  float reg = texture(uRegion, uvw).r * 255.0;
  if (reg < 0.5 && !uShowOutside) discard;       // not in the body at all
  float f = texture(uField, uvw).r;
  if (f <= 0.0) discard;                          // level 0 is "no data"
  // EVERY point inside the half space is drawn, including the ones nearest the
  // fault where the field is largest. Nothing is masked out: there is no
  // near-boundary exclusion to make here, because the free surface is carried
  // analytically by the image terms rather than by a discretized boundary --
  // which is the whole claim this picture exists to show.
  vec3 c = texture(uLut, vec2(f, 0.5)).rgb;
  fragColor = vec4(c, 1.0);
}`;

/** One-voxel dilation of valid data into the no-data shell.
 *
 * A LINEAR sampler blends a boundary texel with its neighbours, and a no-data
 * neighbour is level 0 -- the bottom of the colour map. Without this, every
 * surface of the body is fringed with values that are not small, they are
 * ABSENT, rendered as though they were the minimum. The region mask still
 * decides what is drawn; this only stops the interpolator reading zeros.
 */
function dilate(data: Uint8Array, d: [number, number, number]): Uint8Array {
  const [nx, ny, nz] = d;
  const out = data.slice();
  const at = (x: number, y: number, z: number) => (z * ny + y) * nx + x;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const i = at(x, y, z);
        if (data[i] !== 0) continue;
        let sum = 0, n = 0;
        for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                    [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const X = x + dx, Y = y + dy, Z = z + dz;
          if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
          const v = data[at(X, Y, Z)];
          if (v !== 0) { sum += v; n++; }
        }
        if (n) out[i] = Math.round(sum / n);
      }
  return out;
}

function tex3d(data: Uint8Array, d: [number, number, number], smooth: boolean) {
  const t = new Data3DTexture(data, d[0], d[1], d[2]);
  t.format = RedFormat;
  t.type = UnsignedByteType;
  t.minFilter = t.magFilter = smooth ? LinearFilter : NearestFilter;
  t.wrapS = t.wrapT = t.wrapR = ClampToEdgeWrapping;
  t.unpackAlignment = 1;    // 53 and 103 are odd; the default 4 shears the rows
  t.needsUpdate = true;
  return t;
}

export interface SliceState { x: number; y: number; z: number; }

export class VolumeScene {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private mat: ShaderMaterial;
  private planes: Mesh[] = [];
  private overlays = new Group();
  private p: Payload;
  private el: HTMLElement;
  private raf = 0;

  constructor(el: HTMLElement, p: Payload) {
    this.el = el;
    this.p = p;
    const [ex, ey, ez] = p.extent;
    const o = p.origin;

    this.scene.background = new Color(0xffffff);
    this.camera = new PerspectiveCamera(38, 1, 1, 1e4);
    // The model's vertical is +z (depth is negative z); three.js defaults to
    // +y up, which lays the box on its side and stands the free surface
    // vertically. OrbitControls reads this, so it must be set before them.
    this.camera.up.set(0, 0, 1);
    this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(o[0] + ex / 2, o[1] + ey / 2, o[2] + ez / 2);
    this.camera.position.set(o[0] + ex * 1.35, o[1] - ey * 1.05,
                            o[2] + ez * 1.9);
    // Orient the camera at the target NOW. OrbitControls only recomputes the
    // camera from its spherical state inside update(), and update() runs in
    // the rAF loop, which only starts on interaction -- so without this the
    // first frame is drawn with the camera's default orientation, pointing
    // away from the box, and the canvas is simply blank until you drag it.
    this.controls.update();

    const lutTex = new DataTexture(lut("viridis"), 256, 1, RGBAFormat);
    lutTex.minFilter = lutTex.magFilter = LinearFilter;
    lutTex.needsUpdate = true;

    this.mat = new ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, side: DoubleSide,
      glslVersion: GLSL3,
      uniforms: {
        uOrigin: { value: new Vector3(o[0], o[1], o[2]) },
        uExtent: { value: new Vector3(ex, ey, ez) },
        uDims: { value: new Vector3(p.dims[0], p.dims[1], p.dims[2]) },
        uField: { value: null }, uRegion: { value: null },
        uLut: { value: lutTex }, uShowOutside: { value: false },
      },
    });

    for (let axis = 0; axis < 3; axis++) {
      const g = new PlaneGeometry(1, 1);
      const m = new Mesh(g, this.mat);
      this.planes.push(m);
      this.scene.add(m);
    }
    this.scene.add(this.overlays);
    this.buildOverlays();
    this.setSlices({ x: 0.5, y: 0.5, z: 0.72 });
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  /** Box edges, the free surface and the fault sheet, from geometry.json. */
  private buildOverlays() {
    const o = this.p.origin, [ex, ey, ez] = this.p.extent;
    this.overlays.add(new LineSegments(
      edgesOf(o[0], o[1], o[2], ex, ey, ez),
      new LineBasicMaterial({ color: 0xbbbbbb })));

    const g = this.p.geometry;
    if (!g) return;

    // The free surface, as a grid at z = 0. The sampled box deliberately
    // reaches ABOVE z = 0 and those voxels are no-data, so the body visibly
    // ends partway up the box -- without this the reader sees it stop at a
    // plane with nothing saying what the plane is.
    if (g.free_surface) {
      this.overlays.add(planeGrid(o[0], o[1], ex, ey, g.free_surface.z ?? 0));
    }

    // The fault as its four CORNERS, closed into a loop. A dipping plane's
    // bounding box is a solid volume; drawing that would put a wireframe box
    // in the half space and read as a second body rather than as the source.
    if (g.fault?.outline) {
      const q = g.fault.outline as number[][];
      const pos: number[] = [];
      for (let i = 0; i < q.length; i++) {
        const a = q[i], b = q[(i + 1) % q.length];
        pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
      const geo = new BufferGeometry();
      geo.setAttribute("position", new Float32BufferAttribute(pos, 3));
      this.overlays.add(new LineSegments(
        geo, new LineBasicMaterial({ color: 0xb5367a })));
    }
  }

  setTextures(field: Uint8Array, region: Uint8Array) {
    const d = this.p.dims;
    const u = this.mat.uniforms;
    (u.uField.value as any)?.dispose?.();
    (u.uRegion.value as any)?.dispose?.();
    u.uField.value = tex3d(dilate(field, d), d, true);
    u.uRegion.value = tex3d(region, d, false);   // NEAREST: categorical, never dilated
    this.render();
  }

  setColormap(name: MapName) {
    const t = new DataTexture(lut(name), 256, 1, RGBAFormat);
    t.minFilter = t.magFilter = LinearFilter;
    t.needsUpdate = true;
    (this.mat.uniforms.uLut.value as DataTexture).dispose();
    this.mat.uniforms.uLut.value = t;
    this.render();
  }

  /** Fractional slice positions in [0, 1]. */
  setSlices(s: SliceState) {
    const o = this.p.origin, [ex, ey, ez] = this.p.extent;
    const cx = o[0] + ex / 2, cy = o[1] + ey / 2, cz = o[2] + ez / 2;
    // yz plane at x
    this.planes[0].geometry.dispose();
    this.planes[0].geometry = new PlaneGeometry(ey, ez);
    this.planes[0].rotation.set(0, Math.PI / 2, Math.PI / 2);
    this.planes[0].position.set(o[0] + s.x * ex, cy, cz);
    // xz plane at y
    this.planes[1].geometry.dispose();
    this.planes[1].geometry = new PlaneGeometry(ex, ez);
    this.planes[1].rotation.set(Math.PI / 2, 0, 0);
    this.planes[1].position.set(cx, o[1] + s.y * ey, cz);
    // xy plane at z
    this.planes[2].geometry.dispose();
    this.planes[2].geometry = new PlaneGeometry(ex, ey);
    this.planes[2].rotation.set(0, 0, 0);
    this.planes[2].position.set(cx, cy, o[2] + s.z * ez);
    this.render();
  }

  setVisible(axis: 0 | 1 | 2, on: boolean) {
    this.planes[axis].visible = on;
    this.render();
  }

  resize() {
    const w = this.el.clientWidth || 640;
    const h = this.el.clientHeight || Math.round(w * 0.62);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  /** Damped orbit needs a loop while the pointer is down; otherwise on demand. */
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

  /** The orbit centre, for a caller driving the camera itself. */
  get target(): Vector3 { return this.controls.target; }

  render() { this.renderer.render(this.scene, this.camera); }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

function edgesOf(x: number, y: number, z: number,
                 w: number, h: number, d: number): BufferGeometry {
  const c: number[][] = [];
  for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) {
    c.push([x + i * w, y + j * h, z + k * d]);
  }
  const idx = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6],
               [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
  const pos: number[] = [];
  for (const [a, b] of idx) pos.push(...c[a], ...c[b]);
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  return g;
}

/** A flat wireframe at height `z`, marking the free surface. */
function planeGrid(x0: number, y0: number, w: number, h: number,
                   z: number, n = 12): LineSegments {
  const pos: number[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    pos.push(x0, y0 + t * h, z, x0 + w, y0 + t * h, z);
    pos.push(x0 + t * w, y0, z, x0 + t * w, y0 + h, z);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  return new LineSegments(g, new LineBasicMaterial({
    color: 0x999999, transparent: true, opacity: 0.45 }));
}
