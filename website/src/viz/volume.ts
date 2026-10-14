// Loading the payload: the manifest, then one array at a time.
//
// Arrays are separate files and fetched ON DEMAND, so a visitor who never
// changes the state or the field downloads one of them rather than all nine.
// Already-fetched arrays are cached by key, and the key for `region` is a
// content hash -- here every state shares one mask, because the body is z <= 0
// and nothing about it varies between states, so it is fetched once.
//
// Everything is uint8. Not for size, but because a float32 3-D texture cannot
// be linearly filtered on roughly half of iOS devices, and it fails by
// rendering black with no error raised.

export interface Manifest {
  schema: number;
  grid: { dims: [number, number, number]; origin: [number, number, number];
          spacing: [number, number, number]; n_points: number };
  nodata: number;
  levels: number;
  source: { volume_run: string; sampled_from: string | null; git: string | null };
  states: Record<string, {
    kind: "state" | "difference";
    fields: Record<string, string>;
    region: string;
  }>;
  arrays: Record<string, {
    encoding: "log10_u8" | "linear_u8" | "codes_u8";
    min?: number; max?: number; log_min?: number; log_max?: number;
    bytes: number; sha256: string; values?: number[]; filter?: string;
  }>;
  geometry?: string;
  /** Optional opening view. A payload whose body is not the half space has no
   *  reason to open on the half-space state or on a slice chosen for a free
   *  surface, so it says what it wants instead of the page hardcoding it. */
  defaults?: { state?: string; field?: string;
               slices?: { x?: number; y?: number; z?: number } };
}

export class Payload {
  readonly base: string;
  manifest!: Manifest;
  geometry: any = null;
  private cache = new Map<string, Uint8Array>();

  constructor(baseUrl: string, readonly manifestName = "manifest.json") {
    // BASE_URL, never a leading "/data": a hardcoded absolute path works in
    // dev and 404s under the /mhs base in production.
    this.base = baseUrl.replace(/\/?$/, "/") + "data/";
  }

  async load(): Promise<Manifest> {
    this.manifest = await (await fetch(this.base + this.manifestName)).json();
    if (this.manifest.geometry) {
      this.geometry = await (await fetch(this.base + this.manifest.geometry)).json();
    }
    return this.manifest;
  }

  /** One array, cached. */
  async array(key: string): Promise<Uint8Array> {
    const hit = this.cache.get(key);
    if (hit) return hit;
    const r = await fetch(`${this.base}${key}.bin`);
    if (!r.ok) throw new Error(`${key}.bin: ${r.status}`);
    const a = new Uint8Array(await r.arrayBuffer());
    const want = this.manifest.grid.n_points;
    if (a.length !== want) {
      throw new Error(`${key}.bin is ${a.length} bytes, expected ${want}`);
    }
    this.cache.set(key, a);
    return a;
  }

  /** Physical value of a quantised level, for the colour bar's end labels. */
  decode(key: string, q: number): number {
    const m = this.manifest.arrays[key];
    if (q === this.manifest.nodata) return NaN;
    const t = (q - 1) / (this.manifest.levels - 1);
    if (m.encoding === "log10_u8") {
      return 10 ** (m.log_min! + t * (m.log_max! - m.log_min!));
    }
    if (m.encoding === "linear_u8") return m.min! + t * (m.max! - m.min!);
    return q;
  }

  range(key: string): [number, number] {
    // The TOP level is `levels`, not 255: the encoder writes 1..levels, so
    // asking for 255 decodes past log_max and overstates the bar's top end by
    // 10**(span/(levels-1)) -- 3.7 % on a four-decade array.
    return [this.decode(key, 1), this.decode(key, this.manifest.levels)];
  }

  /** How many decades an array covers. 0 if it is not on a log ramp. */
  decades(key: string): number {
    const m = this.manifest.arrays[key];
    if (!m || m.encoding !== "log10_u8") return 0;
    return Math.max(0, (m.log_max ?? 0) - (m.log_min ?? 0));
  }

  /** The displayed window as the TOP `d` decades: the physical ends, and the
   *  sampler bounds that select them.
   *
   *  THE ARRAYS ARE ALREADY log10 -- the byte is linear in log10 of the value,
   *  so the colour axis is log10 whatever this returns, and a window is not a
   *  second transform. All it moves is where the ramp ENDS, which is a real
   *  choice on a 1/r^3 field: most of a box this size is far field, so the full
   *  span spends colour on distance from the crack rather than on structure
   *  near it, while clipping saturates it. Both ways are visible as the ends
   *  saturating, which is why the default is the array's own span and the
   *  narrowing is the reader's -- on this payload a 2.5-decade window already
   *  flattens 28 % of the voxels onto one colour and a 2-decade one 70 %.
   *
   *  The bounds come back in the shader's own units -- the normalized uint8
   *  level, `level / 255` -- so the fragment shader stays one subtract and one
   *  divide and knows nothing about the encoding. */
  window(key: string, d: number): { lo: number; hi: number;
                                    loValue: number; hiValue: number } {
    const m = this.manifest.arrays[key];
    const L = this.manifest.levels;
    const hiLog = m?.log_max ?? 0, span = this.decades(key);
    const clipped = span > 0 && d > 0 && d < span;
    const t = clipped ? 1 - d / span : 0;        // fraction of the ramp cut off
    return { lo: (1 + t * (L - 1)) / 255, hi: L / 255,
             loValue: clipped ? 10 ** (hiLog - d) : this.decode(key, 1),
             hiValue: this.decode(key, L) };
  }

  get dims() { return this.manifest.grid.dims; }

  /** Physical size of the sampled box, km. */
  get extent(): [number, number, number] {
    const d = this.manifest.grid.dims, s = this.manifest.grid.spacing;
    return [(d[0] - 1) * s[0], (d[1] - 1) * s[1], (d[2] - 1) * s[2]];
  }

  get origin() { return this.manifest.grid.origin; }
}

/** Units and labels, so the colour bar says MPa and mm rather than 0-255. */
export const FIELDS: Record<string, { label: string; unit: string; scale: number }> = {
  // The solve is in km and GPa; nobody reads a surface displacement in km.
  u_mag: { label: "|u|", unit: "mm", scale: 1e6 },
  von_mises: { label: "von Mises", unit: "MPa", scale: 1e3 },
  max_shear: { label: "max shear", unit: "MPa", scale: 1e3 },
};

/** Integer decades inside a displayed window, as fractions along the bar.
 *
 *  `lo` and `hi` are the window's ends in the units the bar LABELS -- MPa, mm
 *  -- not the solve's, so the ticks land on round numbers a reader can check
 *  the picture against. A tick within `pad` of either end is dropped rather
 *  than drawn on top of the numeric ends. */
export function decadeTicks(lo: number, hi: number, pad = 0.07,
                            ): { t: number; exp: number }[] {
  if (!(lo > 0) || !(hi > lo)) return [];
  const a = Math.log10(lo), b = Math.log10(hi);
  const out: { t: number; exp: number }[] = [];
  for (let e = Math.ceil(a); e <= Math.floor(b); e++) {
    const t = (e - a) / (b - a);
    if (t > pad && t < 1 - pad) out.push({ t, exp: e });
  }
  return out;
}

/** Human labels for the payload's state keys. */
export function stateLabel(key: string): string {
  if (key.startsWith("diff_")) {
    const m = key.match(/^diff_(.+)_minus_(.+)$/);
    if (m) return `${pretty(m[1])} \u2212 ${pretty(m[2])}`;
  }
  return pretty(key);
}

function pretty(k: string): string {
  return { half: "half space", full: "full space",
           mollified: "mollified (mhs)",
           classical: "classical TDE (cutde)" }[k] ?? k;
}
