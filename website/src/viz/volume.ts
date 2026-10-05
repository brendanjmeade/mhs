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
}

export class Payload {
  readonly base: string;
  manifest!: Manifest;
  geometry: any = null;
  private cache = new Map<string, Uint8Array>();

  constructor(baseUrl: string) {
    // BASE_URL, never a leading "/data": a hardcoded absolute path works in
    // dev and 404s under the /mhs base in production.
    this.base = baseUrl.replace(/\/?$/, "/") + "data/";
  }

  async load(): Promise<Manifest> {
    this.manifest = await (await fetch(this.base + "manifest.json")).json();
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
    return [this.decode(key, 1), this.decode(key, 255)];
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

/** Human labels for the payload's state keys. */
export function stateLabel(key: string): string {
  if (key.startsWith("diff_")) {
    const m = key.match(/^diff_(.+)_minus_(.+)$/);
    if (m) return `${pretty(m[1])} \u2212 ${pretty(m[2])}`;
  }
  return pretty(key);
}

function pretty(k: string): string {
  return { half: "half space", full: "full space" }[k] ?? k;
}
