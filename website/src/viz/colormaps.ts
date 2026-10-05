// 256-entry lookup tables, uploaded as a 1-D texture and sampled in the slice
// shader. The same two maps the Python figures use, so a reader moving between
// the paper figures and this page sees the same colours mean the same things.
//
// Control points are the matplotlib anchors, interpolated linearly in sRGB.
// That is not perceptually ideal, but it reproduces what matplotlib draws, and
// matching the figures matters more here than being marginally more uniform.

type RGB = [number, number, number];

const VIRIDIS: RGB[] = [
  [68, 1, 84], [71, 24, 106], [72, 41, 122], [69, 56, 132], [64, 70, 136],
  [58, 83, 139], [52, 95, 141], [47, 107, 142], [42, 118, 142], [38, 130, 142],
  [34, 141, 141], [31, 152, 139], [34, 164, 131], [47, 175, 123],
  [68, 186, 110], [94, 196, 94], [122, 205, 75], [155, 213, 55],
  [189, 219, 40], [223, 227, 41], [253, 231, 37],
];

// RdBu reversed: blue (low) -> white -> red (high), for signed differences.
const RDBU_R: RGB[] = [
  [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
  [209, 229, 240], [247, 247, 247], [253, 219, 199], [244, 165, 130],
  [214, 96, 77], [178, 24, 43], [103, 0, 31],
];

const PLASMA: RGB[] = [
  [13, 8, 135], [57, 4, 158], [92, 1, 166], [123, 3, 167], [152, 20, 157],
  [176, 42, 143], [197, 64, 128], [215, 87, 113], [230, 111, 98],
  [241, 137, 83], [249, 165, 68], [253, 194, 55], [252, 225, 56],
  [240, 249, 33],
];

export const MAPS = { viridis: VIRIDIS, plasma: PLASMA, rdbu_r: RDBU_R };
export type MapName = keyof typeof MAPS;

/** `(256 * 4)` RGBA bytes, alpha 255, for a `DataTexture`. */
export function lut(name: MapName): Uint8Array {
  const pts = MAPS[name];
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (pts.length - 1);
    const j = Math.min(Math.floor(t), pts.length - 2);
    const f = t - j;
    for (let c = 0; c < 3; c++) {
      out[i * 4 + c] = Math.round(pts[j][c] + f * (pts[j + 1][c] - pts[j][c]));
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** CSS gradient for an HTML colour bar, so the bar cannot drift from the LUT. */
export function cssGradient(name: MapName): string {
  const pts = MAPS[name];
  const stops = pts.map(
    (p, i) => `rgb(${p[0]},${p[1]},${p[2]}) ${((i / (pts.length - 1)) * 100).toFixed(1)}%`,
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
