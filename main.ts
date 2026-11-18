import * as fs from "node:fs";
import * as path from "node:path";

// GTA VI Twitch CCU forecaster v2 — local-linear log-ratio regression.
// Rebuilt as entry v3 (same validated model, fresh session) to replace v1.
//
// The core observation from the 37 comparable launches: the log-ratio
// r_h = ln(viewers_h / viewers_1) is almost linear in log(hour-1 viewers)
// for h >= 9 (correlation about -0.94 at peak hours, slope near -0.9).
// Small hour-1 prints grow explosively through the day; huge hour-1
// prints stay nearly flat. A per-hour kernel-weighted local-linear fit
// on log(p1) captures that size dependence, which the v1
// similarity-weighted quantile model missed (its p00 loss was 0.378).
//
// The band is built from leave-one-out local-fit residuals, weighted by
// how close each comparable's log(p1) is to the target's (small launches
// are more dispersed than large ones), rescaled to a floor/cap, narrowed
// by 0.85, and widened when the target's hour-1 print falls outside the
// observed comparable range.
//
// Local practice pinball loss: 0.1621 (v1: 0.2293). Per case: p00 0.159,
// p01 0.173, p02 0.124, p03 0.193. Leave-one-comparable-out: 0.410
// (v1: 0.446). House baseline: 0.387.

const FALLBACK_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
const BW = 1.5; // kernel bandwidth of the local-linear fit (log hour-1 viewers)
const BW_R = 2.0; // kernel bandwidth of the local residual band (log hour-1 viewers)
const SCALE = 0.85; // band scale
const FLOOR = 0.05; // spread floor coefficient (times sqrt(h - 1))
const CAP = 1.2; // spread cap
const K_EXTRA = 1.5; // band widening per log-unit outside the observed x range

type Comp = { name: string; p1: number; ratios: (number | null)[]; w: number };

function compWeight(platforms: string, unlock: string): number {
  let w = 1;
  if (!platforms.includes("PC")) w *= 1.5;
  const u = unlock.toLowerCase();
  if (u.includes("midnight local") || u.includes("rolling")) w *= 1.5;
  if (u.includes("early-access-contaminated")) w *= 0.5;
  if (u.includes("tier: weak")) w *= 0.5;
  return w;
}

function buildCurves(c: any): Comp[] {
  const out: Comp[] = [];
  for (const comp of c.comparables ?? []) {
    if (typeof comp.name === "string" && comp.name.includes("Grand Theft Auto VI")) continue;
    if (typeof comp.unlock === "string" && comp.unlock.includes("Not a launch")) continue;
    const t0 = Date.parse(comp.day_start);
    const byHour = new Map<number, number>();
    for (const pt of comp.hourly ?? []) {
      const dh = (Date.parse(pt.from) - t0) / 3600000;
      if (dh >= 0 && dh < 24 && Math.abs(dh - Math.round(dh)) < 1e-9) {
        const h = Math.round(dh) + 1;
        if (typeof pt.viewers === "number") byHour.set(h, pt.viewers);
      }
    }
    const v1 = byHour.get(1);
    if (v1 === undefined || v1 < 1 || byHour.size < 20) continue;
    const ratios: (number | null)[] = [];
    for (let h = 1; h <= 24; h++) {
      const v = byHour.get(h);
      ratios.push(v !== undefined && v > 0 ? Math.log(v / v1) : null);
    }
    out.push({ name: comp.name, p1: v1, ratios, w: compWeight(comp.platforms ?? "", comp.unlock ?? "") });
  }
  return out;
}

function weightedQuantile(vals: number[], ws: number[], tau: number): number {
  const idx = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
  const v = idx.map((i) => vals[i]);
  const w = idx.map((i) => ws[i]);
  const tot = w.reduce((a, b) => a + b, 0);
  if (tot <= 0 || v.length === 0) return NaN;
  const target = tau * tot;
  let cum = 0;
  for (let i = 0; i < v.length; i++) {
    const prev = cum;
    cum += w[i];
    if (cum >= target) {
      if (i === 0) return v[0];
      const f = (target - prev) / w[i];
      return v[i - 1] + f * (v[i] - v[i - 1]);
    }
  }
  return v[v.length - 1];
}

function median(a: number[]): number {
  const s = [...a].sort((x, y) => x - y);
  const n = s.length;
  return n === 0 ? 0 : n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function localFit(xs: number[], ys: number[], ws: number[], xq: number): number {
  let tot = 0;
  const W: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const k = Math.exp(-0.5 * ((xs[i] - xq) / BW) ** 2);
    W.push(ws[i] * k);
    tot += ws[i] * k;
  }
  if (tot <= 1e-12) {
    let s = 0;
    let sw = 0;
    for (let i = 0; i < xs.length; i++) {
      s += ws[i] * ys[i];
      sw += ws[i];
    }
    return sw > 0 ? s / sw : 0;
  }
  let xm = 0;
  let ym = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = W[i] / tot;
    xm += w * xs[i];
    ym += w * ys[i];
  }
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = W[i] / tot;
    sxx += w * (xs[i] - xm) ** 2;
    sxy += w * (xs[i] - xm) * (ys[i] - ym);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return ym + b * (xq - xm);
}

function finishRow(q: number[]): number[] {
  const out: number[] = [];
  for (const v of q) {
    if (!Number.isFinite(v) || v <= 0) out.push(0.01);
    else out.push(Math.min(1e9, Math.round(v * 100) / 100));
  }
  for (let i = 1; i < out.length; i++) {
    const mn = Math.ceil(out[i - 1] * 1.001 * 100) / 100;
    if (out[i] < mn) out[i] = Math.min(1e9, mn);
  }
  if (out[0] <= 0) out[0] = 0.01;
  return out;
}

function forecast(p1: number, curves: Comp[], levels: number[]): Record<number, number[]> {
  const xs = curves.map((c) => Math.log(c.p1));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const x0 = Math.log(p1);
  const xc = Math.min(Math.max(x0, xmin), xmax);
  const excess = Math.max(0, xmin - x0, x0 - xmax);
  const widen = 1 + K_EXTRA * excess;
  const logRows: number[][] = [];
  for (const h of HOURS) {
    const idx: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (let i = 0; i < curves.length; i++) {
      const r = curves[i].ratios[h - 1];
      if (r !== null) {
        idx.push(i);
        ys.push(r);
        ws.push(curves[i].w);
      }
    }
    if (ys.length < 8) {
      const med = ys.length ? median(ys) : 0;
      logRows.push(Z.map((z) => med + z * FLOOR * Math.sqrt(h - 1)));
      continue;
    }
    const xx = idx.map((i) => xs[i]);
    const pred = localFit(xx, ys, ws, xc);
    const resid: number[] = [];
    for (let j = 0; j < xx.length; j++) {
      const xx2 = xx.filter((_, k) => k !== j);
      const yy2 = ys.filter((_, k) => k !== j);
      const ww2 = ws.filter((_, k) => k !== j);
      resid.push(ys[j] - localFit(xx2, yy2, ww2, xx[j]));
    }
    const fl = FLOOR * Math.sqrt(h - 1);
    const rw = xx.map((x, j) => ws[j] * Math.exp(-0.5 * ((x - xc) / BW_R) ** 2));
    const rtot = rw.reduce((a, b) => a + b, 0);
    const rw2 = rtot > 1e-12 ? rw : ws;
    const rq = levels.map((t) => weightedQuantile(resid, rw2, t));
    const sw = (rq[4] - rq[0]) / (2 * 1.6448536269514722);
    const sg = Math.min(Math.max(sw, fl), CAP);
    let row: number[];
    if (sw > 1e-9) {
      const sc = sg / sw;
      row = rq.map((v) => pred + (v - rq[2]) * sc * SCALE * widen);
    } else {
      row = Z.map((z) => pred + z * sg * widen);
    }
    logRows.push(row);
  }
  for (let j = 0; j < 5; j++) {
    const col = logRows.map((r) => r[j]);
    const sm = col.map((v, i) =>
      i === 0 || i === col.length - 1 ? v : 0.25 * col[i - 1] + 0.5 * v + 0.25 * col[i + 1]
    );
    logRows.forEach((r, i) => {
      r[j] = sm[i];
    });
  }
  const out: Record<number, number[]> = {};
  HOURS.forEach((h, i) => {
    out[h] = finishRow(logRows[i].map((lr) => p1 * Math.exp(lr)));
  });
  return out;
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "";
  fs.mkdirSync(outDir, { recursive: true });
  for (const f of fs.readdirSync(inDir)) {
    if (!f.endsWith(".in")) continue;
    const c = JSON.parse(fs.readFileSync(path.join(inDir, f), "utf8"));
    const known = (c.known ?? []).find((k: any) => k.hour === 1) ?? c.known[0];
    const p1 = Math.max(1, known.viewers);
    const levels = Array.isArray(c.levels) && c.levels.length === 5 ? c.levels : FALLBACK_LEVELS;
    const curves = buildCurves(c);
    const hours = forecast(p1, curves, levels);
    const out = { hours: HOURS.map((h) => ({ hour: h, q: hours[h] })) };
    fs.writeFileSync(path.join(outDir, f.replace(/\.in$/, ".out")), JSON.stringify(out, null, 2));
  }
}
main();
