// Rewrite the local Windows workspace paths inside imported DSH session transcripts so that
// header.cwd (and path references) match the remote workspace. Originals stay in a backup dir.
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const HOME = process.env.HOME;
const ROOT = path.join(HOME, ".dsh", "sessions");
// escaped form first (JSONL stores a literal backslash as \\), then the plain form
const PAIRS = [
  ["E:\\\\具身智能\\\\badmin_project", "/home/T7/ojh/badmin_project"],
  ["E:\\\\具身智能", "/home/T7/ojh"],
  ["D:\\\\_eth_data\\\\eth_shuttle_detection", "/home/T7/dgut/robot_sim/eth_shuttle_detection"],
  ["E:\\具身智能\\badmin_project", "/home/T7/ojh/badmin_project"],
  ["E:\\具身智能", "/home/T7/ojh"],
];

let files = 0, changed = 0, totalHits = 0;
for (const dir of fs.readdirSync(ROOT)) {
  const d = path.join(ROOT, dir);
  if (!fs.statSync(d).isDirectory()) continue;
  for (const s of fs.readdirSync(d)) {
    const f = path.join(d, s, "session.jsonl.zstd");
    if (!fs.existsSync(f)) continue;
    files++;
    const text = zlib.zstdDecompressSync(fs.readFileSync(f)).toString("utf8");
    let out = text, hits = 0;
    for (const [from, to] of PAIRS) {
      const n = out.split(from).length - 1;
      if (n > 0) { out = out.split(from).join(to); hits += n; }
    }
    const first = out.split("\n")[0];
    let cwd = "(unparsed)";
    try { cwd = JSON.parse(first).cwd; } catch (e) { cwd = "PARSE_ERROR " + e.message; }
    if (hits > 0) {
      fs.writeFileSync(f, zlib.zstdCompressSync(Buffer.from(out, "utf8")));
      changed++; totalHits += hits;
    }
    console.log([dir, s, "hits=" + hits, "cwd=" + cwd].join("  "));
  }
}
console.log("FILES=" + files + " CHANGED=" + changed + " TOTAL_REPLACEMENTS=" + totalHits);