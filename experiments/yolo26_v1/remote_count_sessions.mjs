import fs from "node:fs"; import path from "node:path"; import zlib from "node:zlib";
const ROOT = path.join(process.env.HOME, ".dsh", "sessions");
let total = 0;
for (const dir of fs.readdirSync(ROOT).sort()) {
  const d = path.join(ROOT, dir);
  if (!fs.statSync(d).isDirectory()) continue;
  const entries = fs.readdirSync(d);
  const sessions = entries.filter((e) => e.startsWith("session-"));
  let cwds = new Set();
  for (const s of sessions) {
    const f = path.join(d, s, "session.jsonl.zstd");
    if (!fs.existsSync(f)) continue;
    try { const o = JSON.parse(zlib.zstdDecompressSync(fs.readFileSync(f)).toString("utf8").split("\n")[0]); cwds.add(o.cwd); } catch (e) { cwds.add("ERR:" + e.message); }
  }
  total += sessions.length;
  console.log(dir + "  session_dirs=" + sessions.length + "  entries=" + entries.length + "  cwd=" + JSON.stringify([...cwds]));
}
console.log("TOTAL_SESSION_DIRS=" + total);