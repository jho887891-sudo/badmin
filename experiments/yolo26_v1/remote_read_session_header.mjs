// Read the header lines of imported DSH session transcripts (proof the remote tooling can parse them).
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
const home = process.env.HOME;
const root = path.join(home, ".dsh", "sessions");
function tryRead(file) {
  const buf = fs.readFileSync(file);
  const dec = (typeof zlib.zstdDecompressSync === "function") ? zlib.zstdDecompressSync(buf) : null;
  if (!dec) return { err: "no zstdDecompressSync in this node" };
  return { text: dec.toString("utf8") };
}
for (const dir of fs.readdirSync(root)) {
  const full = path.join(root, dir);
  const sessions = fs.readdirSync(full).filter((d) => d.startsWith("session-"));
  console.log("DIR " + dir + "  sessions=" + sessions.length);
  for (const s of sessions.slice(0, 2)) {
    const f = path.join(full, s, "session.jsonl.zstd");
    if (!fs.existsSync(f)) { console.log("   " + s + " (no session.jsonl.zstd)"); continue; }
    const r = tryRead(f);
    if (r.err) { console.log("   " + s + " ERR " + r.err); continue; }
    const first = r.text.split("\n")[0] || "";
    let o = null; try { o = JSON.parse(first); } catch (e) { o = { parse_error: String(e), raw: first.slice(0, 120) }; }
    const pick = {};
    for (const k of ["type", "sessionId", "id", "cwd", "workspace", "projectPath", "title", "createdAt", "model", "provider"]) if (k in o) pick[k] = o[k];
    console.log("   " + s + " header=" + JSON.stringify(pick).slice(0, 400));
  }
}