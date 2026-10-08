import zlib from "node:zlib";
console.log("node", process.version);
console.log("zstdDecompressSync:", typeof zlib.zstdDecompressSync);
console.log("zstdCompressSync:", typeof zlib.zstdCompressSync);
if (typeof zlib.zstdCompressSync === "function") {
  const b = zlib.zstdCompressSync(Buffer.from(JSON.stringify({ type: "session", id: "x", cwd: "/home/T7/ojh/badmin_project" }) + "\n"));
  const back = zlib.zstdDecompressSync(b).toString("utf8");
  console.log("roundtrip_ok:", back.includes("/home/T7/ojh/badmin_project"), "bytes:", b.length);
}