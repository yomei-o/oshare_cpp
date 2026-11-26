// node で WASM を動かす: node tools/test_node.js weights.bin prompts.json image.rgba W H
const fs = require("fs");
const OshareModule = require("../docs/oshare.js");
(async () => {
  const [wf, pf, rf, W, H] = process.argv.slice(2);
  const M = await OshareModule();
  const put = (buf) => { const p = M._malloc(buf.length); M.HEAPU8.set(buf, p); return p; };
  const wb = fs.readFileSync(wf), pb = fs.readFileSync(pf), rb = fs.readFileSync(rf);
  let p = put(wb); console.log("weights", M._os_load_weights(p, wb.length)); M._free(p);
  p = put(pb); console.log("prompts", M._os_load_prompts(p, pb.length)); M._free(p);
  p = put(rb);
  const t0 = Date.now(); const r = M.UTF8ToString(M._os_judge(p, +W, +H));
  console.log(Date.now() - t0, "ms", r);
})();
