// run.mjs
import { readFileSync } from "node:fs";
const wasmBuffer = readFileSync("mandelbot.wasm");
const wasmModule = await WebAssembly.instantiate(wasmBuffer);
const { ifMandelbrotIncluded } = wasmModule.instance.exports;

const nx = 100;
const ny = 100;
const xmin = -2.0;
const ymin = -2.0;
const xmax = 2.0;
const ymax = 2.0;
const dx = (xmax - xmin) / nx;
const dy = (ymax - ymin) / ny;

for (let y = 0; y < ny; y++) {
    const cy = ymin + y * dy;
    for (let x = 0; x < nx; x++) {
        const cx = xmin + x * dx;
        const res = ifMandelbrotIncluded(cx, cy);
        //console.log(cx, cy, res);
        if (res == 0) {
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode(" "));
        } else {
            //Deno.writeAllSync(Deno.stdout, new TextEncoder().encode(res.toString()));
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode("*"));
        }
    }
    console.log();
}