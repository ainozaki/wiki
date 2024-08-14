// run.mjs
import { readFileSync } from "node:fs";
const wasmBuffer = readFileSync("mandelbot.wasm");
const wasmModule = await WebAssembly.instantiate(wasmBuffer);
const { ifMandelbrotIncluded } = wasmModule.instance.exports;

const nx = 10;
const ny = 10;
const xmin = -2.0;
const ymin = -2.0;
const xmax = 2.0;
const ymax = 2.0;
const dx = (xmax - xmin) / nx;
const dy = (ymax - ymin) / ny;

for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
        const cx = xmin + x * dx;
        const cy = ymin + y * dy;
        const res = ifMandelbrotIncluded(cx, cy);
        if (res === 0) {
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode(" "));
        } else {
            //Deno.writeAllSync(Deno.stdout, new TextEncoder().encode(res.toString()));
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode("*"));
        }
    }
    console.log();
}