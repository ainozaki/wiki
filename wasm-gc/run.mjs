// run.mjs
import { readFileSync } from "node:fs";

// Get arguments
const args = Deno.args;
const enableGC = args[0] === "gc";
const fileName = enableGC ? "mandelbot_gc.wasm" : "mandelbot.wasm";

const wasmBuffer = readFileSync(fileName);
const wasmModule = await WebAssembly.instantiate(wasmBuffer);
const { Mandelbrot} = wasmModule.instance.exports;

const nx = 100;
const ny = 100;
const xmin = -2.0;
const ymin = -2.0;
const xmax = 2.0;
const ymax = 2.0;
const dx = (xmax - xmin) / nx;
const dy = (ymax - ymin) / ny;

Mandelbrot(xmin, ymin, dx, nx);

/*
for (let y = 0; y < ny; y++) {
    const cy = ymin + y * dy;
    for (let x = 0; x < nx; x++) {
        const cx = xmin + x * dx;
        const res = Mandelbrot(cx, cy);
        if (res == 0) {
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode(" "));
        } else {
            Deno.writeAllSync(Deno.stdout, new TextEncoder().encode("*"));
        }
    }
    console.log();
}
*/

if (enableGC) {
    console.log("============= Mandelbrot with GC-enabled Wasm =============");
}else {
    console.log("============= Mandelbrot with Core Wasm =============");
}