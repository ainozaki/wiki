// run.mjs
import { readFileSync } from "node:fs";
const wasmBuffer = readFileSync("test.wasm");
const wasmModule = await WebAssembly.instantiate(wasmBuffer);
const { make, get } = wasmModule.instance.exports;
const o = make(42);
console.log(o, get(o));
