## Run GC-enabled Wasm
```
$ deno --version
deno 1.45.5 (release, x86_64-unknown-linux-gnu)
v8 12.7.224.13
typescript 5.5.2


# Run GC-enabled Wasm
$ deno run --allow-read run.mjs gc

# Run non-GC Wasm
$ deno run --allow-read run.mjs
```

## Tips
- [binarygen](https://github.com/WebAssembly/binaryen) supports Wasm GC

```
wasm-as ./test.wat
```

