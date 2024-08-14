## Run Wasm GC-enabled Wat
```
$ deno --version
deno 1.45.5 (release, x86_64-unknown-linux-gnu)
v8 12.7.224.13
typescript 5.5.2

$ deno run --allow-read run.mjs
```

## Tips
- [binarygen](https://github.com/WebAssembly/binaryen) supports Wasm GC

```
wasm-as ./test.wat
```

