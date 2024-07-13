# Compile LLVM to Wasm
```
$ ls
add.ll
$ llc-14 -march=wasm32 -filetype=obj add.ll
$ ls
add.ll  add.o
$ file add.o
add.o: WebAssembly (wasm) binary module version 0x1 (MVP)
```
