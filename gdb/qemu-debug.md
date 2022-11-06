B
# qemu上で動いているプログラムをgdbデバッグする
- qdbserver側 + 操作側で動かす

## gdbserver側
- `$(QEMU) -S -gdb tcp::1234`
- -S: gdbserverへのコネクションがあるまで実行停止

## client側
- `gdb`
- `target remote localhost:1234`
- `symbol-file $(FILENAME)`
- 普通のgdb操作が続く
- 起動するgdbはQEMUで動かしているゲストのアーキテクチャでないと上手く動かなかった
- qemu-multiarch使えば良さそう
