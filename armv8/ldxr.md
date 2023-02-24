# [Aarch64] LDXR命令の後に続く命令がgdbでステップ実行されない

## 概要
- QEMU + gdbでのデバッグ中，LDXR命令の後に続く命令がステップ実行されない現象が生じた
- LDXR/STXRはアトミック命令であり，Exclusive Monitorでのマークを乱さないために，gdbは敢えて2命令間はステップ実行していない

## 現象
- LDXR命令のあと，pcが2命令分進んでいる

```
.global test_ldxr
test_ldxr:
        mov x0, #0
        ldxr w0, [x19]
        stxr w2, w1, [x19]
        mov x0, #1
        ret
```

```
```

## 解析
- gdbの表示には出てこない`stxr w2, w1, [x19]`の命令は実行されている


## 原因

### LDXR/STXR
LDXR(Load Exclusive Register), STXR(tore Exclusive Register)は単なるアトミック命令ではなく，2命令をペアで利用し，命令間に他のコアによる当該物理アドレスへのアクセスがなかったかを確認する命令である．
exclusie access