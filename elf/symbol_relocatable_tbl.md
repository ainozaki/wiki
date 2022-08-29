# ELF

```
ubuntu@ubuntu:~/ELF-loader$ cat misc/hello.c
#include <stdio.h>

static int i0 = 1;

void helloworld(){
	printf("Hello world2!\n");
}

int main(){
	helloworld();
	helloworld();
	return 0;
}
```

## シンボルテーブル
- オブジェクトファイルが他のオブジェクトファイルからリンクされるためには、含まれる関数・global変数のリストが必要
- .symtabセクション
- 構造体の配列
- nm: シンボルの情報を表示
- `readelf -s <file>`
```
ubuntu@ubuntu:~/ELF-loader$ nm misc/hello.o
0000000000000000 T helloworld
0000000000000000 d i0
0000000000000020 T main
                 U puts
```

## 再配置テーブル
- .rel.XXX, .rela.XXXセクション
- readelf -r
```
ubuntu@ubuntu:~/ELF-loader$ readelf -r misc/hello.o

Relocation section '.rela.text' at offset 0x2d8 contains 5 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000000008  000700000113 R_AARCH64_ADR_PRE 0000000000000000 .rodata + 0
00000000000c  000700000115 R_AARCH64_ADD_ABS 0000000000000000 .rodata + 0
000000000010  000f0000011b R_AARCH64_CALL26  0000000000000000 puts + 0
000000000028  000e0000011b R_AARCH64_CALL26  0000000000000000 helloworld + 0
00000000002c  000e0000011b R_AARCH64_CALL26  0000000000000000 helloworld + 0

Relocation section '.rela.eh_frame' at offset 0x350 contains 2 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
00000000001c  000200000105 R_AARCH64_PREL32  0000000000000000 .text + 0
00000000003c  000200000105 R_AARCH64_PREL32  0000000000000000 .text + 20
```
