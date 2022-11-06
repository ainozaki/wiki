# GDBよく使うコマンド
- disassemble
	- `disas`
	- pc前後をdias
- メモリダンプ
	- `x/<format> <addr>`
	- xはexamine
	- format: 数字->繰り返し, {b|h|w|g}->サイズ, {x|d|i|...}->フォーマット
	- addr: $pcとか数値とか
	- `x/10i $pc`: pcから10個の命令を表示
	- `x/4xw $sp`: spから4個の32bit数値を表示
- define
	- 決まった処理を実行
	- 下参照
- バックトレース
	- `bt`
- print
	- `print *ポインタ変数` とするとポインタが指すアドレスの内容を表示
- スクリプトで処理
	- `gdb -batch -x $(FILENAME)`
	- ```
		define run-stepi
			si
			printf "QQQW0119  0x%lx 0x%lx 0x%lx\n", $w0, $w1, $w19
			x/i $pc
		end
		b main
		set verbose off
		start
		c
		while $x5!=1
			run-stepi
		end
		```
