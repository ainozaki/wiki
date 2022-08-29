# section
- `readelf -S <file>`
- `readelf -x <section no>`
- リンクの単位
## 各メンバのメモ
- sh_name: 実態は.shstrtabにnull区切りで格納されている。sh_nameはそのindex(shを固定長にするため) 
- sh_type:
	- SHT_PROGBITS: .text, .data
	- SHT_REL, SHT_RELA: 再配置テーブル
	- SHT_NOBITS: ファイル中に実態はないがロード時にメモリ確保される領域 (.bss)
- sh_flags:
	- SHF_WRITE: 書き込み可(.data, .bss)
	- SHF_ALLOC: ロード時にメモリ確保(.bss, .data, .bss) シンボルテーブル・再配置テーブルなど実行に必要でないセクションには立たない
	- SHF_EXECINSTR: 実行可能

# segment
- `readelf -l <file>`
- ロードの単位
- リンカが複数のセクションを1つのセグメントにまとめる
## 各メンバのメモ
- p_type: 
	- PT_LOAD: メモリ上にロード
	- PT_DYNAMIC: ダイナミック・リンク用
	- PT_PHDR: プログラムヘッダ自体が1つのセグメント
- p_paddr: ロード先の物理アドレス。LMA(Load)!=VMAの場合（カーネルなど）に使われる。ロードも実行もvaddrで行われる通常のappでは使用しない
- p_memsz: 通常はp_fileszと等しいが、bssなどファイル中に実態が無いセクションが含まれる場合はその限りではない
- p_flags:
	- PF_X: 実行可
	- PF_W: 書き込み可
	- PF_R: 読み込み可
