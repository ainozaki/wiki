# QEMUでtrace logを見る方法

QEMUにはトレース機能があり、ソースコードには既にトレースポイントで関数呼び出しが組み込まれている。

- トレース機能はデフォルトでdisableなので、enableしてビルドしなおす

	```-
	./configure -enable-trace-backend=stderr..
	```

- トレースポイントは各ディレクトリの`trace-events`ファイルに一覧されている
- フックするトレースポイントをファイルに書き出す
	- ファイル名は任意
	```
	$cat trace_list
	virtio*
	```
- 上記のファイルを指定してQEMUを起動するとログが見える
	```
	`qemu-system-x86_64 ... --trace events=trace_list
	```
