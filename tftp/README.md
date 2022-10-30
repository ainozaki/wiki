#tftp
- APのtftp bootのために用意
- apt install tftpd-hpa
- 設定は/etc/default/tftpd-hpa
	- このファイルはデフォルトで存在する
	- TFTP_DIRECTORYでアップロードされるディレクトリ
	- デフォルトはread-only
	- put可にするためにはTFTP_OPTIONの変更とディレクトリのアクセス権のchmodが必要
- iptablesが効いていて`Destination Unreachable(destination host administratively prohibited)`になっていた

