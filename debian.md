# Debianのインストール
- InstallのGUI画面が表示されるが,しま模様になり文字は読み取れない
	- BIOSでCSM (Compatibility Support Mode) を無効にすると治った
	- [参考](https://www.linuxquestions.org/questions/debian-26/network-auto-configuration-failed-duing-debian-installation-4175527717/)
- Realtekのドライバが存在しない
	- [見つからないファームウエアの読み込み](https://www.debian.org/releases/stable/i386/ch06s04.ja.html)
	- 公式インストールイメージには non-free のファームウェアが含まれないため
	- non-free のファームウェアを含む非公式インストールイメージを使うとよいらしい
		- 今回はUSBアダプタ挿して対応した 
