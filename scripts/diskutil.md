# GUIでは見えていないボリュームを削除する
- GUIだとUSBが2MBくらいしか見えていなかった (実際は64GB)
- GUIでのフォーマットは「パーティションマップを変更出来ませんでした」と失敗する
- ` diskutil eraseVolume [Format] [FormatName] [Identifier]`
	- ex: `diskutil eraseVolume exFAT nozaki /dev/disk4`
