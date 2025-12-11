# vAccel + unikernel

[Hardware acceleration for Unikernels A status update of vAccel](https://archive.fosdem.org/2023/schedule/event/hwacceluk/attachments/slides/5890/export/events/attachments/hwacceluk/slides/5890/hardware_acceleration_unikernels.pdf)

FOSDEM 2023


- [cloudkernels/vaccel](https://github.com/cloudkernels/vaccel) : meta repository for vAccel components
- [nubificus/vaccel](https://github.com/nubificus/vaccel)


## unikernel + GPU
- passthrough：巨大なドライバーが必要
- para-virtualization (virtio-gpu)：2Dグラフィックス向け、汎用計算には向かない
- 高レベルなAPIでホストに委譲：vAccel

## vAccelで unikernel + GPU
RPC versionとvirtio versionがあり、ここではvirtio versionの情報をまとめる
- vAccel用のvirtioデバイスを経由してGuest <-> Host間でAPIコールをやり取りする
  - そのためQEMUにパッチを当てる必要あり
- Guest applicationは事前に定義されているvAccel APIを呼び出す (ex. `vaccel_image_classification`)
  - unikraftではioctl経由
- GuestでvAccel APIを検知し、virtio-vaccel経由でVMMにリクエストを送信
- VMMはvirtio-vaccelコマンドを受け取り、ホストのランタイムを呼び出す
  - パッチ済みのVMMではvaccel-rt (vaccel runtime)をリンクしている

## コードリーディング
- QEMU vAccel patch
  - [commits](https://github.com/nubificus/qemu-vaccel/commits/master%2Bvaccel/)


## 環境構築

- [Linux Guest](https://github.com/nubificus/qemu-x86-build/blob/master/Dockerfile): Linux Guest + vAccel-enabled QEMU + vAccel host runtime, [tutorial](https://github.com/nubificus/vaccel-tutorials/tree/main/lab5#run-an-application-on-guest)
- [Unikraft Guest](https://github.com/cloudkernels/unikraft_vaccel_examples/blob/main/Dockerfile)