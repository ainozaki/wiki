# マイコンで推論

[Sim-to-Real実装：PyTorchで学習した強化学習モデルをONNX経由でSTM32にデプロイする](https://qiita.com/RamTuckey/items/80c8859bbb24d59bee95)
1. Stable Baselines 3 (SB3) で強化学習モデルを学習
1. ONNX形式でexport
1. STM32Cube.AIでONNXモデルをimport、Cコード生成
1. 呼び出しコードを追加してマイコンで実行
1. STM32Cube.AI Runtimeで実行

- STM32Cube.AI Runtime は CMSIS-NN (Cortex Microcontroller Software Interface Standard) を呼び出す
  - = ARM Cortex-M 向けNNライブラリ

## ONNX
- Protocol Buffers として定義されている
```
.onnx (protobuf)
 └─ Model
     └─ Graph
         ├─ NodeProto node
         ├─ TensorProto initializer
         └─ ValueInfoProto Input / Output

message NodeProto {
    repeated string input;
    repeated string output;
    string op_type;
    ...
}

# ValueInfoProtoとの違いは実データを持つこと
message TensorProto {
    repeated int64 dims;
    int32 data_type;
    repeated <T> T_data;
}

message ValueInfoProto {
    string name;
    TypeProto type;
    ...
}
```
- export
 - traceして一度forwardを実行することで、静的グラフを変換(default)
 - TorchScriptでグラフ解析 (制限あり)