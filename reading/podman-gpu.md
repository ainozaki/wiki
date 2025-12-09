# PodmanからGPUを使う

[PodmanのコンテナからmacOSのApple Silicon GPUを使ってAIワークロードを高速処理できるようになりました](https://zenn.dev/orimanabu/articles/podman-libkrun-gpu)


構成
- Hypervisor.framework (KVM相当)
- libkrun (QEMU相当)

どうGPUを呼び出す？
- Guest applicationはVulkan APIを発行
- Guestで動くVenusがVulkan APIをvirtio-gpuコマンドに変換、virtio-gpu queueに投入
- VMMで動くvirglrenderがvirtio-gpu経由でコマンドを受け取り、Vulkanに変換
- MoltenVKがMetal APIに変換

- virtio-gpuはグラフィック専用だが、Vulkan + Venus + virtio-gpuでは `VIRTIO_GPU_CMD_SUBMIT_3D` のペイロードにVulkanコマンドを含めることで汎用計算を実現
- virglrendererのVulkanサポートは最近の話（2021年ごろ議論開始、2024年[QEMUでupstream](https://www.phoronix.com/news/VirtIO-GPU-Vulkan-QEMU)）
- VulkanはOpenGLの後継　(Khronos Group、ベンダ非依存)