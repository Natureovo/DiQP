# LDV 快速测试

本流程先取一个 LDV 视频的少量原始帧，使用“HEVC快速编码代码”压缩包中的 `HM16.3-standard.exe` 和 Low-Delay P 参数生成 HEVC 压缩数据，再验证或微调 DiQP。默认保留视频的原始帧序列，不强制改变帧率。

Python 已经替代 `cfg_hm16.3_LDP.exe` 生成配置，但所有有效编码参数都逐项取自压缩包生成的 `1.cfg`。编码仍必须调用压缩包原始 `HM16.3-standard.exe`；程序会核对 SHA-256，不允许其他 HM 版本冒充该协议。Linux 服务器需通过 Wine 运行这个 32 位 Windows EXE。

注意：这是项目指定的 HM 16.3 LDP 实验协议，并不是 DiQP 原论文的 FFmpeg/NVIDIA 编码协议，因此结果不能称为原论文同编码器复现。

## 0. 编码器文件

经项目维护者确认，压缩包内的编码器随代码仓库统一传输。服务器执行 `git pull` 后文件位于：

```text
/home/cp/桌面/yx/DiQP/tools/hm16_3/HM16.3-standard.exe
```

在服务器核对文件和 Wine；正确 EXE 的 SHA-256 必须是 `3243e12f542273733171d972b92e5a2389f43d58afcce636590cbaf32fae834d`：

```bash
sha256sum /home/cp/桌面/yx/DiQP/tools/hm16_3/HM16.3-standard.exe
```

```bash
wine --version
```

若 `wine` 不存在，需要先离线或通过系统软件源安装 Wine 及 32 位支持。`cfg_hm16.3_LDP.exe` 不必传到服务器，因为 Python 已实现同一配置生成功能。

## 1. 准备一小段数据

在服务器 DiQP 项目根目录执行：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv.py --input /home/cp/datasets/LDV1/training_raw/001.mkv --sequence 101 --qp 42 --frames 12
```

输出结构如下：

```text
data/Raw/101/000.png
data/Encoded/101/QP-42/000.png
data/Encoded/101/QP-42/qp_042_hm16_3_ldp.bin
data/Encoded/101/QP-42/hm16_3_ldp.cfg
data/Encoded/101/QP-42/hm16_3_ldp_encode.log
data/Encoded/101/QP-42/encoding_protocol.txt
```

默认编码器路径为项目内的 `tools/hm16_3/HM16.3-standard.exe`。输入尺寸按压缩包流程补齐到 8 的倍数；`960×536` 已满足要求，不再人为补到 `960×544`。临时输入和重建 YUV 放在 `/tmp/diqp_hm16_3` 的纯英文路径中，成功后自动删除，只保留 `.bin` 码流、cfg、日志、协议和 PNG。

HM 是 CPU 参考编码器，速度会明显慢于 `hevc_nvenc`。旧的服务器 HM Random Access 流程仍可用 `--encoder hm` 显式选择，但新实验必须保持默认 `hm16_3_ldp`，不能混入旧数据。

## 2. 快速测试

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 42 --fraction 0.25 --batch-size 1 --save-limit 6
```

结果写入 `report_v2.csv`。重点看 `PSNR Gain` 和 `SSIM Gain`：正数表示模型优于压缩输入，负数表示模型使结果变差。

`testResults/comparison_*.png` 从左到右是 Encoded、Model、Raw；`difference_x5_*.png` 左侧是 Encoded 误差，右侧是 Model 误差，亮度经过 5 倍放大。

## 3. 扩大测试

快速流程确认无报错后，可去掉抽样：

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 42 --fraction 1 --batch-size 1 --save-limit 12
```

需要测试其他 LDV 视频时使用新的序列号，例如将 `002.mkv` 对应到 `--sequence 102`，不要覆盖已有实验。

## 4. 多视频、多 QP 批量测试

先用已经准备好的序列做一次整帧冒烟测试，确认服务器环境、显存和拼接流程正常：

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 42 --fraction 1 --batch-size 1 --save-limit 3 --full-frame-metrics --report batchResults/fullframe_hm_smoke.csv --results-dir batchResults/fullframe_hm_smoke
```

`512×512` 是预训练模型固定的输入块大小，不是最终评价窗口。整帧模式会把所有重叠块加权拼回原始分辨率，并让每个视频帧只参与一次最终统计。整帧模式必须使用 `--fraction 1`，否则无法覆盖完整画面。

下面的命令自动测试前 5 个 LDV 视频，每个视频使用 HM 测试多 QP 集合 22、27、32、37、42、51；前五档是常用视频增强基准设置，QP 51 对应 DiQP 的 HEVC 最大强压缩档。默认将重叠的 512×512 预测加权拼回完整帧，再计算 RGB-PSNR、Y-PSNR 和 SSIM：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_ldv_batch.py --video-ids 1 2 3 4 5 --qps 22 27 32 37 42 51 --frames 60 --fraction 1 --batch-size 1 --save-limit 2
```

首次切换到 HM 16.3 LDP 前，可以删除旧协议帧数据；不要删除 `batchResults`、`runs`、`pretrained` 或 `modelone/dataset/hm_results`：

```bash
rm -rf -- /home/cp/桌面/yx/DiQP/data/Raw /home/cp/桌面/yx/DiQP/data/Encoded
```

旧的 `data/LDV_finetune` 是 NVENC 微调集。如果确定改用 HM 重新微调，可单独删除并重建；已有 checkpoint 和评估报告位于 `runs`、`batchResults`，不受影响：

```bash
rm -rf -- /home/cp/桌面/yx/DiQP/data/LDV_finetune
```

每次运行都会在 `batchResults/ldv_时间戳/` 下创建独立目录。`metrics.csv` 保存每个视频的完整帧结果，`summary.csv` 保存总体和分 QP 平均结果，`visuals/` 保存完整帧对比图和逐帧指标，`protocol.txt` 记录本次参数。单个组合失败时脚本默认记录到 `failures.csv` 并继续执行后续组合。

`--frames` 是最多抽取的帧数。如果某个源视频不足该数量，脚本会自动使用它的全部可用帧。批量任务中断或部分组合失败后，可以在原目录断点续跑；已有 `Sequence/QP` 指标会被跳过，只执行缺失组合并重新生成汇总：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_ldv_batch.py --video-ids 1 2 3 4 5 --qps 42 --frames 200 --fraction 1 --batch-size 1 --save-limit 2 --resume-run /home/cp/桌面/yx/DiQP/batchResults/ldv_20260901_093450
```

## 5. HM 多 QP 微调

训练集固定使用 LDV 021–030，验证集使用 031–033；测试集保留 001–020，三者不重叠。默认准备 QP 22、27、32、37、42、51，每个视频 120 帧，并支持通过 `.prepared` 标记断点续做：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv_finetune.py --qps 22 27 32 37 42 51 --frames 120 --encoder hm16_3_ldp
```

准备完成后先进行两步 decoder 冒烟训练：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 1 --train-scope decoder --max-steps 2 --val-fraction 0.02 --run-name smoke_decoder_hm16_3_ldp_multiqp
```

冒烟测试通过后再执行正式训练：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 5 --train-scope decoder --val-fraction 0.1 --run-name decoder_hm16_3_ldp_multiqp
```

`history.csv` 保存全部 QP 的综合验证结果，`validation_by_qp.csv` 保存每个 QP 的独立 PSNR 增益。checkpoint 同时记录 HM 编码器和完整 QP 列表，不能误接到旧 NVENC 单 QP 训练任务上。

## 6. 直接评测已有 Raw/重建 YUV

如果数据已经包含成对的裸 YUV，例如 Raw 为 `{编号}.yuv`，QP 37 重建结果为 `{编号}_37rec.yuv`，可以跳过编码并直接评测。脚本会检查配对关系、文件大小和帧数；每次只把当前视频转换为临时 PNG，评测结束后自动清理，避免批量展开占用大量磁盘。

先用一个视频和 12 帧做冒烟测试：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_preencoded_yuv.py --raw-dir /media/cp/PHD3/DATASET/LDV_EX --encoded-dir /media/cp/PHD3/dataset_for_third_result/CQP/woLF/result_hevc_qp37_woLF_oneI --video-ids 100 --qp 37 --width 960 --height 536 --frames 12 --save-limit 3 --ffmpeg /usr/bin/ffmpeg
```

冒烟测试通过后评测 5 个完整视频：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_preencoded_yuv.py --raw-dir /media/cp/PHD3/DATASET/LDV_EX --encoded-dir /media/cp/PHD3/dataset_for_third_result/CQP/woLF/result_hevc_qp37_woLF_oneI --video-ids 100 101 102 103 104 --qp 37 --width 960 --height 536 --frames 120 --save-limit 2 --ffmpeg /usr/bin/ffmpeg
```

确认流程和指标正常后，可评测全部匹配视频：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_preencoded_yuv.py --raw-dir /media/cp/PHD3/DATASET/LDV_EX --encoded-dir /media/cp/PHD3/dataset_for_third_result/CQP/woLF/result_hevc_qp37_woLF_oneI --all-matched --qp 37 --width 960 --height 536 --frames 120 --save-limit 0 --ffmpeg /usr/bin/ffmpeg
```

结果保存在 `batchResults/preencoded_yuv/qp37_时间戳/`。`metrics.csv` 是逐视频指标，`summary.csv` 是总体汇总，`failures.csv` 记录失败项。默认模型输入是 `*_37rec.yuv`，对应的 `*_37str.bin` 不参与推理。目录名中的 `woLF` 和 `oneI` 表示特殊编码条件，这组结果应与标准 HM16.3 LDP 结果分开报告。

## 7. 在 woLF + oneI 数据上微调

如果最终实验目标固定为 QP 37、`woLF + oneI`，应直接使用这套重建 YUV 微调，不能混入标准 HM LDP 或 NVENC 数据。下面示例保留之前使用的测试视频，只把互不重叠的 30/5 个视频用于训练和验证：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_preencoded_finetune.py --raw-dir /media/cp/PHD3/DATASET/LDV_EX --encoded-dir /media/cp/PHD3/dataset_for_third_result/CQP/woLF/result_hevc_qp37_woLF_oneI --train-ids 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 30 31 32 --val-ids 33 34 35 36 37 --test-ids 20 40 60 80 100 101 102 103 104 120 140 160 180 200 220 261 267 274 --qp 37 --width 960 --height 536 --frames 120 --ffmpeg /usr/bin/ffmpeg
```

先做两步冒烟训练，再进行正式 decoder 微调：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --data-root data/LDV_woLF_oneI_qp37 --epochs 1 --train-scope decoder --max-steps 2 --val-fraction 0.02 --run-name smoke_decoder_woLF_oneI_qp37
```

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --data-root data/LDV_woLF_oneI_qp37 --epochs 5 --train-scope decoder --val-fraction 1 --run-name decoder_woLF_oneI_qp37
```

正式测试仍直接读取原始 YUV，并指定微调产生的最佳 checkpoint。测试 ID 必须保持与训练、验证集合互斥：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_preencoded_yuv.py --raw-dir /media/cp/PHD3/DATASET/LDV_EX --encoded-dir /media/cp/PHD3/dataset_for_third_result/CQP/woLF/result_hevc_qp37_woLF_oneI --video-ids 20 40 60 80 100 101 102 103 104 120 140 160 180 200 220 261 267 274 --qp 37 --width 960 --height 536 --frames 120 --save-limit 0 --ffmpeg /usr/bin/ffmpeg --model-path runs/ldv_finetune/decoder_woLF_oneI_qp37/best.pt
```

视频 240 的 YUV 大小与 `960x536 yuv420p` 不匹配，确认其真实分辨率前不要放入统一测试。最终至少同时报告原始预训练模型和微调模型相对于压缩重建帧的 RGB-PSNR、Y-PSNR 与 SSIM 增益。
