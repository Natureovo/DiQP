# LDV 快速测试

本流程先取一个 LDV 视频的少量原始帧，使用服务器已有的 HM `TAppEncoder` 和 `encoder_randomaccess_main.cfg` 生成标准 HEVC Random Access 压缩数据，再验证预训练 DiQP。默认保留视频的原始帧序列，不强制改变帧率。

注意：HM 是本项目当前采用的标准化对照协议，便于和 MFQE/STDF 类实验比较；DiQP 原论文的数据由 FFmpeg/NVIDIA 编码，因此 HM 结果不能直接称为原论文同编码器复现。

## 1. 准备一小段数据

在服务器 DiQP 项目根目录执行：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv.py --input /home/cp/datasets/LDV1/training_raw/001.mkv --sequence 101 --qp 42 --frames 12
```

输出结构如下：

```text
data/Raw/101/000.png
data/Encoded/101/QP-42/000.png
data/Encoded/101/QP-42/qp_042.bin
data/Encoded/101/QP-42/hm_encode.log
data/Encoded/101/QP-42/encoding_protocol.txt
```

默认 HM 路径为 `/home/cp/桌面/yx/HM/bin/umake/gcc-9.4/x86_64/release/TAppEncoder`，配置为 `/home/cp/桌面/yx/HM/cfg/encoder_randomaccess_main.cfg`。输入尺寸按 32 像素补齐，例如 `960×536` 编码为 `960×544`，重建帧再裁回 `960×536`。临时输入和重建 YUV 会在成功后自动删除，只保留较小的 `.bin` 码流、日志、协议和 PNG。

HM 是 CPU 参考编码器，速度会明显慢于 `hevc_nvenc`。如需重跑旧 NVIDIA 协议，显式传入 `--encoder hevc_nvenc`。

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

下面的命令自动测试前 5 个 LDV 视频，每个视频使用 HM 测试标准多 QP 集合 22、27、32、37、42。默认将重叠的 512×512 预测加权拼回完整帧，再计算 RGB-PSNR、Y-PSNR 和 SSIM：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_ldv_batch.py --video-ids 1 2 3 4 5 --qps 22 27 32 37 42 --frames 60 --fraction 1 --batch-size 1 --save-limit 2
```

首次切换到 HM 前，可以删除 DiQP 临时 NVENC 帧数据；不要删除 `batchResults`、`runs`、`pretrained` 或 `modelone/dataset/hm_results`：

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

训练集固定使用 LDV 021–030，验证集使用 031–033；测试集保留 001–020，三者不重叠。默认准备 QP 22、27、32、37、42，每个视频 120 帧，并支持通过 `.prepared` 标记断点续做：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv_finetune.py --qps 22 27 32 37 42 --frames 120 --encoder hm
```

准备完成后先进行两步 decoder 冒烟训练：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 1 --train-scope decoder --max-steps 2 --val-fraction 0.02 --run-name smoke_decoder_hm_multiqp
```

冒烟测试通过后再执行正式训练：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 5 --train-scope decoder --val-fraction 0.1 --run-name decoder_hm_multiqp
```

`history.csv` 保存全部 QP 的综合验证结果，`validation_by_qp.csv` 保存每个 QP 的独立 PSNR 增益。checkpoint 同时记录 HM 编码器和完整 QP 列表，不能误接到旧 NVENC 单 QP 训练任务上。
