# LDV 快速测试

本流程先取一个 LDV 视频的少量原始帧，验证 HEVC QP 37 和预训练 DiQP 是否有效，避免一次拆解 200 个视频占满磁盘。默认保留视频的原始帧序列，不强制改变帧率。

## 1. 准备一小段数据

在服务器 DiQP 项目根目录执行：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv.py --input /home/cp/datasets/LDV1/training_raw/001.mkv --sequence 101 --qp 37 --frames 12
```

输出结构如下：

```text
data/Raw/101/000.png
data/Encoded/101/QP-37/000.png
data/Encoded/101/QP-37/qp_037.mp4
```

如果服务器的 `hevc_nvenc` 不可用，脚本会自动回退到 CPU `libx265`。

## 2. 快速测试

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 37 --fraction 0.25 --batch-size 1 --save-limit 6
```

结果写入 `report_v2.csv`。重点看 `PSNR Gain` 和 `SSIM Gain`：正数表示模型优于压缩输入，负数表示模型使结果变差。

`testResults/comparison_*.png` 从左到右是 Encoded、Model、Raw；`difference_x5_*.png` 左侧是 Encoded 误差，右侧是 Model 误差，亮度经过 5 倍放大。

## 3. 扩大测试

快速流程确认无报错后，可去掉抽样：

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 37 --fraction 1 --batch-size 1 --save-limit 12
```

需要测试其他 LDV 视频时使用新的序列号，例如将 `002.mkv` 对应到 `--sequence 102`，不要覆盖已有实验。

## 4. 多视频、多 QP 批量测试

先用已经准备好的序列做一次整帧冒烟测试，确认服务器环境、显存和拼接流程正常：

```bash
/home/cp/anaconda3/envs/diqp/bin/python test.py --sequence 101 --qp 37 --fraction 1 --batch-size 1 --save-limit 3 --full-frame-metrics --report batchResults/fullframe_smoke.csv --results-dir batchResults/fullframe_smoke
```

`512×512` 是预训练模型固定的输入块大小，不是最终评价窗口。整帧模式会把所有重叠块加权拼回原始分辨率，并让每个视频帧只参与一次最终统计。整帧模式必须使用 `--fraction 1`，否则无法覆盖完整画面。

下面的命令自动测试前 5 个 LDV 视频，每个视频测试 QP 32、37、42。默认将重叠的 512×512 预测加权拼回完整帧，再计算 RGB-PSNR、Y-PSNR 和 SSIM：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_ldv_batch.py --video-ids 1 2 3 4 5 --qps 32 37 42 --frames 60 --fraction 1 --batch-size 1 --save-limit 2
```

每次运行都会在 `batchResults/ldv_时间戳/` 下创建独立目录。`metrics.csv` 保存每个视频的完整帧结果，`summary.csv` 保存总体和分 QP 平均结果，`visuals/` 保存完整帧对比图和逐帧指标，`protocol.txt` 记录本次参数。单个组合失败时脚本默认记录到 `failures.csv` 并继续执行后续组合。
