# DiQP LDV 微调试验

这套流程用于验证预训练 HEVC 模型能否通过少量 LDV 数据适配 `960×536` 内容。当前零样本测试使用 001–005，因此它们继续保留为测试视频，不参与训练或验证。数据准备默认使用压缩包原始 `HM16.3-standard.exe` 和 Low-Delay P 配置；旧 HM Random Access、NVENC 或 x265 数据不能混入同一次训练。

## 1. 固定试验划分

- 训练：021–030（10 个视频）
- 验证：031–033（3 个视频）
- 测试：001–005（现有零样本基线）
- 第一阶段只使用 QP42，每个训练/验证视频取前 120 帧

准备数据：

```bash
/home/cp/anaconda3/envs/diqp/bin/python prepare_ldv_finetune.py --frames 120 --qp 42
```

脚本默认写入 `data/LDV_finetune/`，并通过 `.prepared/` 标记支持断点续跑。再次执行时会跳过已经完整准备的视频。

## 2. 两步显存冒烟

先只训练输出层 2 步，并使用少量验证块：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 1 --train-scope output --max-steps 2 --val-fraction 0.02 --run-name smoke_output_qp42
```

成功标准：不出现 CUDA OOM，生成 `runs/ldv_finetune/smoke_output_qp42/best.pt` 和 `history.csv`，并打印峰值显存。

## 3. 输出层试训

冒烟成功后运行 5 个 epoch：

```bash
/home/cp/anaconda3/envs/diqp/bin/python train_ldv.py --epochs 5 --train-scope output --val-fraction 0.1 --run-name output_qp42
```

训练中每个 epoch 都进行固定验证并覆盖保存 `last.pt`；按验证 PSNR 增益保存 `best.pt`。如输出层适配不足，可在独立实验中将 `--train-scope` 改为 `decoder`，不要覆盖前一次运行目录。

## 4. 保留集测试

用微调后的 best checkpoint 重跑 001–005、QP42：

```bash
/home/cp/anaconda3/envs/diqp/bin/python evaluate_ldv_batch.py --video-ids 1 2 3 4 5 --qps 42 --frames 200 --fraction 1 --batch-size 1 --save-limit 2 --model-path /home/cp/桌面/yx/DiQP/runs/ldv_finetune/output_qp42/best.pt
```

与零样本基线比较时重点看平均值、中位数、`positive RGB` 数量和 SSIM。第一阶段成功标准为平均 RGB/Y PSNR 增益为正、至少 3/5 视频 RGB-PSNR 提升且平均 SSIM 不下降。
