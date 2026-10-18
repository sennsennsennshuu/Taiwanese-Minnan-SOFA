# 台湾闽南语 SOFA 配置调整（2026-10-15）

本次根据用户选择配置普通话预训练模型迁移训练，数据为 leku、sutiau 的完整标签与 Common Voice 27 台湾闽南语弱标签。正式目录为云端 `/root/autodl-tmp/SOFA`，以下命令和程序内的数据路径均相对于该项目根目录。

后续修订：用户已明确选择将骨干峰值学习率由首次发布的 `0.0001` 提高到 `0.0003`，输出层保持 `0.0005`。当前 `train_config.yaml` 已同步该值；最新核对结果见 `lr_backbone_0003/verification_report.json`，原始发布与验证报告保留其历史内容。云端本次修改前的配置另存于 `configs/backup_backbone_lr_0003_20261015_223318/`。

## 修改及依据

| 文件/字段 | 原值 | 现值与原因 |
|---|---|---|
| `configs/train_config.yaml` 的 `model_name` | `mandarin_opencpop-extension` | `nan_tw_leku_sutiau_cv27`，使本次模型与普通话模型的输出目录分离 |
| `optimizer_config.lr.backbone` | `0.0005` | 首次发布为 `0.0001`，后按用户选择改为 `0.0003`，采用更积极的骨干适配幅度 |
| `optimizer_config.lr.head` | `0.0005` | 保持 `0.0005`，当前闽南语 56 类词表需要新输出层 |
| `dataloader_workers` | `0` | `4`，利用已有 CPU 工作进程并行加载 |
| `accelerator` / `devices` | `auto` / `auto` | `gpu` / `1`，明确使用当前单张 RTX 4090 |
| `val_check_interval` | `500` | `2000`，当前项目会为每条验证数据绘图，扩大留出集后控制验证开销 |
| `configs/binarize_config.yaml` 的 `valid_set_size` | `15` | `128`，扩大留出验证集 |
| `valid_set_preferred_folders` | `test` | `full_label`，优先从现有 40,163 条完整标签中留出，避免落到无边界监督的弱标签 |
| `data_augmentation.size` | `2` | `1`，保存原始特征加一份增强，仍满足双视图一致性训练并减少空间占用 |
| `key_shift_choices` | 正负 1～6 半音 | `[-2, -1, 1, 2]`，采用较小移调范围 |
| `dataset.py` 的采样器 | 最后一个桶未提交 | 统一提交分桶并补交尾桶，防止最短样本永远不进入采样 |

新配置有中文注释，编码为 UTF-8。实际权重顺序由标签编号决定，为 `[无标签, 弱标签, 完整标签]`；三个值保留 `[1, 1, 1]`。原注释顺序相反，本次已纠正。`dataloader_prefetch_factor` 虽在配置中存在，当前 `train.py` 实际固定使用 2，因此配置保持 2。

## 迁移兼容性

现有检查点为 `ckpt/Mandarin_Singing_Model.ckpt`，65 类词表；当前数据为 56 类词表。已复现项目的真实 `load_from_checkpoint` 与 `load_pretrained` 调用：复用骨干，重新初始化输出层，线性层总输出维度为 58。该维度包含项目的边界与 CTC 分支结构，并非增加字典音素。

当前前端保持 `44100 Hz / 128 mel / fmax=16000`。检查点的历史 `fmax=32000` 高于 44.1 kHz 的奈奎斯特频率，未照搬；当前前端与其不同，因此骨干不冻结。前向运行成功只能证明兼容和数值有限，不能证明模型已经学会闽南语边界。

## 验证范围

使用现有 `/root/miniconda3/envs/SOFA/bin/python`，没有安装或更改环境。SOFA 环境的 PyTorch 为 `2.4.1+cu121`，Lightning 为 `2.3.3`，实际 CUDA 和 BF16 可用。

- 实际 SOFA 元数据读取：64,248 条，其中完整标签 40,163 条、弱标签 24,085 条；词表 56 类，字典外音素为 0。
- 验证优先规则模拟：128 条均为完整标签。原 `binarize.py` 的抽样没有固定种子；训练配置的 `random_seed` 不会固定该划分。模拟种子仅用于本次检查。
- 全数据时长约 164,363.206 秒（45.66 小时），最长约 33.882 秒，超过 `max_length=45` 的样本为 0。
- 修复前全数据分桶漏 14 条，修复后 64,248 条全部覆盖；原有完整分桶行为保持一致。另验证单样本、单桶、多桶三类回归场景，以及留出 128 条后 64,120 条训练数据的实际迭代覆盖。
- leku、sutiau、Common Voice 各一条真实 WAV，完成载入、重采样、原始/移调 mel、GPU BF16 前向检查，输出形状正确且数值均有限。
- 按全数据 float32 特征与标签估算，主要二进制内容约 14,986,621,056 字节（13.96 GiB）；不含全部 HDF5 元数据开销、检查点和训练日志，实际大小须预处理完成后测量。
- `validation_published.json` 为发布后真实文件的检查结果，包含正式文件摘要核对。

## 备份与产物

原配置和 `dataset.py` 已在本地 `before/` 保留，也已备份到云端 `configs/backup_nan_tw_20261015_221845/`，发布前后摘要在 `publication_report.json`。`configs.diff` 和 `dataset.diff` 可查看精确改动。原音频、三份正式标签 CSV、训练入口、预处理入口和模型模块的受保护文件摘要均保持不变。

本次未生成二进制数据，未启动训练，`data/binary` 仍为空。待开始训练时，在 SOFA 项目根目录、已激活 SOFA 环境的终端依次执行：

```bash
python binarize.py -c configs/binarize_config.yaml
python train.py -c configs/train_config.yaml -d data -p ckpt/Mandarin_Singing_Model.ckpt
```

`-p` 是迁移训练所必需的命令行参数，程序未从 YAML 读取预训练路径；不传该参数会从零训练。完整标签来自现有 TextGrid，留出集不是另行人工校正的边界金标准。

需要重复验证时，在本地项目根目录执行：

```powershell
python -u -B test/test_sofa_cloud_config.py --published
```

