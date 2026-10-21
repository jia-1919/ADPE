# ADPE

本项目以 `trainers/ADPE.py` 中的实现为准，在 TAP 的视觉/文本提示学习和 Gaussian-weighted Parameter Averaging（GPA）基础上，组合属性分支 Dropout 与测试阶段的 Batch Leave-One-Out 属性感知 DPE。

训练入口、配置和实验脚本统一使用 **ADPE**：

- 训练器：`--trainer ADPE`。
- 方法配置：`MODEL.ADPE.*`。
- 精度配置：`TRAINER.ADPE.PREC`。
- 默认配置：`configs/trainers/ADPE/ADPE.yaml`。

`README.d` 与本文件内容一致。

## 1. 方法流程

### 训练阶段

1. 使用 CLIP ViT-B/16，学习视觉提示、文本上下文和属性注意力池化参数。
2. 按视觉/文本两个阶段交替训练，默认每 5 个视觉 epoch 后执行 1 个文本 epoch，共 50 + 10 = 60 个 epoch。
3. 属性分支参与各自的分类损失；融合分类损失前随机屏蔽部分属性分支，至少保留一个属性分支。`ATTR_BRANCH_DROPOUT_RESCALE` 控制是否重新归一化保留分支的权重。
4. 保留原实现的教师约束与文本对比损失，两个训练阶段分别进行 GPA 参数平均。

属性 Dropout 在评估时关闭。融合分数为：

```text
logits = alpha * class_logits + (1 - alpha) * weighted_attribute_logits
```

### 评估阶段

- 根据模型的原始预测生成伪标签，并按预测熵维护每类 Top-K 历史缓存。
- 对当前 batch 的第 i 张图片，临时缓存包含历史样本及本 batch 的其他图片，排除图片 i 自身；整个 batch 完成预测后再更新历史缓存。
- 类别缓存相似度与各属性缓存相似度按 rho 融合，然后进行一次指数缓存映射。
- 在每个 batch 内优化文本原型和类别缓存原型的残差；残差每个 batch 重新初始化，只有特征缓存跨 batch 保留。
- 每次开始评估都会清空缓存。测试标签仅用于最终准确率统计。

```text
mixed_affinity = (1 - rho) * class_affinity + rho * mean(attribute_affinity)
```

`rho=0` 为类别缓存 DPE 消融；关闭 `DPE.ENABLED` 则直接使用提示模型的融合输出。测试 batch 大小和数据顺序会影响在线缓存结果，比较实验时应保持一致。

## 2. 环境安装

沿用原项目的 Python 3.9、PyTorch 2.0.0 / torchvision 0.15.1 环境。当前模型包含 CUDA 运算，需要可用的 NVIDIA GPU；请为本机驱动安装匹配的 CUDA 版 PyTorch。

在项目根目录运行，下面的脚本命令适用于 Linux、WSL 或已激活相应 Python 环境的 Git Bash：

```bash
conda create -y -n adpe python=3.9
conda activate adpe

python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

# Dassl 安装在项目目录外。
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
python -m pip install -r ../Dassl.pytorch/requirements.txt
python -m pip install -e ../Dassl.pytorch

python -m pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

仓库自带修改过的 `clip/`，运行时使用这份本地实现。首次使用会将 CLIP 权重下载到 `~/.cache/clip`；离线运行时需提前准备权重。

## 3. 数据准备

数据结构及划分文件沿用 [CoOp 的数据集准备说明](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)。`DATA` 指向包含各数据集子目录的总目录；脚本顶部默认使用 `/mnt/a/cs_ai_wys/lijia/Data`，可直接修改或通过环境变量覆盖。

| 配置文件名 / 脚本参数 | 数据子目录 | 属性描述目录 |
| --- | --- | --- |
| `caltech101` | `caltech-101` | `Caltech101` |
| `oxford_pets` | `oxford_pets` | `OxfordPets` |
| `stanford_cars` | `stanford_cars` | `StanfordCars` |
| `oxford_flowers` | `oxford_flowers` | `OxfordFlowers` |
| `food101` | `food-101` | `Food101` |
| `fgvc_aircraft` | `fgvc_aircraft` | `FGVCAircraft` |
| `sun397` | `sun397` | `SUN397` |
| `dtd` | `dtd` | `DescribableTextures` |
| `eurosat` | `eurosat` | `EuroSAT` |
| `ucf101` | `ucf101` | `UCF101` |
| `imagenet` | `imagenet` | `ImageNet` |

描述文件位于 `descriptions/<描述目录>/descriptions_bag.json`，按类别和属性组织：

```json
{
  "class name": {
    "Texture": ["description 1", "description 2"],
    "Shape": ["description 3"]
  }
}
```

属性数量和顺序从 JSON 动态读取；类别名称按小写、下划线转空格的规则匹配。ImageNet 的同名类别通过 synset 对应描述键，保持数据集的数值标签不变。ImageNet-A/R/Sketch/V2 复用 ImageNet 的描述文件。

## 4. Base-to-New 训练和评估

### 一键运行

```bash
# DTD，16-shot，seed=1：属性 Dropout 训练 Base，然后用 AA-DPE 测试 New。
DATA=/path/to/datasets bash scripts/base_to_new.sh dtd 1

# 全部 11 个数据集，3 个 seed。
DATA=/path/to/datasets SEEDS="1 2 3" bash scripts/base_to_new.sh all

# 指定部分数据集、训练 batch 和 Dropout 参数。
DATA=/path/to/datasets DATASETS="dtd eurosat" TRAIN_BATCH_SIZE=8 \
TAP_ATTR_BRANCH_DROPOUT_P=0.25 DPE_ATTR_AFFINITY_WEIGHT=0.4 \
bash scripts/base_to_new.sh all 1

# 同一份训练权重分别测试 Base 和 New。
DATA=/path/to/datasets SUBSAMPLE_CLASSES="base new" bash scripts/base_to_new.sh dtd 1
```

`base_to_new.sh` 直接合并属性 Dropout 训练和 Batch-LOO AA-DPE 测试，无需其他 Shell 脚本。默认训练 Base，再加载同一目录中的视觉 / 文本权重测试 New；用 `SUBSAMPLE_CLASSES` 选择 `base`、`new`、`all` 或 `"base new"`。默认训练 50 + 10 个 epoch、16-shot，训练 / 测试 batch 为 32 / 100。

顶部保留 `DATASET_SETTINGS` 表，列顺序为数据集、描述长度分位数、文本学习率、视觉学习率、U3。DTD 使用 `1.0 / 0.002 / 0.006 / 3.0`，其他数据集按表中的设置运行；训练和测试使用同一组参数。默认只运行 DTD；传入 `all` 运行表中的全部 11 个数据集，`DATASETS="dtd eurosat"` 可选择子集。

训练参数为 `TAP_ATTR_BRANCH_DROPOUT_P=0.15`、`TAP_ATTR_BRANCH_DROPOUT_RESCALE=0`、`TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT=0.5`。这些兼容环境变量由现有配置代码读取；实际训练器、配置和参数键统一使用仓库已有的 `ADPE`，无需旧的 `TAP_drop` 或 `TAP_drop_LOO_AA_DPE` 训练器。

测试参数在脚本顶部集中设置：`DPE_SHOTS=3`、`DPE_ALPHA=6.0`、`DPE_BETA=5.0`、`DPE_TEXT_LR=0.0006`、`DPE_IMAGE_LR=0.0006`、`DPE_ALIGN=0.5`、`DPE_ENTROPY_THRESHOLD=0.1`、`DPE_ATTR_AFFINITY_WEIGHT=0.4`。训练时关闭 DPE，测试时关闭属性 Dropout 并默认启用 AA-DPE。可直接编辑顶部变量，也可用环境变量覆盖；`VISION_LR`、`TEXT_LR`、`U3`、`MAX_LEN_QUANTILE` 可覆盖数据集表。短名称 `ATTR_DROPOUT_P`、`FUSION_CE_WEIGHT`、`AA_WEIGHT` 仍可使用。

默认检查点和结果目录：

```text
output/base2new/
├── train_base/dtd/shots_16/ADPE/seed1/0.15_0.5/
│   ├── vision_prompt_learner/model.pth.tar-60
│   ├── text_prompt_learner/model.pth.tar-60
│   └── log.txt
└── test_new_aa_dpe/dtd/shots_16/ADPE/seed1/0.15_0.5/rho_0.4/
    ├── log.txt
    └── console.log
```

只有视觉、文本两个最终 epoch 的权重都存在时才跳过训练，不再检查第 50 轮的视觉权重来判断 60 轮训练是否完成。`SKIP_EXISTING_TRAIN=0` 取消跳过；如需从头训练或修改其他训练参数，请同时指定新的 `OUTPUT_ROOT`，以免 Dassl 自动恢复旧输出目录中的检查点。默认测试最后一轮，也可以用 `LOAD_EPOCH` 指定已保存的轮次。不同 dropout / 融合损失权重、测试划分和 rho 分目录保存；任何训练、检查点检查或测试失败都会终止脚本。

### 单独训练

```bash
python train.py \
  --root /path/to/datasets \
  --trainer ADPE --seed 1 \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/ADPE/base_to_new.yaml \
  --output-dir output/my_adpe/train_base/dtd/shots_16/ADPE/seed1/0.25_0.5 \
  OPTIM.VISION_LR 0.006 OPTIM.TEXT_LR 0.002 \
  MODEL.ADPE.U3 3.0 \
  MODEL.ADPE.ATTR_BRANCH_DROPOUT_P 0.25 \
  MODEL.ADPE.ATTR_DROPOUT_FUSION_CE_WEIGHT 0.5 \
  MODEL.ADPE.DPE.ENABLED False TEST.NO_TEST True
```

### 评估已有权重

```bash
DATA=/path/to/datasets RUN_TRAIN=0 OUTPUT_ROOT=output/my_adpe \
TAP_ATTR_BRANCH_DROPOUT_P=0.25 DPE_ATTR_AFFINITY_WEIGHT=0.8 \
bash scripts/base_to_new.sh dtd 1
```

`RUN_TRAIN=0` 从上述训练目录加载权重并测试；dropout 概率和融合损失权重需与目录名称一致，属性顺序、数量、池化方式和描述长度设置也应与原训练一致。类别缓存 DPE 消融可设置 `DPE_ATTR_AFFINITY_WEIGHT=0`。也可以通过 `python train.py --eval-only --model-dir ... --load-epoch 60` 直接评估；单独使用 Python 且未指定 `--load-epoch` 时会加载 `model-best.pth.tar`。

Base / New 的调和平均为 `H = 2 * Base * New / (Base + New)`，两者同时为 0 时令 H=0。

## 5. 全类别 Few-shot 实验

该协议对全部类别分别进行 1/2/4/8/16-shot 训练，每个 shot 使用独立权重，报告全类别测试准确率。

```bash
DATA=/path/to/datasets bash scripts/few-shot.sh dtd 1

# 小训练 batch 版本，默认每个数据集训练 batch=1，测试 batch 仍为 100。
DATA=/path/to/datasets bash scripts/few-shot_batch.sh dtd 1

# 仅运行 1-shot 和 4-shot。
DATA=/path/to/datasets SHOTS_LIST="1 4" bash scripts/few-shot.sh dtd 1

# 已有权重，仅执行普通评估与 ADPE 评估。
DATA=/path/to/datasets PHASES="standard loo_aa_dpe" \
bash scripts/few-shot.sh dtd 1
```

默认阶段是 `train standard loo_aa_dpe`。普通版本输出到 `output/few_shot/ADPE/<dataset>/shot<shot>/seed<seed>/`，小 batch 版本输出到 `output/few_shot/ADPE_small_batch/`。每个实验目录包含 `train_all`、`test_standard_all` 和 `test_loo_aa_dpe_all`。

## 6. 主要配置

`train.py` 先加载完整 `ADPE.yaml`，再加载数据集配置和指定的方法配置。因此 `base_to_new.yaml`、`ADPE_class_dpe.yaml` 可以只写需要覆盖的字段。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `TRAINER.ADPE.PREC` | `fp32` | `fp32` 或 `amp` |
| `MODEL.ADPE.ATTR_BRANCH_DROPOUT_P` | `0.15` | 训练时属性分支屏蔽概率 |
| `MODEL.ADPE.ATTR_BRANCH_DROPOUT_RESCALE` | `False` | 对保留的属性融合权重重新归一化 |
| `MODEL.ADPE.ATTR_DROPOUT_FUSION_CE_WEIGHT` | `0.5` | 融合分类损失权重 |
| `MODEL.ADPE.ALPHA` | `0.4` | 类别分支融合权重 |
| `MODEL.ADPE.U1/U2/U3` | `10.0/2.5/1.5` | 图像教师约束、logit 蒸馏、文本对比损失权重 |
| `MODEL.ADPE.MAX_LEN_QUANTILE` | `1.0` | 属性描述数量分位数 |
| `MODEL.ADPE.DPE.ENABLED` | `True` | 评估时启用 Batch-LOO AA-DPE |
| `MODEL.ADPE.DPE.ATTR_AFFINITY_WEIGHT` | `0.4` | 属性缓存相似度权重 rho，范围 `[0,1]` |
| `MODEL.ADPE.DPE.SHOT_CAPACITY` | `3` | 每类历史缓存容量 |
| `MODEL.ADPE.DPE.ALPHA/BETA` | `6.0/5.0` | 指数缓存映射参数 |
| `MODEL.ADPE.DPE.TEXT_LR/IMAGE_LR` | `0.0006/0.0006` | 测试时两个原型残差的学习率 |
| `MODEL.ADPE.DPE.ALIGN` | `0.5` | 原型对齐损失权重 |
| `MODEL.ADPE.DESCRIPTIONS_PATH` | 空字符串 | 自定义描述 JSON，默认相对项目目录解析 |
| `MODEL.ADPE.ATTR_ORDER` | 空字符串 | 逗号分隔的属性名称或从 1 开始的编号 |
| `MODEL.ADPE.ATTR_COUNT` | `0` | 选定顺序中保留的属性数量；0 表示全部 |
| `OPTIM.VISION_MAX_EPOCH/TEXT_MAX_EPOCH` | `50/10` | 两个阶段的训练预算 |
| `DATALOADER.TRAIN_X/TEST.BATCH_SIZE` | `32/100` | 训练/测试 batch 大小 |

表格是通用 YAML 默认值，一键脚本会应用 `DATASET_SETTINGS` 中的学习率、U3 和描述长度设置，并支持环境变量覆盖。`DPE.ENTROPY_THRESHOLD` 保留为旧命令兼容字段；当前实现按预测熵排序维护 Top-K 缓存，不以该字段进行阈值筛选。

总 epoch 自动计算为视觉与文本预算之和。调整预算时也需保持交替阶段长度与预算一致，例如 5+1、10+2 或 50+10。单独设置 `OPTIM.MAX_EPOCH` 不会覆盖这两个阶段的预算。

参数优先级由低到高为：通用配置、数据集配置、指定方法配置、环境变量、显式命令行参数。推荐在 YAML 或命令行配置方法参数。兼容环境变量包括 `ADPE_ATTR_BRANCH_DROPOUT_P`、`ADPE_ATTR_DROPOUT_FUSION_CE_WEIGHT`、`ADPE_DPE_ATTR_AFFINITY_WEIGHT` 等；原有 `TAP_*` 和 `DPE_*` 环境变量别名仍可读取，别名值冲突时会报错。旧的 `MODEL.TAP.*` / `TRAINER.TAP.*` 配置键应改成 ADPE。

## 7. 其他实验脚本

| 脚本 | 功能 |
| --- | --- |
| `base_to_new_attr_drop.sh` | 转发至统一 ADPE Base-to-New 流程 |
| `base_to_new_choose.sh` | 按脚本顶部设置选择属性顺序和数量，训练及评估使用相同选择 |
| `eval_attr_affinity_dpe.sh` | 单个数据划分的 ADPE 评估入口 |
| `eval_tap_drop_loo_aa_dpe.sh` | 保留的旧文件名，转发至 `eval_adpe.sh` |
| `eval_branch_ablation.sh` | 固定权重逐步移除属性分支，输出 CSV / JSON，不更新 DPE 缓存 |
| `grid_search_attr_dropout.sh` | 搜索训练 Dropout 概率和融合分类损失权重，使用普通评估计算 Base / New / H |
| `grid_img.sh` | ImageNet 的 Dropout 搜索入口 |
| `tune_attr_affinity_rho.sh` | 对已有权重搜索 rho，计算 Base / New / H |
| `eval_attr_affinity_dpe_all.sh` | 按 rho 分目录保存全部扫描结果 |
| `eval_xd.sh` | ImageNet 训练、源域评估及 ImageNet-A/R/Sketch/V2 泛化评估 |

搜索示例：

```bash
DATA=/path/to/datasets DROPOUT_P_VALUES="0.15 0.25" \
FUSION_CE_WEIGHT_VALUES="0.2 0.5" \
bash scripts/grid_search_attr_dropout.sh dtd 1

DATA=/path/to/datasets RHO_VALUES="0 0.4 0.8" \
MODEL_DIR=output/base2new/ADPE/dtd/seed1/train_base \
bash scripts/tune_attr_affinity_rho.sh dtd 1
```

两个搜索脚本按 H 选择候选，默认只保留最优候选的目录和完整搜索表。用 New 测试标签选出的参数属于该搜索设置，报告独立测试结果时应单独说明调参协议。

`eval_xd.sh` 默认关闭跨数据集分类评估，保留源域和四个 ImageNet 分布偏移数据集的评估。跨数据集迁移需要目标类别的描述使用与 ImageNet 训练权重**相同的属性名称和顺序**；准备相应 JSON 后，复用已有 ImageNet 权重，设置 `RUN_TRAIN=0 RUN_IMAGENET_EVAL=0 RUN_DOMAIN_GENERALIZATION=0`，再用 `ADPE_DESCRIPTIONS_PATH` 指定单个目标数据集的 JSON、`CROSS_DATASETS` 指定该目标，并设置 `RUN_CROSS_DATASET=1`。代码会拒绝属性结构不兼容的描述，避免把不同含义的分支直接对齐。

## 8. 检查与项目结构

```bash
# 仅检查一键脚本流程，无需安装训练依赖。
python -m unittest discover -s tests -p test_base_to_new.py -v

# 依赖安装完成后执行；配置、描述和脚本流程检查不需要 GPU。
python -m unittest discover -s tests -v
```

测试会读取真实 Dassl 配置，并以模拟训练进程检查脚本参数、检查点路径和阶段衔接。它们不验证 CUDA 前向/反向、真实训练准确率或显存占用。本次修改未执行完整 GPU 训练，也未提供 ADPE 的新实验结果。

```text
train.py                         # 训练/评估入口
trainers/ADPE.py                  # 模型、交替训练、GPA、Batch-LOO AA-DPE
utils/adpe_config.py              # 配置注册、环境变量兼容、参数校验
utils/adpe_descriptions.py        # 描述读取、属性选择和类别对齐
utils/adpe_ablation.py            # 可复现的属性消融掩码
configs/trainers/ADPE/            # 通用配置和实验覆盖配置
configs/datasets/                # 15 个数据集配置
scripts/base_to_new.sh           # 数据集参数表、Dropout 训练和 AA-DPE 测试
tests/test_base_to_new.py         # 独立的一键脚本流程检查
tests/test_adpe.py                # 无需 GPU 的回归检查
```

## 9. 来源与引用

视觉/文本提示学习的基础来自 [TAP 项目](https://github.com/HHenryD/TAP) 和论文 [Tree of Attributes Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2410.11201)。数据集与训练框架沿用 CoOp / Dassl。请保留原有许可证，并按实际使用情况引用基础工作。本 README 不沿用原 TAP 表格作为 ADPE 的实验结果。

```bibtex
@inproceedings{ding2025tree,
  title={Tree of Attributes Prompt Learning for Vision-Language Models},
  author={Tong Ding and Wanhua Li and Zhongqi Miao and Hanspeter Pfister},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
