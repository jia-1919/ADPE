# ADPE

基于 [TAP](https://github.com/HHenryD/TAP) 的属性分支 Dropout 与 Batch-LOO AA-DPE 实现。

## 安装

使用 Python 3.9 和 NVIDIA GPU，在项目根目录执行：

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
pip install -r ../Dassl.pytorch/requirements.txt
pip install -e ../Dassl.pytorch
pip install -r requirements.txt
```

## 运行

按 [CoOp 数据说明](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) 准备数据，将下方路径替换为数据集总目录：

```bash
DATA=/path/to/datasets bash scripts/base_to_new.sh dtd 1
```

默认进行 16-shot Base 类训练（60 个 epoch），然后评估 New 类。将 `dtd` 改为 `all` 可运行全部数据集。

仅评估已有权重：

```bash
RUN_TRAIN=0 DATA=/path/to/datasets bash scripts/base_to_new.sh dtd 1
```

配置文件：[`configs/trainers/ADPE/ADPE.yaml`](configs/trainers/ADPE/ADPE.yaml)。权重和结果保存在 `output/`。
