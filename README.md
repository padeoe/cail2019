# 法研杯(CAIL 2019)相似案例匹配任务

## 1 比赛介绍
1. 赛题描述。
[tbd]
2. 难点
[tbd]
## 2 项目方案介绍
主要采用了数据增广、孪生BERT，没有模型集成。描述如下：

### 2.1 数据预处理
问题：三元组样本过少
训练集: (A,B,C),0

解法：利用三元组性质做增广

|序号   | 描述  | 增广产物  | 
|---|-----------------|-----------|
| 1 | 反对称增广       | (A,C,B), 1 |
| 2 | 自反性增广       | (C,C,A), 0 |
| 3 | 自反性+反对称    | (C,A,C), 1 |
| 4 | 启发式增广       | (B,A,C), 0 |
| 5 | 启发式增广+反对称 | (B,C,A), 1 |

效果：第一阶段交叉验证 acc 提升 6.4%

### 2.2 模型结构

<img src="doc/models.jpg" alt="模型结构" width="800"/>

### 2.3 尝试的其他方法
[tbd]

## 3 项目构建

### 3.1 运行环境

#### 软件依赖

- Python 3.6+
- PyTorch 1.1.0+
- [requirements.txt](requirements.txt)
- Windows 和 Linux 均可
- Nvidia Apex：可选，用于混合精度训练，安装参见[https://github.com/NVIDIA/apex#quick-start](https://github.com/NVIDIA/apex#quick-start)，将代码中 `fp16 = True` 修改为 `fp16 = False` 可不依赖 Apex。使用 apex 可以降低显存消耗并提速。

#### 硬件要求

建议使用多块显卡且显存总量大于 15GB，否则需要降低 batch size。cpu 训练很慢。

我们测试了使用了多种参数和硬件资源资源组合，训练一个 epoch 所需的显存和时间如下（除 CPU 外都启用了混合精度训练）：

|设备   | batch_size  | 显存消耗  | 训练时间  |
|---|---|---|---|
| 4*TITAN X (Pascal) 12g | 12 | 4114MiB/gpu | 23min |
| 2*TITAN X (Pascal) 12g | 12 | 7291MiB/gpu | 35min |
| TITAN X (Pascal) 12g | 10 | 11823MiB  | 56min  |
| TITAN X (Pascal) 12g | 5 | 7035MiB  | 57min  |
| i7-5930K（6核3.5GHz） | 12 |  35GB内存 | 24h  |

可见 CPU 基本训练不动…… GPU 的话要显存要大约 15GB 才能用本项目的默认 batch size 跑起来。
如果显存不足，可以尝试自行修改训练代做软 batch。

### 3.2 代码结构和数据集
项目结构如下：
```
$ tree
.
├── cli_pred.py    ->  命令行交互式预测
├── data           ->  任务数据集
│   └── ...
├── data.py        ->  划分固定数据集进行快速测试
├── docker
│   └── ...
├── judger.py      ->  比赛官方的模型性能评测脚本
├── main.py        ->  比赛要求的模型调用脚本
├── model.py       ->  模型核心代码，训练入口
└── requirements.txt
```
#### 数据集

由于版权原因，本项目不提供数据集，主办方提供了[第二阶段数据集的下载链接](https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip)。可以直接运行本项目的 `data.py` 完成数据集的下载和处理。

数据集处理好后放在 `data` 目录下，按照如下结构组织：
```
$ tree data
data
├── raw
│   └── SCM_5k
│       └── SCM_5k.json
├── test
│   ├── ground_truth.txt
│   └── input.txt
└── train
    └── input.txt
```
`raw` 存放原始数据集文件。`train`、`test` 则是划分产生的固定训练测试集，用于快速测试模型的可用性和性能。

### 3.2 训练
[`model.py`](model.py) 的 `main` 定义了训练参数：
```python
if __name__ == '__main__':
    TRAINING_DATASET = 'data/raw/SCM_5k/SCM_5k.json'

    test_input_path = 'data/test/input.txt'
    test_ground_truth_path = 'data/test/ground_truth.txt'

    config = {
        "max_length": 512,
        "epochs": 2,
        "batch_size": 12,
        "learning_rate": 2e-5,
        "fp16": True
    }
    # ...
    trainer.train(MODEL_DIR, 1)
```

确保其中 `BERT_PRETRAINED_MODEL` 为 pytorch 版本的 BERT 预训练模型的路径，即可开始训练：
```bash
python model.py
```

运行过程中会打印每一个 epoch 的 loss 和 accuracy。`trainer.train(MODEL_DIR, 1)` 
的第二个参数是 k 折交叉验证的折数，设置为大于 1 的数会进行 k-fold cv 并打印每一折的性能和平均性能。

训练完毕后模型存储在 `model` 目录下。

### 3.3 测试、评价
`main.py`、`judger.py` 分别是比赛主办方给出的线上测试和评分脚本。
可以运行这两个脚本模拟比赛线上评测过程。

我们使用项目 `data/test` 目录下划分好的测试数据进行预测和评分：

预测：
```bash
INPUT_FILE=data/test/input.txt
PREDICT_OUTPUT_FILE=data/test/output.txt
python main.py $INPUT_FILE $PREDICT_OUTPUT_FILE
```
测试完会将结果写入 `PREDICT_OUTPUT_FILE`。

评分：
```bash
PREDICT_OUTPUT_FILE=data/test/output.txt
GROUND_TRUTH_FILE=data/test/ground_truth.txt
python judger.py $GROUND_TRUTH_FILE $PREDICT_OUTPUT_FILE
```

运行后会输出模型准确率。

### 3.4 使用Docker构建
参见[docker](docker)目录下的文档。

## 4 致谢

感谢队友 [@raven4752](https://github.com/raven4752) 提出的大量idea以及对本项目的参与和贡献!
