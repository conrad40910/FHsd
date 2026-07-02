# FHsd-Data

本目录包含 **FHsd 方法** 的实验代码与数据说明，用于 GitHub 公开。**原始数据不上传 GitHub**，请从下方百度网盘下载。

## 目录结构

```
FHsd-Data/
├── README.md
├── results.txt                 # 运行后自动生成
├── data/
│   └── longhorizon2/           # 网盘下载后解压至此（仓库中不包含）
│       ├── benchmark/          # 8 个基准数据集 (Y_df.csv)
│       │   ├── ETTh1/
│       │   ├── ETTh2/
│       │   ├── ETTm1/
│       │   ├── ETTm2/
│       │   ├── ECL/
│       │   ├── Exchange/
│       │   ├── ILI/
│       │   ├── TrafficL/
│       │   └── Weather/
│       ├── pems07/
│       │   └── PEMS07.npz      # PEMS07 交通流 (883 传感器)
│       └── solar/
│           └── solar_AL.txt    # Solar 数据集 (137 电站)
└── code/
    ├── run_fhsd.py             # 统一实验入口
    └── neuralforecast/         # FHsd 方法补丁（需配合完整 neuralforecast 使用）
        ├── __init__.py
        ├── auto.py             # AutoFHsd 超参搜索
        └── models/
            ├── __init__.py
            └── fhsd.py         # FHsd 模型实现（类名 FHsd）
```

## 方法说明

FHsd 在 N-HiTS 基础上集成三个模块：


| 模块       | 代码位置                                                          | 说明         |
| -------- | ------------------------------------------------------------- | ---------- |
| **FDI**  | `FrequencyAwareInterpolation`                                 | 频域感知动态插值   |
| **FDAP** | `FrequencyEntropyPooling`（`pooling_mode="frequency_dynamic"`） | 频域感知动态自适应池化     |
| **BSD**  | `enable_self_distill`                                         | 块级自蒸馏 |


核心实现文件：`code/neuralforecast/models/fhsd.py`（模型类 `FHsd`）

## 环境依赖

本目录是 **FHsd 方法补丁包**，不能单独运行，需配合完整的 [neuralforecast](https://github.com/Nixtla/neuralforecast) 环境：

```bash
pip install neuralforecast ray[tune] scikit-learn pandas numpy torch pytorch-lightning
```

此外还需要：


| 依赖                               | 用途                          |
| -------------------------------- | --------------------------- |
| `neuralforecast.core`            | 训练与交叉验证框架                   |
| `neuralforecast.losses`          | 损失函数                        |
| `neuralforecast.common`          | `AutoFHsd` 基类（`auto.py` 依赖） |
| `datasetsforecast.long_horizon2` | 加载 benchmark 数据集（ETTh1 等）   |


`datasetsforecast.long_horizon2` 不在本目录内，请从完整实验项目中获取并安装。

## 数据下载

**GitHub 仓库不包含任何数据集文件。** 请从百度网盘下载 `longhorizon2` 文件夹，解压到 `FHsd-Data/data/` 下，使路径为 `FHsd-Data/data/longhorizon2/`。

```
链接: https://pan.baidu.com/s/1LkK2U8cHH5k2RAoOG4s3Lg?pwd=1234
提取码: 1234
```

网盘内容包含完整的 `longhorizon2` 目录：


| 子目录 / 文件             | 说明                                                |
| -------------------- | ------------------------------------------------- |
| `benchmark/`         | ETTh1/2、ETTm1/2、ECL、Exchange、ILI、TrafficL、Weather |
| `pems07/PEMS07.npz`  | PEMS07 交通流 (28224×883)                            |
| `solar/solar_AL.txt` | Solar 发电数据 (T×137)                                |


## 使用方法

### 1. 加载 FHsd 补丁

先安装 neuralforecast，再通过 `PYTHONPATH` 让 Python 优先加载本目录中的补丁代码：

**Linux / macOS：**

```bash
export PYTHONPATH="/path/to/FHsd-Data/code:$PYTHONPATH"
```

**Windows PowerShell：**

```powershell
$env:PYTHONPATH="K:\path\to\FHsd-Data\code"
```

> 不推荐仅 `cp` 两个文件到官方 neuralforecast 目录：官方包使用 `nhits.py` / `AutoNHITS`，且 `auto.py` 依赖 `neuralforecast.common` 等模块，本目录未包含完整框架。

### 2. 准备数据

确认已按上文将网盘中的 `longhorizon2` 解压到 `data/longhorizon2/`。

### 3. 运行实验

在 `FHsd-Data` 目录下执行：

```bash
# ETTh1 基准数据集
python code/run_fhsd.py --horizon 96 --dataset ETTh1 --num_samples 20

# PEMS07（npz，指标默认在标准化尺度上报告）
python code/run_fhsd.py --horizon 12 --dataset pems07 --num_samples 5

# Solar
python code/run_fhsd.py --horizon 96 --dataset solar --num_samples 20 --metric_scale both
```

### 4. 主要参数


| 参数                   | 说明                                  | 默认值                                    |
| -------------------- | ----------------------------------- | -------------------------------------- |
| `--horizon`          | 预测步长                                | 必填                                     |
| `--dataset`          | 数据集名称                               | 必填                                     |
| `--num_samples`      | Ray Tune 搜索次数                       | 5                                      |
| `--metric_scale`     | 指标尺度：`scaled` / `original` / `both` | scaled                                 |
| `--data_root`        | 数据根目录                               | `data/longhorizon2`                    |
| `--pems07_npz`       | PEMS07 文件路径                         | `data/longhorizon2/pems07/PEMS07.npz`  |
| `--solar_path`       | Solar 文件路径                          | `data/longhorizon2/solar/solar_AL.txt` |
| `--solar_n_time`     | Solar 使用的时间步数；`-1` 表示全量             | 52560                                  |
| `--solar_start_date` | Solar 起始时间                          | `2006-01-01 00:00:00`                  |
| `--solar_freq`       | Solar 时间频率                          | `15min`                                |


## 数据集说明

所有数据集统一按 **6:2:2** 划分（train 60% / val 20% / test 20%）。


| 数据集                                                     | 路径                                     | 格式              | 划分                             |
| ------------------------------------------------------- | -------------------------------------- | --------------- | ------------------------------ |
| ETTh1/2, ETTm1/2, ECL, Exchange, ILI, TrafficL, Weather | `data/longhorizon2/benchmark/`         | CSV 长表          | train 60% / val 20% / test 20% |
| PEMS07                                                  | `data/longhorizon2/pems07/PEMS07.npz`  | npz (28224×883) | train 60% / val 20% / test 20% |
| Solar                                                   | `data/longhorizon2/solar/solar_AL.txt` | txt (T×137)     | train 60% / val 20% / test 20% |


PEMS07 与 Solar 在加载时使用 `StandardScaler`（仅用 train 段拟合）。指标默认在**标准化尺度**（`--metric_scale scaled`）上计算。

## 结果输出

运行结果追加写入 `FHsd-Data/results.txt`。以下为 PEMS07 在 **scaled** 尺度下的示例（非原始交通流量尺度）：

```
Dataset=pems07, horizon=12, num_samples=5 | MSE=0.080780, MAE=0.187669
```

## 引用

如使用本数据或代码，请引用相应论文（待发表）。