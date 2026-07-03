# FHsd-Data

本目录包含 **FHsd 方法** 的数据说明，用于 GitHub 公开。**部分原始数据不上传 GitHub**，请从下方百度网盘下载。

## 目录结构

```
FHsd-Data/
├── README.md
├── results.txt                 # 运行后自动生成
├── DATABASE/
   └── longhorizon2/           # 网盘下载后解压至此（仓库中不包含）
       ├── benchmark/          # 8 个基准数据集 (Y_df.csv)
       │   ├── ETTh1/
       │   ├── ETTh2/
       │   ├── ETTm1/
       │   ├── ETTm2/
       │   ├── ECL/
       │   ├── Exchange/
       │   ├── ILI/
       │   ├── TrafficL/
       │   └── Weather/
       ├── pems07/
       │   └── PEMS07.npz      # PEMS07 交通流 (883 传感器)
       └── solar/
           └── solar_AL.txt    # Solar 数据集 (137 电站)
```

## 方法说明

FHsd 在 N-HiTS 基础上集成三个模块：


| 模块       | 代码位置                                                          | 说明         |
| -------- | ------------------------------------------------------------- | ---------- |
| **FDI**  | `FrequencyAwareInterpolation`                                 | 频域感知动态插值   |
| **FDAP** | `FrequencyEntropyPooling`（`pooling_mode="frequency_dynamic"`） | 频域感知动态自适应池化     |
| **BSD**  | `enable_self_distill`                                         | 块级自蒸馏 |


核心实现文件：`code/neuralforecast/models/fhsd.py`（模型类 `FHsd`）

## 数据下载

**GitHub 仓库不包含Solar、PEMS07数据集文件。** 完整数据集文件请从百度网盘下载 `longhorizon2` 文件夹，解压到 `FHsd-Data/data/` 下，使路径为 `FHsd-Data/data/longhorizon2/`。

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

### 主要参数


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
