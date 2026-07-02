import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys

FHSD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_ROOT = os.path.join(FHSD_ROOT, "code")
DATA_ROOT = os.path.join(FHSD_ROOT, "data", "longhorizon2")

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

import argparse
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple

from ray import tune

from neuralforecast.auto import AutoFHsd
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info
from sklearn.preprocessing import StandardScaler

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def compute_split_sizes(n_time: int) -> Tuple[int, int, int]:
    num_train = int(n_time * TRAIN_RATIO)
    num_val = int(n_time * VAL_RATIO)
    num_test = n_time - num_train - num_val
    return num_train, num_val, num_test


def _node_codes(unique_id_series: pd.Series) -> np.ndarray:
    uid = unique_id_series
    if hasattr(uid, "cat") and hasattr(uid.cat, "codes"):
        return uid.cat.codes.to_numpy()
    uid_str = uid.astype(str).to_numpy()
    return np.array([int(s.split("_")[-1]) for s in uid_str], dtype=np.int32)


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    unique_ids: pd.Series,
    scaler: Optional[StandardScaler] = None,
    metric_scale: str = "scaled",
):
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    diff_scaled = (y_hat - y_true)[mask]
    mse_scaled = float(np.mean(diff_scaled ** 2))
    mae_scaled = float(np.mean(np.abs(diff_scaled)))

    mse_original = None
    mae_original = None
    if scaler is not None and metric_scale in ("original", "both") and unique_ids is not None:
        codes = _node_codes(unique_ids)
        mean = scaler.mean_.astype(np.float32)
        scale = scaler.scale_.astype(np.float32)
        y_true_o = y_true.astype(np.float32) * scale[codes] + mean[codes]
        y_hat_o = y_hat.astype(np.float32) * scale[codes] + mean[codes]
        mask_o = np.isfinite(y_true_o) & np.isfinite(y_hat_o)
        diff_o = (y_hat_o - y_true_o)[mask_o]
        mse_original = float(np.mean(diff_o ** 2))
        mae_original = float(np.mean(np.abs(diff_o)))

    if metric_scale == "scaled":
        return mse_scaled, mae_scaled, mse_scaled, mae_scaled, mse_original, mae_original
    if metric_scale == "original":
        if mse_original is None:
            return mse_scaled, mae_scaled, mse_scaled, mae_scaled, None, None
        return mse_original, mae_original, mse_scaled, mae_scaled, mse_original, mae_original

    return mse_scaled, mae_scaled, mse_scaled, mae_scaled, mse_original, mae_original


def load_pems07_npz(
    npz_path: str,
    freq: str = "5min",
    scale: bool = True,
):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"PEMS07 npz not found: {npz_path}")

    data_npz = np.load(npz_path, allow_pickle=True)
    if "data" in data_npz:
        arr = data_npz["data"]
    else:
        arr = data_npz[data_npz.files[0]]

    if arr.ndim == 3:
        if arr.shape[0] >= arr.shape[1]:
            t, n, c = arr.shape
            data_all = arr.reshape(t, n * c)
        else:
            n, t, c = arr.shape
            data_all = arr.transpose(1, 0, 2).reshape(t, n * c)
    elif arr.ndim == 2:
        data_all = arr if arr.shape[0] >= arr.shape[1] else arr.T
    else:
        t = arr.shape[0]
        data_all = arr.reshape(t, -1)

    data_all = data_all.astype(np.float32, copy=False)
    num_samples, n_nodes = data_all.shape
    num_train, num_val, num_test = compute_split_sizes(num_samples)

    train_end = num_train
    val_end = num_train + num_val

    scaler = None
    if scale:
        scaler = StandardScaler()
        scaler.fit(data_all[:num_train])
        data_all = scaler.transform(data_all).astype(np.float32, copy=False)

    ds = pd.date_range(start="2017-05-01", periods=num_samples, freq=freq)
    node_ids = [f"node_{i}" for i in range(n_nodes)]

    frames = []
    for i, node_id in enumerate(node_ids):
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": node_id,
                    "ds": ds,
                    "y": data_all[:, i],
                }
            )
        )

    Y_long = pd.concat(frames, ignore_index=True)
    Y_long["unique_id"] = Y_long["unique_id"].astype("category")

    info = {
        "freq": freq,
        "val_size": num_val,
        "test_size": num_test,
        "n_time": num_samples,
        "n_nodes": n_nodes,
        "train_size": num_train,
        "split": {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, num_samples),
        },
        "scaled": scale,
    }
    return Y_long, info, scaler


def _open_text_robust(path: str):
    with open(path, "rb") as fb:
        head = fb.read(4)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return open(path, "r", encoding="utf-16", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def load_solar(
    file_path: str,
    freq: str = "15min",
    start_date: str = "2006-01-01 00:00:00",
    n_time: int = 52560,
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Solar file not found: {file_path}")

    rows = []
    expected_cols = None
    with _open_text_robust(file_path) as f:
        for line in f:
            parts = line.strip("\n").split(",")
            parts = [x for x in parts if x.strip() != ""]
            if len(parts) < 2:
                continue

            if expected_cols is None:
                expected_cols = len(parts)
            if len(parts) != expected_cols:
                continue

            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(row)
            if n_time is not None and n_time > 0 and len(rows) >= n_time:
                break

    if len(rows) == 0:
        raise ValueError(f"No valid rows found in Solar file: {file_path}")

    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Invalid Solar shape: ndim={data.ndim}, shape={getattr(data, 'shape', None)}")
    if data.shape[1] != 137:
        logging.warning(f"[Solar] expected 137 stations, got {data.shape[1]}")

    t_steps, n_series = data.shape
    num_train, num_val, num_test = compute_split_sizes(t_steps)

    scaler = StandardScaler()
    scaler.fit(data[:num_train])
    data_scaled = scaler.transform(data).astype(np.float32, copy=False)

    dates = pd.date_range(start=start_date, periods=t_steps, freq=freq)
    codes = np.repeat(np.arange(n_series, dtype=np.int32), t_steps)
    categories = [f"station_{i}" for i in range(n_series)]
    unique_id = pd.Categorical.from_codes(codes, categories=categories, ordered=False)
    ds = np.tile(dates.to_numpy(), n_series)
    y = data_scaled.T.reshape(-1)

    Y_df = pd.DataFrame({"unique_id": unique_id, "ds": ds, "y": y})

    info = {
        "freq": freq,
        "n_time": t_steps,
        "n_series": n_series,
        "train_size": num_train,
        "val_size": num_val,
        "test_size": num_test,
        "scaled": True,
    }
    return Y_df, info, scaler


def load_dataset(args) -> Tuple[pd.DataFrame, str, int, int, Optional[StandardScaler], Dict[str, Any]]:
    dataset_lower = args.dataset.lower()
    metric_scaler = None
    meta: Dict[str, Any] = {"dataset": dataset_lower}

    if dataset_lower == "pems07":
        Y_df, info, metric_scaler = load_pems07_npz(args.pems07_npz, scale=True)
        freq = info["freq"]
        val_size = info["val_size"]
        test_size = info["test_size"]
        meta.update(info)
        print(
            f"[PEMS07] path={args.pems07_npz}, freq={freq}, "
            f"n_time={info['n_time']}, n_nodes={info['n_nodes']}, "
            f"split={info['train_size']}/{val_size}/{test_size}, scaled={info['scaled']}"
        )
    elif dataset_lower == "solar":
        solar_n_time = None if args.solar_n_time == -1 else args.solar_n_time
        print(f"[Solar] path={args.solar_path}")
        Y_df, info, metric_scaler = load_solar(
            file_path=args.solar_path,
            freq=args.solar_freq,
            start_date=args.solar_start_date,
            n_time=solar_n_time,
        )
        freq = info["freq"]
        val_size = info["val_size"]
        test_size = info["test_size"]
        meta.update(info)
        print(
            f"[Solar] n_time={info['n_time']}, n_series={info['n_series']}, "
            f"split={info['train_size']}/{val_size}/{test_size}"
        )
    else:
        benchmark_dir = os.path.join(args.data_root, "benchmark")
        Y_df = LongHorizon2.load(directory=benchmark_dir, group=args.dataset)
        freq = LongHorizon2Info[args.dataset].freq
        n_time = len(Y_df.ds.unique())
        num_train, val_size, test_size = compute_split_sizes(n_time)
        meta["n_time"] = n_time
        meta["train_size"] = num_train
        print(
            f"[Benchmark] dataset={args.dataset}, dir={benchmark_dir}, "
            f"split={num_train}/{val_size}/{test_size}"
        )

    return Y_df, freq, val_size, test_size, metric_scaler, meta


def build_fhsd_config(dataset_lower: str, horizon: int) -> Dict[str, Any]:
    if dataset_lower == "solar" and horizon in [96, 192, 336, 720]:
        if horizon in [96, 192, 336]:
            input_size = 4 * horizon
            max_steps = 800
            windows_batch_size = 128
        else:
            input_size = 3 * horizon
            max_steps = 1200
            windows_batch_size = 64

        return {
            "learning_rate": tune.choice([1e-3]),
            "max_steps": tune.choice([max_steps]),
            "input_size": tune.choice([input_size]),
            "batch_size": tune.choice([1]),
            "windows_batch_size": tune.choice([windows_batch_size]),
            "n_pool_kernel_size": tune.choice([[2, 2, 2]]),
            "n_freq_downsample": tune.choice([[1, 1, 1]]),
            "pooling_mode": tune.choice(["MaxPool1d"]),
            "freq_pooling_kwargs": None,
            "dropout_prob_theta": tune.choice([0.3]),
            "activation": tune.choice(["ReLU"]),
            "n_blocks": tune.choice([[1, 1, 1]]),
            "mlp_units": tune.choice([[[256, 256], [256, 256], [256, 256]]]),
            "interpolation_mode": tune.choice(["linear"]),
            "use_frequency_interpolation": False,
            "enable_self_distill": tune.choice([False]),
            "self_distill_weight": tune.choice([0.1]),
            "val_check_steps": tune.choice([100]),
            "random_seed": tune.randint(1, 10),
        }

    return {
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "max_steps": tune.choice([200, 1000]),
        "input_size": tune.choice([7 * horizon]),
        "batch_size": tune.choice([7]),
        "windows_batch_size": tune.choice([256]),
        "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),
        "n_freq_downsample": tune.choice([
            [(96 * 7) // 2, 96 // 2, 1],
            [(24 * 7) // 2, 24 // 2, 1],
            [1, 1, 1],
        ]),
        "pooling_mode": tune.choice(["MaxPool1d", "frequency_dynamic"]),
        "freq_pooling_kwargs": tune.choice([
            None,
            {"use_stft": False, "num_freq_bands": 4, "preserve_energy": True},
            {"use_stft": True, "stft_window_size": 32, "num_freq_bands": 4, "preserve_energy": True},
        ]),
        "dropout_prob_theta": tune.choice([0.5]),
        "activation": tune.choice(["ReLU"]),
        "n_blocks": tune.choice([[1, 1, 1]]),
        "mlp_units": tune.choice([[[512, 512], [512, 512], [512, 512]]]),
        "interpolation_mode": tune.choice(["linear"]),
        "use_frequency_interpolation": True,
        "enable_self_distill": tune.choice([True, False]),
        "self_distill_weight": tune.choice([0.05, 0.1, 0.2]),
        "val_check_steps": tune.choice([100]),
        "random_seed": tune.randint(1, 10),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHsd long-horizon forecasting experiments")
    parser.add_argument("-horizon", "--horizon", type=int, required=True)
    parser.add_argument("-dataset", "--dataset", type=str, required=True)
    parser.add_argument("-num_samples", "--num_samples", default=5, type=int)
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
        help="Data root directory (default: data/longhorizon2)",
    )
    parser.add_argument(
        "--pems07_npz",
        type=str,
        default=os.path.join(DATA_ROOT, "pems07", "PEMS07.npz"),
        help="Path to PEMS07 npz file",
    )
    parser.add_argument(
        "--solar_path",
        type=str,
        default=os.path.join(DATA_ROOT, "solar", "solar_AL.txt"),
        help="Path to Solar txt file",
    )
    parser.add_argument(
        "--solar_n_time",
        type=int,
        default=52560,
        help="Number of time steps to use for Solar; -1 uses the full series",
    )
    parser.add_argument("--solar_start_date", type=str, default="2006-01-01 00:00:00")
    parser.add_argument("--solar_freq", type=str, default="15min")
    parser.add_argument(
        "--metric_scale",
        type=str,
        default="scaled",
        choices=["scaled", "original", "both"],
        help="Metric scale: scaled, original, or both",
    )

    args = parser.parse_args()
    horizon = args.horizon
    dataset = args.dataset
    dataset_lower = dataset.lower()
    num_samples = args.num_samples

    assert horizon in [12, 24, 36, 48, 60, 96, 192, 336, 720]

    Y_df, freq, val_size, test_size, metric_scaler, _ = load_dataset(args)
    fhsd_config = build_fhsd_config(dataset_lower, horizon)

    models = [
        AutoFHsd(
            h=horizon,
            loss=HuberLoss(delta=0.5),
            valid_loss=MAE(),
            config=fhsd_config,
            num_samples=num_samples,
            refit_with_val=True,
        )
    ]

    nf = NeuralForecast(models=models, freq=freq)
    Y_hat_df = nf.cross_validation(
        df=Y_df,
        val_size=val_size,
        test_size=test_size,
        n_windows=None,
    )

    y_true = Y_hat_df["y"].to_numpy()
    y_hat = Y_hat_df["AutoFHsd"].to_numpy()
    unique_ids = Y_hat_df["unique_id"] if "unique_id" in Y_hat_df.columns else None

    mse_value, mae_value, mse_scaled, mae_scaled, mse_original, mae_original = compute_forecast_metrics(
        y_true=y_true,
        y_hat=y_hat,
        unique_ids=unique_ids,
        scaler=metric_scaler,
        metric_scale=args.metric_scale,
    )

    n_series = len(Y_df.unique_id.unique())
    print("\n" * 4)
    print("Parsed results")
    print(f"FHsd {dataset} h={horizon}")
    print("test_size", test_size)
    print(f"n_series: {n_series}, n_forecasts: {len(y_true)}")
    print(f"y_true min/max: {np.nanmin(y_true):.6f} / {np.nanmax(y_true):.6f}")
    print(f"y_hat min/max: {np.nanmin(y_hat):.6f} / {np.nanmax(y_hat):.6f}")
    print("MSE: ", mse_value)
    print("MAE: ", mae_value)
    if args.metric_scale == "both":
        print("MSE(scaled): ", mse_scaled)
        print("MAE(scaled): ", mae_scaled)
        print("MSE(original): ", mse_original)
        print("MAE(original): ", mae_original)

    result_file = os.path.join(FHSD_ROOT, "results.txt")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(
            f"Dataset={dataset}, horizon={horizon}, num_samples={num_samples} | "
            f"MSE={mse_value:.6f}, MAE={mae_value:.6f}\n"
        )
