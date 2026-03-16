from pathlib import Path

import numpy as np
import pandas as pd


def quantize_symmetric_signed(x: np.ndarray, bits: int = 8, axis: int = 0, eps: float = 1e-12):
    qmax = (2 ** (bits - 1)) - 1
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(max_abs < eps, 1.0, max_abs / qmax)
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int32)
    dq = (q.astype(np.float32) * scale).astype(np.float32)
    return q, dq, scale


def quantize_csv_inputs(src_csv: Path, dst_csv: Path, bits: int = 8):
    df = pd.read_csv(src_csv)
    x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy()

    q, _, _ = quantize_symmetric_signed(x, bits=bits, axis=0)

    out = pd.DataFrame(q, columns=df.columns[:-1])
    out[df.columns[-1]] = y
    out.to_csv(dst_csv, index=False)


def main():
    root = Path(__file__).resolve().parents[1]
    data_sets = root / "Data_Sets"

    sources = ["fingers_train.csv", "fingers_test.csv"]
    for name in sources:
        src = data_sets / name
        dst = data_sets / name.replace(".csv", "_quant8_signed_symmetric.csv")
        quantize_csv_inputs(src, dst, bits=8)
        print(f"created: {dst}")


if __name__ == "__main__":
    main()
