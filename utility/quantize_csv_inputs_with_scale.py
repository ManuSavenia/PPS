import pandas as pd

from utility.quantize_symmetric_signed import quantize_symmetric_signed


def quantize_csv_inputs_with_scale(src_csv, dst_csv, scaler, bits=8):
    df = pd.read_csv(src_csv)
    x_raw = df.iloc[:, :-1].to_numpy(dtype="float32")
    y = df.iloc[:, -1].to_numpy()

    x_norm = scaler.transform(x_raw)
    q_x, _, scale = quantize_symmetric_signed(x_norm, bits=bits, axis=0)

    out = pd.DataFrame(q_x, columns=df.columns[:-1])
    out[df.columns[-1]] = y
    out.to_csv(dst_csv, index=False)
    return scale.astype("float32"), list(df.columns[:-1])
