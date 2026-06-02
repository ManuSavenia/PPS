#!/usr/bin/env python3
import re
import numpy as np
from pathlib import Path

Q_SCALE = 1 << 15

def parse_example_qp(example_h):
    txt = example_h.read_text()
    m_scale = re.search(r"\(int32_t\)\s*\(\s*([0-9.eE+\-]+)\s*\*\s*Q_SCALE", txt)
    m_zp = re.search(r"\*Q_SCALE\)\s*,.*\n\s*([0-9]+)\s*//\s*Punto cero", txt)
    if not m_scale or not m_zp:
        raise RuntimeError('sample_data_qp not found')
    scale_f = float(m_scale.group(1).strip())
    zp = int(m_zp.group(1))
    return {'scale_q': int(round(scale_f * Q_SCALE)), 'zero_point': zp, 'scale': scale_f}

def parse_sample_data(example_h, max_samples=10):
    txt = example_h.read_text()
    # find lines with a single sample initializer like '{   107, -70, 127, -128, 92, 49 }'
    lines = re.findall(r"^\s*\{[^}]+\}", txt, flags=re.M)
    samples = []
    for a in lines[:max_samples]:
        vals = [int(x) for x in re.findall(r"-?\d+", a)]
        samples.append(vals)
    return np.array(samples, dtype=np.int32)

def parse_layer_data(c_file, layer_name):
    txt = c_file.read_text()
    # find the init function for the specific layer
    func_re = rf"dense_layer_t\s+init_{layer_name}_data\s*\(void\)\s*\{{(.*?)return layer;\s*\}}"
    m = re.search(func_re, txt, re.S)
    if not m:
        raise RuntimeError(f'init function for {layer_name} not found')
    body = m.group(1)
    w_m = re.search(r"weights\[\]\s*=\s*\{\s*([^}]+)\}\s*;", body, re.S)
    b_m = re.search(r"biases\[\]\s*=\s*\{\s*([^}]+)\}\s*;", body, re.S)
    struct_m = re.search(r"weights, biases, \{\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\}, \{\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\}", body)
    if not w_m or not b_m or not struct_m:
        raise RuntimeError('Could not parse weights/biases/qparams in layer function')
    def parse_array_text(s):
        # remove C-style comments first
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
        s = re.sub(r"//.*?$", "", s, flags=re.M)
        vals = []
        for tok in s.split(','):
            tok = tok.strip()
            if not tok:
                continue
            m2 = re.match(r"^([-+]?\d+)", tok)
            if m2:
                vals.append(int(m2.group(1)))
        return vals

    w_vals = parse_array_text(w_m.group(1))
    bias_vals = parse_array_text(b_m.group(1))
    # debug: show raw snippet
    raw_snip = w_m.group(1)[:200].replace('\n','\\n')
    print('[debug parse] layer=', layer_name, ' raw_weights[:200]:\n', raw_snip)
    w_scale_q = int(struct_m.group(1)); w_zp = int(struct_m.group(2)); out_scale_q = int(struct_m.group(3)); out_zp = int(struct_m.group(4))
    return w_vals, bias_vals, (w_scale_q, w_zp), (out_scale_q, out_zp)

def simulate_integer_dense(input_q, in_qp, weights_q, biases_int32, w_qp, out_qp, input_shape, output_shape):
    # input_q: 1D int array length input_shape
    in_zp = in_qp['zero_point']; w_zp = w_qp[1]; out_zp = out_qp[1]
    in_scale_q = in_qp['scale_q']; w_scale_q = w_qp[0]; out_scale_q = out_qp[0]
    QUANT_SCALE_ONE = 1 << 15
    multiplier_fixed = 1
    if out_scale_q != 0:
        multiplier_fixed = (in_scale_q * w_scale_q) // out_scale_q
        if multiplier_fixed == 0:
            multiplier_fixed = 1

    weights_q = weights_q.reshape((output_shape, input_shape))
    outputs_q = np.zeros(output_shape, dtype=np.int32)
    accs = np.zeros(output_shape, dtype=np.int64)
    for i in range(output_shape):
        acc = 0
        for j in range(input_shape):
            acc += (int(input_q[j]) - in_zp) * (int(weights_q[i, j]) - w_zp)
        acc += int(biases_int32[i])
        accs[i] = acc
        tmp = acc * multiplier_fixed + (QUANT_SCALE_ONE // 2)
        scaled = int(tmp // QUANT_SCALE_ONE)
        out_q = scaled + out_zp
        outputs_q[i] = out_q
    return outputs_q, accs

def main():
    base = Path(__file__).parent
    example_h = base / 'embedia' / 'example_file.h'
    model_c = base / 'embedia' / 'sequential_3_model.c'

    qp = parse_example_qp(example_h)
    samples_q = parse_sample_data(example_h, max_samples=10)

    # parse layer data for first layer from model_c
    w_vals, b_vals, w_qp, out_qp = parse_layer_data(model_c, 'dense_6')
    print('parsed weights count', len(w_vals), 'bias count', len(b_vals))
    # debug raw snippet
    print('raw weights snippet:', w_m.group(1)[:200].replace('\n','\\n'))
    print('first weights:', w_vals[:12])
    print('first bias:', b_vals[:12])
    w_vals = np.array(w_vals, dtype=np.int64)
    b_vals = np.array(b_vals, dtype=np.int64)
    # deduce shapes: from known model: 6 inputs, 8 neurons
    in_sz = 6; out_sz = 8

    print('Layer dense_6')
    for si, s in enumerate(samples_q):
        inputs_q = s
        out_q, accs = simulate_integer_dense(inputs_q, qp, w_vals, b_vals, w_qp, out_qp, in_sz, out_sz)
        print(f'Sample {si}: accs={accs.tolist()} out_q={out_q.tolist()}')

    # For layer 7 parse second occurrence of weights/biases
    txt_all = model_c.read_text()
    all_w = re.findall(r"static EMBEDIA_MODEL_STORAGE quant8 weights\[\] = \{\s*([^}]+)\}\s*;", txt_all, re.S)
    all_b = re.findall(r"static EMBEDIA_MODEL_STORAGE int32_t biases\[\] = \{\s*([^}]+)\}\s*;", txt_all, re.S)
    structs = re.findall(r"weights, biases, \{\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\}, \{\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\}", txt_all)
    if len(all_w) >= 2 and len(all_b) >= 2 and len(structs) >= 2:
        w2_vals = np.array([int(x) for x in re.findall(r"-?\d+", all_w[1])], dtype=np.int32)
        b2_vals = np.array([int(x) for x in re.findall(r"-?\d+", all_b[1])], dtype=np.int32)
        w2_qp = (int(structs[1][0]), int(structs[1][1]))
        out2_qp = (int(structs[1][2]), int(structs[1][3]))
        in2 = out_sz; out2 = 6
        print('\nLayer dense_7')
        for si, s in enumerate(samples_q):
            inputs_q = out_q.astype(np.int32)
            out_q2, accs2 = simulate_integer_dense(inputs_q, {'scale_q': out_qp[0], 'zero_point': out_qp[1]}, w2_vals, b2_vals, w2_qp, out2_qp, in2, out2)
            print(f'Sample {si}: accs={accs2.tolist()} out_q={out_q2.tolist()}')

if __name__ == '__main__':
    main()
