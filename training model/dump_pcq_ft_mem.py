"""
Dump .mem files from PCQ-FT INT8 ONNX model for RTL.
Uses the same format as the original tank_phase5_export.py.
Input: pcq_finetune_200/weights/best_int8_pcq.onnx
"""
import onnx
from onnx import numpy_helper
import os, math
import numpy as np

if __name__ == '__main__':
    onnx_int8 = r'v:\SRNet\runs\detect\kiitmita_tank\pcq_finetune_200\weights\best_int8_pcq.onnx'
    out_dir = r'v:\SRNet\runs\detect\kiitmita_tank\yolov8n_tank_pcq_finetune_mem'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Dumping .mem from PCQ-FT INT8 model")
    print("=" * 60)

    model_onnx = onnx.load(onnx_int8)
    init_dict = {i.name: numpy_helper.to_array(i) for i in model_onnx.graph.initializer}

    def get_act_scales(model_onnx, w_name, init_dict):
        for node in model_onnx.graph.node:
            if node.op_type == 'Conv' and node.input[1] == w_name:
                x, y = node.input[0], node.output[0]
                si, so = 1.0, 1.0
                dq = next((n for n in model_onnx.graph.node if x in n.output), None)
                if dq and dq.op_type == 'DequantizeLinear' and dq.input[1] in init_dict:
                    si = float(init_dict[dq.input[1]])
                q = next((n for n in model_onnx.graph.node if y in n.input), None)
                if q and q.op_type == 'QuantizeLinear' and q.input[1] in init_dict:
                    so = float(init_dict[q.input[1]])
                return si, so
        return 1.0, 1.0

    lc = 0
    for name, data in init_dict.items():
        if 'weight_quantized' in name and 'scale' not in name and 'zero_point' not in name:
            base = name.replace('_quantized', '')
            safe = base.replace(".", "_").replace("/", "_")
            lc += 1

            # Weight quantized
            with open(os.path.join(out_dir, f"{safe}_quantized.mem"), "w") as f:
                for w in data.flatten():
                    f.write(f"{int(w) & 0xFF:02x}\n")

            # Scale, shift, multiplier
            sw_name = base + "_scale"
            sw_arr = np.atleast_1d(init_dict.get(sw_name, np.array([1.0])))
            si, so = get_act_scales(model_onnx, name, init_dict)

            with open(os.path.join(out_dir, f"{safe}_scale.mem"), "w") as fs, \
                 open(os.path.join(out_dir, f"{safe}_shift.mem"), "w") as fsh, \
                 open(os.path.join(out_dir, f"{safe}_multiplier.mem"), "w") as fm:
                for sw in sw_arr.flatten():
                    fs.write(f"{np.float32(sw).view(np.uint32):08x}\n")
                    rm = (si * sw) / so if so != 0 else 0
                    if rm > 0:
                        mv, ev = math.frexp(rm)
                        shift = int(round(-math.log2(rm)))
                        m0 = int(round(mv * (2 ** 31)))
                    else:
                        shift, m0 = 0, 0
                    fsh.write(f"{shift & 0xFF:02x}\n")
                    fm.write(f"{m0 & 0xFFFFFFFF:08x}\n")

            # Zero point
            zp_name = base + "_zero_point"
            if zp_name in init_dict:
                zp = np.atleast_1d(init_dict[zp_name])
                with open(os.path.join(out_dir, f"{safe}_zero.mem"), "w") as fz:
                    for z in zp.flatten():
                        fz.write(f"{int(z) & 0xFF:02x}\n")

            print(f"  [{lc:3d}] {safe}")

    # Bias
    for name, data in init_dict.items():
        if 'bias_quantized' in name and 'scale' not in name and 'zero_point' not in name:
            safe = name.replace('_quantized', '').replace(".", "_").replace("/", "_")
            with open(os.path.join(out_dir, f"{safe}_quantized.mem"), "w") as f:
                for w in data.flatten():
                    f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")

    total = len(os.listdir(out_dir))
    print(f"\n  DONE! {lc} layers, {total} .mem files")
    print(f"  Output: {out_dir}")
