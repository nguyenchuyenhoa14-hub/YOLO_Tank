"""
KIIT-MiTA Tank - PCQ-Aware Fine-tuning
Bỏ QAT. Thay vào đó:
  Step 1: Export Phase 3 ReLU → ONNX → PCQ → lấy INT8 scales
  Step 2: Import PCQ scales vào PyTorch FakeQuantize
  Step 3: Fine-tune 100 epochs với FakeQuantize (PCQ scales frozen)
  Step 4: Export lại → PCQ → validate → .mem
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, math, cv2, onnx
from pathlib import Path
from collections import OrderedDict
from onnx import numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import ultralytics.nn.modules.conv as conv_modules
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

conv_modules.Conv.default_act = nn.ReLU()
from ultralytics import YOLO

# ====================================================================
# Config
# ====================================================================
RELU_PT = r'v:\SRNet\runs\detect\kiitmita_tank\phase3_relu_kd\weights\best.pt'
DATA_YAML = r'v:/SRNet/datasets/KIIT-MiTA-Tank/kiitmita_tank.yaml'
CAL_FOLDER = r'v:\SRNet\datasets\KIIT-MiTA-Tank\train\images'
PROJECT = r'v:/SRNet/runs/detect/kiitmita_tank'
IMGSZ = 320
FINETUNE_EPOCHS = 200

class TankCalReader(CalibrationDataReader):
    def __init__(self, folder, model_path, img_size=320, num_samples=200):
        self.img_size = img_size
        files = (list(Path(folder).glob('*.jpeg')) + list(Path(folder).glob('*.jpg'))
                 + list(Path(folder).glob('*.png')) + list(Path(folder).glob('*.JPEG')))
        self.files = files[:num_samples]
        self.idx = 0
        m = onnx.load(model_path)
        self.input_name = m.graph.input[0].name
    def get_next(self):
        if self.idx >= len(self.files): return None
        img = cv2.imread(str(self.files[self.idx]))
        self.idx += 1
        if img is None: return self.get_next()
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return {self.input_name: img}

if __name__ == '__main__':
    # ================================================================
    # STEP 1: Export ReLU → ONNX → PCQ → Extract scales
    # ================================================================
    print("=" * 70)
    print("  STEP 1: Get PCQ scales from Phase 3 ReLU model")
    print("=" * 70)

    # Export to ONNX
    model_tmp = YOLO(RELU_PT)
    model_tmp.export(format='onnx', imgsz=IMGSZ, dynamic=False, simplify=False)
    onnx_fp32 = RELU_PT.replace('.pt', '.onnx')
    onnx_int8 = RELU_PT.replace('.pt', '_pcq_cal.onnx')

    # Run PCQ to get scales
    reader = TankCalReader(CAL_FOLDER, onnx_fp32)
    quantize_static(model_input=onnx_fp32, model_output=onnx_int8,
                    calibration_data_reader=reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, reduce_range=True)

    # Parse INT8 ONNX to extract all scales
    pcq_model = onnx.load(onnx_int8)
    pcq_init = {i.name: numpy_helper.to_array(i) for i in pcq_model.graph.initializer}

    # Build map: conv_weight_name → (weight_scale, act_input_scale, act_output_scale)
    pcq_scales = {}
    for node in pcq_model.graph.node:
        if node.op_type == 'Conv':
            w_name = node.input[1]  # weight input
            # Find DequantizeLinear feeding the weight
            for n2 in pcq_model.graph.node:
                if n2.op_type == 'DequantizeLinear' and n2.output[0] == w_name:
                    w_scale_name = n2.input[1]
                    w_zp_name = n2.input[2] if len(n2.input) > 2 else None
                    if w_scale_name in pcq_init:
                        pcq_scales[w_name] = {
                            'w_scale': pcq_init[w_scale_name],
                            'w_zp': pcq_init.get(w_zp_name, np.zeros(1, dtype=np.int8)) if w_zp_name else np.zeros(1, dtype=np.int8)
                        }
                    break

    print(f"  Extracted PCQ scales for {len(pcq_scales)} conv layers")
    del model_tmp, pcq_model

    # ================================================================
    # STEP 2: Import PCQ scales into PyTorch FakeQuantize + Fine-tune
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  STEP 2: PCQ-Aware Fine-tuning ({FINETUNE_EPOCHS} epochs)")
    print("=" * 70)

    model = YOLO(RELU_PT)
    device = next(model.model.parameters()).device

    # Map ONNX weight names to PyTorch module names
    # ONNX names look like: "/model.0/conv/Conv_output_0_DequantizeLinear"
    # We need to match them to PyTorch conv modules

    # Build FakeQuantize hooks using PCQ scales
    fake_quant_modules = {}
    hook_handles = []
    quant_enabled = False

    # Get all conv modules in order
    conv_modules_list = []
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_modules_list.append((name, module))

    # Match with PCQ scales by order
    pcq_scale_list = list(pcq_scales.values())
    matched = min(len(conv_modules_list), len(pcq_scale_list))

    for i in range(matched):
        pt_name, conv = conv_modules_list[i]
        pcq_s = pcq_scale_list[i]
        ln = pt_name.replace('.', '_')

        # Create weight FakeQuantize with PCQ scales
        w_scale_tensor = torch.tensor(pcq_s['w_scale'], dtype=torch.float32).to(device)
        w_zp_tensor = torch.tensor(pcq_s['w_zp'], dtype=torch.int32).to(device)

        # Per-channel or per-tensor based on scale shape
        out_ch = conv.weight.shape[0]
        if w_scale_tensor.numel() == out_ch:
            # Per-channel
            wfq = FakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric, ch_axis=0,
            )().to(device)
        else:
            wfq = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
            )().to(device)

        # Calibrate once to initialize, then override with PCQ scales
        _ = wfq(conv.weight.data)
        with torch.no_grad():
            if hasattr(wfq, 'scale') and w_scale_tensor.shape == wfq.scale.shape:
                wfq.scale.copy_(w_scale_tensor)
                wfq.zero_point.copy_(w_zp_tensor.to(wfq.zero_point.dtype))
        # Freeze the observer - we want to keep PCQ scales fixed
        wfq.disable_observer()
        wfq.enable_fake_quant()

        fake_quant_modules[f"{ln}_weight"] = wfq

        # Hook: apply FakeQuantize to weights during forward
        def make_pre(conv_mod, wfq_mod):
            def pre(module, inputs):
                if not quant_enabled:
                    return inputs
                module._orig_w = module.weight.data.clone()
                module.weight.data = wfq_mod(module.weight.data)
                return inputs
            return pre

        def make_post(conv_mod):
            def post(module, inputs, output):
                if hasattr(module, '_orig_w'):
                    module.weight.data = module._orig_w
                    del module._orig_w
                return output
            return post

        hook_handles.append(conv.register_forward_pre_hook(make_pre(conv, wfq)))
        hook_handles.append(conv.register_forward_hook(make_post(conv)))

    print(f"  Matched {matched} conv layers with PCQ scales")

    # Callbacks
    FP32_WARMUP = 5

    def on_train_epoch_start(trainer):
        global quant_enabled
        if trainer.epoch == FP32_WARMUP and not quant_enabled:
            quant_enabled = True
            print(f"\n  >>> EPOCH {FP32_WARMUP}: PCQ FakeQuantize ENABLED <<<")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    # Fine-tune
    model.train(
        data=DATA_YAML,
        epochs=FINETUNE_EPOCHS, imgsz=IMGSZ, batch=32, device=0,
        lr0=0.0005, lrf=0.001, cos_lr=True, warmup_epochs=3,
        project=PROJECT, name='pcq_finetune_200',
        val=True, deterministic=True, workers=0, cache=False,
        patience=0,
    )

    # Remove hooks
    for h in hook_handles:
        h.remove()

    print(f"\n  PCQ-Aware Fine-tuning DONE!")

    # ================================================================
    # STEP 3: Export fine-tuned model → ONNX → PCQ → Validate
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  STEP 3: Export fine-tuned → PCQ → Validate")
    print("=" * 70)

    ft_pt = os.path.join(PROJECT, 'pcq_finetune_200', 'weights', 'best.pt')
    ft_model = YOLO(ft_pt)
    ft_model.export(format='onnx', imgsz=IMGSZ, dynamic=False, simplify=False)
    ft_onnx_fp32 = ft_pt.replace('.pt', '.onnx')
    ft_onnx_int8 = ft_pt.replace('.pt', '_int8_pcq.onnx')

    reader2 = TankCalReader(CAL_FOLDER, ft_onnx_fp32)
    quantize_static(model_input=ft_onnx_fp32, model_output=ft_onnx_int8,
                    calibration_data_reader=reader2, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, reduce_range=True)

    # Validate FP32
    print("\n  Validating FP32...")
    r_fp32 = ft_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=32, device=0, workers=0)

    # Validate PCQ INT8
    print("\n  Validating PCQ INT8...")
    r_int8 = YOLO(ft_onnx_int8).val(data=DATA_YAML, imgsz=IMGSZ, batch=32, workers=0)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS:")
    print(f"    PCQ-FT FP32:   P={r_fp32.box.mp:.4f} R={r_fp32.box.mr:.4f} mAP50={r_fp32.box.map50:.4f} mAP50-95={r_fp32.box.map:.4f}")
    print(f"    PCQ-FT INT8:   P={r_int8.box.mp:.4f} R={r_int8.box.mr:.4f} mAP50={r_int8.box.map50:.4f} mAP50-95={r_int8.box.map:.4f}")
    print(f"    Drop: {(r_fp32.box.map - r_int8.box.map)*100:.2f}%")
    print(f"")
    print(f"    (vs old PCQ from ReLU:  mAP50-95 = 0.376)")
    print(f"    (vs old PCQ from QAT:   mAP50-95 = 0.363)")
    print(f"={'=' * 69}")
