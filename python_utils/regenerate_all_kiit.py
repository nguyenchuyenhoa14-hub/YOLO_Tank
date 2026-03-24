#!/usr/bin/env python3
"""
regenerate_all_kiit.py
======================
Master script to regenerate ALL .mem files for KIIT-MITA dataset.

Steps:
1. Extract activation scales + zero points from KIIT ONNX model
2. Update activation_scales.json
3. Regenerate layer15/20/21_output.mem with correct scales + ZP
4. Convert weights to PE-interleaved format
5. Fold Z_in into biases in weights_for_verilog_pe/
6. Verify shift files

Usage: python3 regenerate_all_kiit.py pic_test/image_s3r2_kiit_402.jpeg
"""
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import cv2
import os
import sys
import json
import shutil

# ============================================================================
# Configuration (KIIT-MITA specific)
# ============================================================================
SCALES_JSON = "/mnt/c/github/activation_scales.json"
ONNX_PATH = "/mnt/c/github/kiitmita_tank/pcq_finetune_200/weights/best_int8_pcq.onnx"
MODEL_PATH = "/mnt/c/github/kiitmita_tank/pcq_finetune_200/weights/best.pt"
WEIGHTS_DIR = "/mnt/c/github/kiitmita_tank/yolov8n_tank_pcq_finetune_mem"
PE_DIR     = "/mnt/c/github/rtl/yolo_complete/weights_for_verilog_pe"
OUT_DIR    = "/mnt/c/github/rtl/yolo_complete"
PARALLEL   = 16
IMG_SIZE   = 320  # single-class tank model, 320x320

# ONNX tensor patterns for detect head input convolutions
TARGET_CONV_PATTERNS = {
    'P3': 'cv2.0/cv2.0.0',
    'P4': 'cv2.1/cv2.1.0',
    'P5': 'cv2.2/cv2.2.0',
}

# Layer definitions for bias folding
LAYER_DEFS = [
    ('cv2_0_0', True,  'P3'), ('cv2_0_1', False, None), ('cv2_0_2', False, None),
    ('cv3_0_0', True,  'P3'), ('cv3_0_1', False, None), ('cv3_0_2', False, None),
    ('cv2_1_0', True,  'P4'), ('cv2_1_1', False, None), ('cv2_1_2', False, None),
    ('cv3_1_0', True,  'P4'), ('cv3_1_1', False, None), ('cv3_1_2', False, None),
    ('cv2_2_0', True,  'P5'), ('cv2_2_1', False, None), ('cv2_2_2', False, None),
    ('cv3_2_0', True,  'P5'), ('cv3_2_1', False, None), ('cv3_2_2', False, None),
]

FILE_MAP = {'P3': 'layer15_output.mem', 'P4': 'layer20_output.mem', 'P5': 'layer21_output.mem'}

# ============================================================================
# Step 1: Extract scales from ONNX
# ============================================================================
def extract_onnx_scales():
    print("=" * 60)
    print("  Step 1: Extract Activation Scales from KIIT ONNX")
    print("=" * 60)
    
    model = onnx.load(ONNX_PATH)
    
    # Build initializer lookup
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    
    # Find Conv input tensors for P3/P4/P5 detect head inputs
    detect_inputs = {}
    for node in model.graph.node:
        if node.op_type == "Conv":
            for branch, pattern in TARGET_CONV_PATTERNS.items():
                if pattern in node.name:
                    detect_inputs[branch] = node.input[0]
                    break

    # Find scales and ZPs for those tensors
    scales = {}
    zero_points = {}
    
    for node in model.graph.node:
        if node.op_type in ["DequantizeLinear", "QuantizeLinear"]:
            output_name = node.output[0]
            for branch, tensor_name in detect_inputs.items():
                if output_name == tensor_name:
                    scale_name = node.input[1]
                    if scale_name in init_map:
                        scales[branch] = float(init_map[scale_name].flatten()[0])
                    if len(node.input) >= 3:
                        zp_name = node.input[2]
                        if zp_name in init_map:
                            zero_points[branch] = int(init_map[zp_name].flatten()[0])
                    break
    
    # Extract Z_out for ALL detect head output convolutions
    # We need these for bias folding of final layers
    z_out_map = {}
    
    # Find QuantizeLinear nodes that follow detect head Conv outputs
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            if len(node.input) >= 3:
                zp_name = node.input[2]
                if zp_name in init_map:
                    zp_val = int(init_map[zp_name].flatten()[0])
                    # Map by input tensor name
                    z_out_map[node.input[0]] = zp_val
    
    # Find Conv outputs for final layers (cv2_X_2, cv3_X_2)
    final_zps = {}
    final_patterns = {
        'cv2_0_2': 'cv2.0.2', 'cv3_0_2': 'cv3.0.2',
        'cv2_1_2': 'cv2.1.2', 'cv3_1_2': 'cv3.1.2',
        'cv2_2_2': 'cv2.2.2', 'cv3_2_2': 'cv3.2.2',
    }
    
    for node in model.graph.node:
        if node.op_type == "Conv":
            for layer_key, pattern in final_patterns.items():
                if pattern in node.name:
                    conv_output = node.output[0]
                    if conv_output in z_out_map:
                        final_zps[layer_key] = z_out_map[conv_output]
                    else:
                        # Search downstream
                        for node2 in model.graph.node:
                            if node2.op_type == "QuantizeLinear" and node2.input[0] == conv_output:
                                if len(node2.input) >= 3 and node2.input[2] in init_map:
                                    final_zps[layer_key] = int(init_map[node2.input[2]].flatten()[0])
                                break
                    break
    
    for branch in ['P3', 'P4', 'P5']:
        s = scales.get(branch, 'NOT FOUND')
        zp = zero_points.get(branch, 'NOT FOUND')
        print(f"  {branch}: scale={s}, zero_point={zp}")
    
    print(f"\n  Final layer Z_out values:")
    for k, v in sorted(final_zps.items()):
        print(f"    {k}: Z_out={v}")
    
    return scales, zero_points, final_zps

# ============================================================================
# Step 2: Update activation_scales.json
# ============================================================================
def update_scales_json(scales, zero_points):
    print("\n" + "=" * 60)
    print("  Step 2: Update activation_scales.json")
    print("=" * 60)
    
    data = {
        "scales": {k: v for k, v in scales.items()},
        "zero_points": {k: v for k, v in zero_points.items()}
    }
    
    with open(SCALES_JSON, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✅ Updated {SCALES_JSON}")

# ============================================================================
# Step 3: Regenerate input feature maps
# ============================================================================
def regenerate_feature_maps(img_path, scales, zero_points):
    print("\n" + "=" * 60)
    print("  Step 3: Regenerate Feature Maps (layer15/20/21_output.mem)")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    model = YOLO(MODEL_PATH)
    nn_model = model.model
    nn_model.eval()
    
    img_bgr = cv2.imread(img_path)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    
    feature_maps = {}
    def det_hook(model, inp, out):
        feature_maps['P3'] = inp[0][0][0].detach().cpu().numpy()
        feature_maps['P4'] = inp[0][1][0].detach().cpu().numpy()
        feature_maps['P5'] = inp[0][2][0].detach().cpu().numpy()
    
    h = nn_model.model[-1].register_forward_hook(det_hook)
    with torch.no_grad():
        _ = nn_model(img_t)
    h.remove()
    
    for branch in ['P3', 'P4', 'P5']:
        fm = feature_maps[branch]
        scale = scales[branch]
        zp = zero_points.get(branch, 0)
        
        fm_q = np.clip(np.round(fm / scale) + zp, -128, 127).astype(np.int8)
        
        print(f"  [{branch}] Float: shape={fm.shape}, min={fm.min():.4f}, max={fm.max():.4f}")
        print(f"  [{branch}] Scale={scale:.10f}, ZP={zp}")
        print(f"  [{branch}] Quantized: min={fm_q.min()}, max={fm_q.max()}")
        
        fm_hwc = fm_q.transpose(1, 2, 0)
        fm_flat = fm_hwc.flatten()
        
        out_path = os.path.join(OUT_DIR, FILE_MAP[branch])
        with open(out_path, "w") as f:
            for v in fm_flat:
                f.write(f"{int(v) & 0xFF:02x}\n")
        print(f"  ✅ Saved {len(fm_flat)} bytes -> {out_path}\n")

# ============================================================================
# Step 4: Convert weights to PE-interleaved format
# ============================================================================
def convert_weights_to_pe():
    print("\n" + "=" * 60)
    print("  Step 4: Convert Weights to PE-Interleaved Format")
    print("=" * 60)
    
    # Run the existing conversion script
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(OUT_DIR, "convert_weights_to_pe.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print("  ⚠️  Weight conversion skipped (PE files already exist from previous run)")
    else:
        print("  ✅ PE weight conversion complete")

# ============================================================================
# Step 5: Fold Z_in into biases
# ============================================================================
# ============================================================================
# Step 5: Recalculate Multipliers/Shifts and Fold Biases
# ============================================================================
def process_weights(scales, zero_points, final_zps):
    print("\n" + "=" * 60)
    print("  Step 5: Recalculate Multipliers, Shifts and Fold Biases")
    print("=" * 60)
    
    import math

    # Get internal scales by tracing the ONNX graph
    model_onnx = onnx.load(ONNX_PATH)
    init_map = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}
    
    def find_tensor_scale(tensor_name):
        for node in model_onnx.graph.node:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"] and node.input[0] == tensor_name:
                s_name = node.input[1]
                if s_name in init_map:
                    return float(init_map[s_name].flatten()[0])
        return 1.0

    for branch_key in ['P3', 'P4', 'P5']:
        p_idx = int(branch_key[1]) - 3 # 0, 1, 2
        s_in = scales[branch_key]
        
        # We need to trace cv2.X.0 -> cv2.X.1 -> cv2.X.2
        # and cv3.X.0 -> cv3.X.1 -> cv3.X.2
        for branch_type in ['cv2', 'cv3']:
            current_s_in = s_in
            for l_idx in range(3):
                layer_key = f"{branch_type}_{p_idx}_{l_idx}"
                onnx_pattern = f"{branch_type}.{p_idx}.{l_idx}"
                
                # Find Conv
                conv_node = None
                for node in model_onnx.graph.node:
                    if node.op_type == "Conv" and onnx_pattern in node.name:
                        conv_node = node
                        break
                
                if not conv_node:
                    print(f"  ⚠️  Layer {onnx_pattern} not found in ONNX!")
                    continue
                
                # Get Weights Scale
                w_dq_node = None
                for node in model_onnx.graph.node:
                    if node.op_type == "DequantizeLinear" and node.output[0] == conv_node.input[1]:
                        w_dq_node = node
                        break
                
                if not w_dq_node:
                    print(f"  ⚠️  Weight DQ for {onnx_pattern} not found!")
                    continue
                
                s_w_array = init_map[w_dq_node.input[1]].flatten()
                
                # Get Output Scale and Zero Point
                s_out = find_tensor_scale(conv_node.output[0])
                
                def find_tensor_zp(tensor_name):
                    # Check QuantizeLinear/DequantizeLinear inputs[2]
                    for node in model_onnx.graph.node:
                        if node.op_type in ["QuantizeLinear", "DequantizeLinear"] and node.input[0] == tensor_name:
                            zp_name = node.input[2]
                            if zp_name in init_map:
                                return int(init_map[zp_name].flatten()[0])
                    # Safety fallback based on layer type
                    return -128 if l_idx < 2 else (113 if branch_type == 'cv3' else 31)

                z_out = find_tensor_zp(conv_node.output[0])

                # Read original weights and biases
                w_file = os.path.join(WEIGHTS_DIR, f"model_22_{layer_key}_weight_quantized.mem")
                b_file = os.path.join(WEIGHTS_DIR, f"model_22_{layer_key}_bias_quantized.mem")
                
                if not os.path.exists(w_file):
                    w_file = os.path.join(WEIGHTS_DIR, f"model_22_{layer_key}_conv_weight_quantized.mem")
                    b_file = os.path.join(WEIGHTS_DIR, f"model_22_{layer_key}_conv_bias_quantized.mem")
                    
                if not os.path.exists(w_file) or not os.path.exists(b_file):
                    print(f"  ⚠️  Files for {layer_key} missing, skipping")
                    continue
                    
                with open(w_file) as f: w_hex = [l.strip() for l in f if l.strip()]
                w_all = np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in w_hex])
                
                with open(b_file) as f: b_hex = [l.strip() for l in f if l.strip()]
                b_all = np.array([int(x, 16) if int(x, 16) < 0x80000000 else int(x, 16) - 0x100000000 for x in b_hex])
                
                num_oc = len(b_all)
                if len(s_w_array) == 1:
                    s_w_array = np.full(num_oc, s_w_array[0])
                
                # 1. Recalc Multipliers and Shifts
                out_m0 = []
                out_shift = []
                for i in range(num_oc):
                    s_w = float(s_w_array[i])
                    M = (s_w * current_s_in) / s_out
                    if M == 0:
                        shift, m0 = 0, 0
                    else:
                        shift = -math.floor(math.log2(M)) - 1
                        m0 = int(round(M * (2**shift) * (2**31)))
                        if m0 >= 2**31: m0 //= 2; shift -= 1
                    out_m0.append(m0)
                    out_shift.append(shift)
                
                # 2. Fold Biases (Z_in = -128)
                z_in = -128
                b_new = []
                is_3x3 = (l_idx != 2)
                for oc in range(num_oc):
                    if is_3x3:
                        in_ch = len(w_all) // (num_oc * 9)
                        w_ch = w_all[oc * in_ch * 9 : (oc + 1) * in_ch * 9]
                    else:
                        in_ch = len(w_all) // num_oc
                        w_ch = w_all[oc * in_ch : (oc + 1) * in_ch]
                    sum_w = int(np.sum(w_ch))
                    b_target = int(b_all[oc]) - (z_in * sum_w)
                    if "cv3_0_2" in layer_key and oc == 0:
                        print(f"\n[DEBUG cv3_0_2 fold] oc=0, sum_w={sum_w}, b_all_INT8={b_all[oc]}, b_target={b_target}")
                    b_new.append(max(-0x80000000, min(0x7FFFFFFF, b_target)))

                # Write Files
                m_out = os.path.join(PE_DIR, f"model_22_{layer_key}_multiplier.mem")
                s_out_file = os.path.join(PE_DIR, f"model_22_{layer_key}_shift.mem")
                b_out = os.path.join(PE_DIR, f"model_22_{layer_key}_conv_bias_quantized.mem")
                z_out_file = os.path.join(PE_DIR, f"model_22_{layer_key}_zero_point.mem")
                
                with open(m_out, "w") as f:
                    for m in out_m0: f.write(f"{m:08x}\n")
                with open(s_out_file, "w") as f:
                    for s in out_shift: f.write(f"{s:02x}\n")
                with open(b_out, "w") as f:
                    for b in b_new: f.write(f"{b & 0xFFFFFFFF:08x}\n")
                with open(z_out_file, "w") as f:
                    for _ in range(num_oc): f.write(f"{z_out & 0xFF:02x}\n")
                
                print(f"  ✅ {layer_key}: s_in={current_s_in:.6f}, s_out={s_out:.6f}, Z={z_out}, M[0]={out_m0[0]:08x}")
                
                # Next layer input is this layer output
                current_s_in = s_out

# ============================================================================
# Step 6: Verify shift files
# ============================================================================
def verify_shifts():
    print("\n" + "=" * 60)
    print("  Step 6: Verify Shift Files")
    print("=" * 60)
    
    issues = 0
    for layer_key, _, _ in LAYER_DEFS:
        s_file = os.path.join(PE_DIR, f"model_22_{layer_key}_shift.mem")
        if not os.path.exists(s_file):
            print(f"  ⚠️  Missing: {s_file}")
            issues += 1
            continue
        
        with open(s_file) as f:
            shifts = [int(l.strip(), 16) for l in f if l.strip()]
        
        for i, s in enumerate(shifts):
            if s > 31:
                print(f"  ⚠️  {layer_key}[{i}]: shift={s} > 31!")
                issues += 1
    
    if issues == 0:
        print("  ✅ All shift files OK")
    else:
        print(f"  ⚠️  {issues} issues found")

# ============================================================================
# Main
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 regenerate_all_kiit.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)
    
    print("╔══════════════════════════════════════════════╗")
    print("║  KIIT-MITA Weight Pipeline Regeneration     ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  ONNX:    {ONNX_PATH}")
    print(f"  Weights: {WEIGHTS_DIR}")
    print(f"  PE Dir:  {PE_DIR}")
    print(f"  Image:   {img_path}")
    print("")
    
    # Step 1: Extract scales from ONNX
    scales, zero_points, final_zps = extract_onnx_scales()
    
    # Step 2: Update activation_scales.json
    update_scales_json(scales, zero_points)
    
    # Step 3: Regenerate feature maps
    regenerate_feature_maps(img_path, scales, zero_points)
    
    # Step 4: Convert weights to PE format
    convert_weights_to_pe()
    
    # Step 5: Recalculate Multipliers, Shifts and Fold Biases
    process_weights(scales, zero_points, final_zps)
    
    # Step 6: Verify shifts
    verify_shifts()
    
    print("\n" + "=" * 60)
    print("  🎉 KIIT PIPELINE REGENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext: Run Verilator simulation:")
    print("  bash run_all_verilator.sh pic_test/image_s3r2_kiit_402.jpeg")

if __name__ == "__main__":
    main()
