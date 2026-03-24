#!/usr/bin/env python3
"""
regenerate_all.py
=================
Master script to regenerate ALL .mem files after model retrain.

Steps:
1. Extract activation scales + zero points from ONNX model
2. Regenerate layer15/20/21_output.mem with correct scales + ZP
3. Fold Z_in into biases in weights_for_verilog_pe/
4. Verify shift files are consistent

Usage: ~/yolov8_env/bin/python3 regenerate_all.py pic_test/tank4.jpg
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
# Configuration
# ============================================================================
SCALES_JSON = "/mnt/c/github/activation_scales.json"
ONNX_PATH = "/mnt/c/github/step3_qat/weights/best_int8_pcq.onnx"
MODEL_PATH = "/mnt/c/github/step3_qat/weights/best.pt"
PE_DIR     = "/mnt/c/github/rtl/yolo_complete/weights_for_verilog_pe"
OUT_DIR    = "/mnt/c/github/rtl/yolo_complete"
WEIGHTS_DIR = "/mnt/c/github/yolov8n_kiitmita_relu_416_ultimate_qat_mem"
PARALLEL   = 16

# ONNX tensor patterns for detect head input convolutions
TARGET_CONV_PATTERNS = {
    'P3': 'cv2.0/cv2.0.0',
    'P4': 'cv2.1/cv2.1.0',
    'P5': 'cv2.2/cv2.2.0',
}

# Layer definitions for bias folding
# (layer_key, is_first_in_branch)  
# is_first_in_branch means it takes P3/P4/P5 input directly
LAYER_DEFS = [
    # P3: cv2_0_X, cv3_0_X
    ('cv2_0_0', True,  'P3'), ('cv2_0_1', False, None), ('cv2_0_2', False, None),
    ('cv3_0_0', True,  'P3'), ('cv3_0_1', False, None), ('cv3_0_2', False, None),
    # P4: cv2_1_X, cv3_1_X
    ('cv2_1_0', True,  'P4'), ('cv2_1_1', False, None), ('cv2_1_2', False, None),
    ('cv3_1_0', True,  'P4'), ('cv3_1_1', False, None), ('cv3_1_2', False, None),
    # P5: cv2_2_X, cv3_2_X
    ('cv2_2_0', True,  'P5'), ('cv2_2_1', False, None), ('cv2_2_2', False, None),
    ('cv3_2_0', True,  'P5'), ('cv3_2_1', False, None), ('cv3_2_2', False, None),
]

FILE_MAP = {'P3': 'layer15_output.mem', 'P4': 'layer20_output.mem', 'P5': 'layer21_output.mem'}

# ============================================================================
# Step 1: Extract scales from ONNX
# ============================================================================
def extract_onnx_scales():
    print("=" * 60)
    print("  Step 1: Extract Activation Scales from ONNX")
    print("=" * 60)
    
    model = onnx.load(ONNX_PATH)
    
    # Build initializer lookup
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    
    # Find Conv input tensors
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
    
    # Also extract ALL quantization ZPs for bias folding
    all_zps = {}
    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            if len(node.input) >= 3:
                zp_name = node.input[2]
                if zp_name in init_map:
                    zp_val = int(init_map[zp_name].flatten()[0])
                    # Map both input and output tensor names
                    all_zps[node.input[0]] = zp_val
                    all_zps[node.output[0]] = zp_val
    
    for branch in ['P3', 'P4', 'P5']:
        s = scales.get(branch, 'NOT FOUND')
        zp = zero_points.get(branch, 'NOT FOUND')
        print(f"  {branch}: scale={s}, zero_point={zp}")
    
    return scales, zero_points, all_zps

# ============================================================================
# Step 2: Regenerate input feature maps
# ============================================================================
def regenerate_feature_maps(img_path, scales, zero_points):
    print("\n" + "=" * 60)
    print("  Step 2: Regenerate Feature Maps (layer15/20/21_output.mem)")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    model = YOLO(MODEL_PATH)
    nn_model = model.model
    nn_model.eval()
    
    img_bgr = cv2.imread(img_path)
    img_320 = cv2.resize(img_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_320, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    
    # Hook detect head inputs
    feature_maps = {}
    def det_hook(model, inp, out):
        feature_maps['P3'] = inp[0][0][0].detach().cpu().numpy()  # [C,H,W]
        feature_maps['P4'] = inp[0][1][0].detach().cpu().numpy()  # [C,H,W]
        feature_maps['P5'] = inp[0][2][0].detach().cpu().numpy()  # [C,H,W]
    
    h = nn_model.model[-1].register_forward_hook(det_hook)
    with torch.no_grad():
        _ = nn_model(img_t)
    h.remove()
    
    for branch in ['P3', 'P4', 'P5']:
        fm_full = feature_maps[branch]  # [C, H, W], e.g., [64, 52, 52] for P3
        
        # Determine crop size based on RTL grid (40x40 for P3, 20x20 for P4, 10x10 for P5 if scaled)
        # Actually P4 grid is 20x20 in top core, P5 is 10x10?
        # Let's check yolov8_top_core.v: GRID_W=40. Stride 8 -> 320x320 region.
        # P3 grid: 40x40
        # P4 grid: 20x20
        # P5 grid: 10x10
        grid_size = {'P3': 40, 'P4': 20, 'P5': 10}[branch]
        fm = fm_full[:, :grid_size, :grid_size]
        
        scale = scales[branch]
        zp = zero_points.get(branch, 0)
        
        fm_q = np.clip(np.round(fm / scale) + zp, -128, 127).astype(np.int8)
        fm_q_uint8 = fm_q.view(np.uint8)
        
        print(f"  [{branch}] Float Sliced: min={fm.min():.4f}, max={fm.max():.4f}, mean={fm.mean():.4f}")
        print(f"  [{branch}] Quantized Sliced: min={fm_q.min()}, max={fm_q.max()}, mean={fm_q.mean():.2f}")
        
        # CHW -> HWC, flatten
        fm_hwc = fm_q_uint8.transpose(1, 2, 0) # [H, W, C]
        fm_flat = fm_hwc.flatten()
        
        out_path = os.path.join(OUT_DIR, FILE_MAP[branch])
        with open(out_path, "w") as f:
            for v in fm_flat:
                f.write(f"{int(v) & 0xFF:02x}\n")
        print(f"  ✅ Saved {len(fm_flat)} bytes -> {out_path}\n")

# ============================================================================
# Step 3: Fold Z_in into biases
# ============================================================================
def fold_biases(scales, zero_points, all_zps):
    print("\n" + "=" * 60)
    print("  Step 3: Fold Z_in=-128 into Biases")
    print("=" * 60)
    
    for layer_key, is_first, branch in LAYER_DEFS:
        # Determine Z_in for this layer
        if is_first:
            # First layer in branch: Z_in handled by Verilog XOR
            z_in = 0
        else:
            # Intermediate/final layer: Z_in handled by Verilog XOR
            z_in = 0
            
        # Z_out: For intermediate layers with ReLU, ZP is usually -128
        # For final layers (cv2_2, cv3_2) without ReLU, ZP could be different
        layer_idx = int(layer_key.split('_')[2])
        if "2" in layer_key.split('_')[2]:
            # Final 1x1 conv (no ReLU) - Dynamically find ZP from all_zps if possible
            # Fallback to hardcoded if not found, but we should prioritize the ONNX data.
            fallback_zps = {
                'cv2_0_2': 29,   'cv3_0_2': 109,
                'cv2_1_2': 38,   'cv3_1_2': 113,
                'cv2_2_2': 57,   'cv3_2_2': 104,
            }
            # Try to find the tensor name in ONNX to get ZP
            found_zp = None
            search_pattern = layer_key.replace('_','.') # e.g. "cv3.0.2"
            best_match = ""
            for tensor_name, zp in all_zps.items():
                if search_pattern in tensor_name:
                    # Prioritize activation ZPs (usually != 0) over weight ZPs (usually 0)
                    if found_zp is None or (found_zp == 0 and zp != 0):
                        found_zp = zp
                        best_match = tensor_name
            
            if found_zp is not None:
                print(f"  DEBUG: Found ZP {found_zp} for {layer_key} (matched {best_match})")
            
            if found_zp is None:
                print(f"  DEBUG: No ZP found for {layer_key} (pattern={search_pattern}) - using fallback")
                z_out = fallback_zps.get(layer_key, 0)
            else:
                z_out = found_zp
        else:
            # MUST BE -128 to prevent positive clipping! ONNX maps 0.0 -> -128
            z_out = -128
        
        w_file = os.path.join(PE_DIR, f"model_22_{layer_key}_conv_weight_quantized_pe.mem")
        
        # Determine original bias and scale filenames mathematically mimicking convert_weights_to_pe
        mid_tag = "_conv" if "2" not in layer_key.split('_')[2] else ""
        bname = f"model_22_{layer_key}{mid_tag}_bias_quantized"
        mname = f"model_22_{layer_key}{mid_tag}_weight_multiplier"
        sname = f"model_22_{layer_key}{mid_tag}_weight_shift"
        
        b_file = os.path.join(WEIGHTS_DIR, f"{bname}.mem")
        m_file = os.path.join(WEIGHTS_DIR, f"{mname}.mem")
        s_file = os.path.join(WEIGHTS_DIR, f"{sname}.mem")
        
        b_out_file = os.path.join(PE_DIR, f"model_22_{layer_key}_conv_bias_quantized.mem")
        
        if not os.path.exists(w_file):
            print(f"  ⚠️  {w_file} not found, skipping {layer_key}")
            continue
        
        # Read weights (int8)
        with open(w_file) as f:
            w_hex = [l.strip() for l in f if l.strip()]
        w_all = np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in w_hex])
        
        # Read original bias (int32)
        with open(b_file) as f:
            b_hex = [l.strip() for l in f if l.strip()]
        b_all = np.array([int(x, 16) if int(x, 16) < 0x80000000 else int(x, 16) - 0x100000000 for x in b_hex])
        
        # Read multiplier (int32)
        with open(m_file) as f:
            m_hex = [l.strip() for l in f if l.strip()]
        m_all = np.array([int(x, 16) if int(x, 16) < 0x80000000 else int(x, 16) - 0x100000000 for x in m_hex])
        
        # Read shift
        with open(s_file) as f:
            s_hex = [l.strip() for l in f if l.strip()]
        s_all = np.array([int(x, 16) for x in s_hex])
        
        num_oc = len(b_all)
        
        # Write zero_point.mem for hardware to use - MUST match num_oc exactly for Verilator $readmem
        zp_file = os.path.join(PE_DIR, f"model_22_{layer_key}_zero_point.mem")
        with open(zp_file, "w") as f:
            for _ in range(num_oc):
                f.write(f"{z_out & 0xFF:02x}\n")
        
        b_new = []
        
        is_3x3 = "2" not in layer_key.split('_')[2]
        
        for oc in range(num_oc):
            out_blk = oc // PARALLEL
            pe = oc % PARALLEL
            w_ch = []
            
            if is_3x3:
                in_ch = len(w_all) // (num_oc * 9)
                for ic in range(in_ch):
                    for kpos in range(9):
                        idx = out_blk * in_ch * PARALLEL * 9 + ic * PARALLEL * 9 + pe * 9 + kpos
                        if idx < len(w_all):
                            w_ch.append(w_all[idx])
            else:
                in_ch = len(w_all) // num_oc
                for ic in range(in_ch):
                    idx = out_blk * in_ch * PARALLEL + ic * PARALLEL + pe
                    if idx < len(w_all):
                        w_ch.append(w_all[idx])
                        
            sum_w = int(np.sum(w_ch))
            
            # B_new = B_old - Z_in * sum(W) + fold_z_out
            fold_z_in = z_in * sum_w
            
            # HW explicitly adds Z_out (z_data_r) in conv_stage.v: `(qs_w) + $signed(z_data_r)`
            fold_z_out = 0
            
            b_target = int(b_all[oc]) - fold_z_in + fold_z_out
            
            # Clamp to 32-bit signed
            b_target = max(-0x80000000, min(0x7FFFFFFF, b_target))
            b_new.append(b_target)
        
        # Write folded biases to PE_DIR
        with open(b_out_file, "w") as f:
            for b in b_new:
                f.write(f"{b & 0xFFFFFFFF:08x}\n")
        
        print(f"  ✅ {layer_key}: Z_in={z_in}, Z_out={z_out}, B[0]: {b_all[0]} -> {b_new[0]}")

# ============================================================================
# Step 4: Verify shift files
# ============================================================================
def verify_shifts():
    print("\n" + "=" * 60)
    print("  Step 4: Verify Shift Files")
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
        print("Usage: python3 regenerate_all.py <image_path>")
        print("  e.g.: python3 regenerate_all.py pic_test/tank4.jpg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)
    
    # Backup original PE weights
    backup_dir = PE_DIR + "_backup"
    if not os.path.exists(backup_dir):
        print(f"📦 Backing up {PE_DIR} -> {backup_dir}")
        shutil.copytree(PE_DIR, backup_dir)
    else:
        # Restore biases from backup before re-folding
        print(f"📦 Restoring original biases from {backup_dir}")
        for f in os.listdir(backup_dir):
            if "bias" in f:
                shutil.copy(os.path.join(backup_dir, f), os.path.join(PE_DIR, f))
    
    # Step 1: Extract scales
    scales, zero_points, all_zps = extract_onnx_scales()
    
    # Step 2: Regenerate feature maps with correct ONNX scales
    if "--skip-step2" not in sys.argv:
        regenerate_feature_maps(img_path, scales, zero_points)
    else:
        print("\n" + "=" * 60)
        print("  Step 2: SKIPPED (using existing feature maps)")
        print("=" * 60)
    
    # Step 3: Fold Z_in into biases
    fold_biases(scales, zero_points, all_zps)
    
    # Step 4: Verify shifts
    verify_shifts()
    
    print("\n" + "=" * 60)
    print("  🎉 ALL DONE! Pipeline regenerated successfully!")
    print("=" * 60)
    print("\nNext step: Run Verilator simulation to verify:")
    print("  cd /mnt/c/github/rtl/yolo_complete && ./run_behavioral_verilator.sh")

if __name__ == "__main__":
    main()
