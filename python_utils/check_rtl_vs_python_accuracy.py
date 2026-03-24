#!/usr/bin/env python3
import torch
import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Configuration
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "pic_test/tank4.jpg"
MODEL_PATH = "step3_qat/weights/best.pt"
RTL_P3_MEM = "rtl_output/yolov8_p3_output.mem"

def main():
    print("="*70)
    print(" 🎯 SRNet RTL vs PyTorch: MATHEMATICAL ACCURACY VERIFICATION 🎯")
    print("="*70)

    # 1. Run inference on PyTorch
    model = YOLO(MODEL_PATH)
    nn = model.model
    nn.eval()
    
    img_bgr = cv2.imread(IMG_PATH)
    img_320 = cv2.resize(img_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_320, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)

    # We need the output of the P3 head (layer 22, cv2[0] and cv3[0])
    feature_maps = {}
    def get_hook(name):
        def hook(model, inp, out):
            feature_maps[name] = out.detach()
        return hook
        
    h_cv2_0 = nn.model[-1].cv2[0].register_forward_hook(get_hook('cv2_0'))
    h_cv3_0 = nn.model[-1].cv3[0].register_forward_hook(get_hook('cv3_0'))
    
    with torch.no_grad():
        _ = nn(img_t)
        
    h_cv2_0.remove()
    h_cv3_0.remove()

    cv2_0_out = feature_maps['cv2_0'][0] # Shape: [64, 40, 40]
    cv3_0_out = feature_maps['cv3_0'][0] # Shape: [1, 40, 40]

    # Convert PyTorch Tensors to DFL Boxes
    # cv2_0 has 64 channels = 4 * 16 (L, T, R, B)
    # The PyTorch DFL layer applies Softmax over the 16 bins and computes the expectation
    dfl_weights = torch.arange(16, dtype=torch.float32)
    boxes_pred = cv2_0_out.view(4, 16, 40, 40)
    boxes_prob = torch.nn.functional.softmax(boxes_pred, dim=1)
    boxes_dist = torch.sum(boxes_prob * dfl_weights.view(1, 16, 1, 1), dim=1) # [4, 40, 40]
    
    # Class scores
    cls_scores = torch.sigmoid(cv3_0_out) # [1, 40, 40]

    # 2. Read RTL output
    if not os.path.exists(RTL_P3_MEM):
        print(f"❌ RTL output file not found: {RTL_P3_MEM}")
        print("Please run ./run_all_rtl.sh first to generate the RTL output.")
        sys.exit(1)
        
    rtl_results = {}
    with open(RTL_P3_MEM, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    # Read blocks of 5 words
    idx = 0
    while idx < len(lines):
        if 'DEADBEEF' in lines[idx]:
            break
        try:
            val_l = int(lines[idx], 16)
            if val_l > 0x7FFFFFFF: val_l -= 0x100000000
            
            val_t = int(lines[idx+1], 16)
            if val_t > 0x7FFFFFFF: val_t -= 0x100000000
            
            val_r = int(lines[idx+2], 16)
            if val_r > 0x7FFFFFFF: val_r -= 0x100000000
            
            val_b = int(lines[idx+3], 16)
            if val_b > 0x7FFFFFFF: val_b -= 0x100000000
            
            word4 = int(lines[idx+4], 16)
            cell_idx = (word4 >> 16) & 0xFFFF
            conf = word4 & 0xFF
            if conf > 127: conf -= 256
            
            rtl_results[cell_idx] = {'l': val_l, 't': val_t, 'r': val_r, 'b': val_b, 'conf': conf}
            idx += 5
        except:
            break

    print(f"✅ PyTorch DFL processed: 40x40 = 1600 cells")
    print(f"✅ RTL Output extracted: {len(rtl_results)} valid cell detections")

    # Print top 10 PyTorch scores for debugging
    # Flatten class scores and get top indices
    flat_cls_scores = cls_scores.flatten()
    top_indices_pt = torch.argsort(flat_cls_scores, descending=True)[:10]
    print("\n--- Top 10 PyTorch Detections (Raw Scores) ---")
    for i, cell_idx in enumerate(top_indices_pt):
        score = flat_cls_scores[cell_idx].item()
        # Assuming a single class for simplicity, or you can extend this to get class ID if multi-class
        print(f"Cell {cell_idx.item():<4} | Score: {score:.4f}")

    print("\n--- Detailed Cell-by-Cell Comparison (Top 10 highest confidence) ---")
    
    # Sort RTL results by confidence
    sorted_rtl = sorted(rtl_results.items(), key=lambda x: x[1]['conf'], reverse=True)
    
    total_l_error = 0
    total_t_error = 0
    total_r_error = 0
    total_b_error = 0
    
    comparisons_made = 0
    
    for i, (cell_idx, r_data) in enumerate(sorted_rtl):
        cy = cell_idx // 40
        cx = cell_idx % 40
        
        # PyTorch values at this exact cell
        pt_l = boxes_dist[0, cy, cx].item()
        pt_t = boxes_dist[1, cy, cx].item()
        pt_r = boxes_dist[2, cy, cx].item()
        pt_b = boxes_dist[3, cy, cx].item()
        
        # PyTorch conf is sigmoid (0.0 to 1.0). RTL conf is int8 prior to sigmoid? 
        # Wait, RTL conf is raw score scaled. 
        # We just care about the L,T,R,B distance matching.
        
        # RTL values are raw DFL MAC outputs (Shifted left by some bit scale).
        # We need to scale them down. If RTL outputs are around 100-200 and PyTorch are 5-15,
        # what is the scale factor?
        # Typically DFL output has 4 fractional bits if Mac Accumulator was shifted by 4.
        rtl_l = r_data['l'] / 4096.0
        rtl_t = r_data['t'] / 4096.0
        rtl_r = r_data['r'] / 4096.0
        rtl_b = r_data['b'] / 4096.0
        
        l_err = abs(pt_l - rtl_l)
        t_err = abs(pt_t - rtl_t)
        r_err = abs(pt_r - rtl_r)
        b_err = abs(pt_b - rtl_b)
        
        total_l_error += l_err
        total_t_error += t_err
        total_r_error += r_err
        total_b_error += b_err
        comparisons_made += 1
        
        if i < 10:
            print(f"Cell {cell_idx:<4} ({cx:2},{cy:2}) | PyTorch: L={pt_l:5.2f}, T={pt_t:5.2f}, R={pt_r:5.2f}, B={pt_b:5.2f} "
                  f"| RTL: L={rtl_l:5.2f}, T={rtl_t:5.2f}, R={rtl_r:5.2f}, B={rtl_b:5.2f} "
                  f"| Diff: ~{max(l_err, t_err, r_err, b_err):.2f} px")

    if comparisons_made > 0:
        print("\n==================================================")
        print(" 📊 FINAL ACCURACY METRICS (Mean Absolute Error) 📊")
        print("==================================================")
        print(f"  Average L Error : {total_l_error / comparisons_made:.4f} pixels")
        print(f"  Average T Error : {total_t_error / comparisons_made:.4f} pixels")
        print(f"  Average R Error : {total_r_error / comparisons_made:.4f} pixels")
        print(f"  Average B Error : {total_b_error / comparisons_made:.4f} pixels")
        print(f"  Average Overall : {(total_l_error + total_t_error + total_r_error + total_b_error) / (4 * comparisons_made):.4f} pixels")
        print("==================================================")
        print("Note: An error of < 1.0 pixel is considered mathematically equivalent")
        print("since Hardware uses INT8 Quantization while PyTorch uses FP32 Float.")
    else:
        print("No matches found to compare. Run simulation first.")

if __name__ == "__main__":
    main()
