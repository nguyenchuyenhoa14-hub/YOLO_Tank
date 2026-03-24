#!/usr/bin/env python3
"""
golden_all_layers.py
====================
RTL-accurate golden model for ALL backbone+neck layers.
Chains layers sequentially: L0 output → L1 input → ... 
Handles C2f split/concat logic.

Each layer output is saved as .mem for individual RTL verification.

Usage:
  python3 golden_all_layers.py <image_path>
"""
import numpy as np
import os, sys

# ============================================================================
# Configuration
# ============================================================================
WEIGHTS_DIR = "/mnt/c/github/kiitmita_tank/yolov8n_tank_pcq_finetune_mem"
PE_DIR      = "/mnt/c/github/rtl/yolo_complete/weights_backbone_pe"
OUT_DIR     = "/mnt/c/github/rtl/yolo_complete"
GOLDEN_DIR  = "/mnt/c/github/rtl/yolo_complete/golden_layers"
PARALLEL    = 16
IMG_SIZE    = 320
PAD_VAL     = -128  # 0x80 as signed int8


# ============================================================================
# RTL-accurate functions
# ============================================================================
def to_signed_64(val):
    val = val & 0xFFFFFFFFFFFFFFFF
    if val >= (1 << 63): val -= (1 << 64)
    return val

def to_signed_32(val):
    val = val & 0xFFFFFFFF
    if val >= (1 << 31): val -= (1 << 32)
    return val

def asr_64(val, shift):
    """Arithmetic right shift for signed 64-bit."""
    if val >= 0: return val >> shift
    return -((-val - 1) >> shift) - 1


def read_mem_int8(path):
    vals = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                v = int(l, 16)
                if v >= 128: v -= 256
                vals.append(v)
    return np.array(vals, dtype=np.int8)

def read_mem_hex32(path, signed=False):
    vals = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                v = int(l, 16)
                if signed and v >= 0x80000000: v -= 0x100000000
                vals.append(v)
    return vals

def write_mem(path, data_flat, bits=8):
    with open(path, 'w') as f:
        for v in data_flat:
            f.write(f"{int(v) & 0xFF:02x}\n")

def write_mem_hwc(path, fm):
    """Write HWC feature map as flat .mem file (transposed to CHW for RTL testing)."""
    fm_chw = np.transpose(fm, (2, 0, 1))
    write_mem(path, fm_chw.flatten())


# ============================================================================
# Core: RTL-accurate convolution
# ============================================================================
def rtl_conv_golden(input_fm, weights, bias, mult, shift, zp,
                    IC, OC, K, stride, img_w, img_h, pad_val=PAD_VAL, unsigned_in=False):
    """
    Bit-exact RTL-matching convolution.
    
    Simulates backbone_seq's sequential DDR read: for 3×3 kernels, each IC pass
    consumes PIX + W + 1 bytes. DDR is read continuously, so IC=N starts at
    DDR offset N*(PIX+W+1), reading into the next channel's or zero-padding data.
    
    Args:
        input_fm: numpy int8 array, shape (img_h, img_w, IC) — HWC layout
        weights:  numpy int8 array, shape (OC, IC, K, K)
        bias:     list of int32 (folded biases, length OC)
        mult:     list of uint32 (multipliers, length OC)
        shift:    list of int (shifts, length OC)
        zp:       list of int8 (output zero points, length OC)
        IC, OC, K: channels and kernel size
        stride:   1 or 2
        img_w, img_h: input spatial dimensions
        pad_val:  padding value (signed int8)
        unsigned_in: if True, XOR pixels with 0x80 before MAC (RTL unsigned_in_cfg)
    
    Returns:
        output HWC int8 array: (out_h, out_w, OC) where out_h = img_h//stride
    """
    PIX = img_w * img_h
    
    # Accumulate partial sums across input channels
    psum = np.zeros((PIX, OC), dtype=np.int64)
    
    for ic in range(IC):
        # Simulate line buffer for this input channel
        if K == 3:
            _line_buffer_3x3(input_fm, ic, img_w, img_h, weights, bias, psum, OC, pad_val, unsigned_in=unsigned_in)
        else:  # K == 1
            _direct_1x1(input_fm, ic, img_w, img_h, weights, bias, psum, OC, pad_val, unsigned_in=unsigned_in)
    
    # Quantization (exactly matching RTL 4-stage pipeline)
    output_full = np.zeros((PIX, OC), dtype=np.int8)
    for px in range(PIX):
        for oc in range(OC):
            acc = int(psum[px, oc])
            M = to_signed_32(int(mult[oc]))
            S = int(shift[oc])
            Z = int(zp[oc])
            
            mul_res = to_signed_64(acc * M)
            shifted = asr_64(asr_64(mul_res, 31), S)
            qs = to_signed_32(shifted + Z)
            
            if qs > 127: qc = 127
            elif qs < -128: qc = -128
            else: qc = qs
            
            if qc < Z: qc = Z  # ReLU
            output_full[px, oc] = np.int8(qc)
    
    # Stride selection: RTL uses even-row, even-col from win_idx array
    # Confirmed: win_idx=0 matches RTL's first MAC window (center at (0,0))
    if stride == 2:
        out_h, out_w = img_h // 2, img_w // 2
        output = np.zeros((out_h, out_w, OC), dtype=np.int8)
        for py in range(img_h):
            for px in range(img_w):
                if py % 2 == 0 and px % 2 == 0:
                    output[py // 2, px // 2, :] = output_full[py * img_w + px, :]
    else:
        output = output_full.reshape(img_h, img_w, OC)
    
    return output


def _line_buffer_3x3(input_fm, ic, img_w, img_h, weights, bias, psum, OC, pad_val, unsigned_in=False, ddr_pixels=None):
    """Simulate line buffer for 3×3 kernel, one input channel.
    
    If ddr_pixels is provided, uses those exact bytes (matching RTL DDR stream)
    instead of reading from input_fm. This simulates backbone_seq's sequential
    DDR read where flush pixels come from the next channel's data.
    """
    PIX = img_w * img_h
    K = 3
    
    line_ram0 = np.full(img_w, pad_val, dtype=np.int64)
    line_ram1 = np.full(img_w, pad_val, dtype=np.int64)
    row0_reg = [pad_val] * 3
    row1_reg = [pad_val] * 3
    row2_reg = [pad_val] * 3
    col_cnt = 0
    row_cnt = 0
    win_idx = 0
    
    for pix_idx in range(PIX + img_w + 1):
        if ddr_pixels is not None:
            # Use exact DDR stream bytes (matches RTL backbone_seq behavior)
            pixel_in = ddr_pixels[pix_idx]
        elif pix_idx < PIX:
            h = pix_idx // img_w
            w = pix_idx % img_w
            pixel_in = int(input_fm[h, w, ic])
        else:
            pixel_in = pad_val
        
        ram_idx = col_cnt % img_w
        old_ram0 = int(line_ram0[ram_idx])
        old_ram1 = int(line_ram1[ram_idx])
        
        pad_top = (row_cnt == 1) or (row_cnt == 2 and col_cnt < 2)
        pad_left = (col_cnt == 2)
        pad_right = (col_cnt == 1)
        
        w00 = pad_val if (pad_top or pad_left) else row0_reg[2]
        w01 = pad_val if pad_top else row0_reg[1]
        w02 = pad_val if (pad_top or pad_right) else row0_reg[0]
        w10 = pad_val if pad_left else row1_reg[2]
        w11 = row1_reg[1]
        w12 = pad_val if pad_right else row1_reg[0]
        w20 = pad_val if pad_left else row2_reg[2]
        w21 = row2_reg[1]
        w22 = pad_val if pad_right else row2_reg[0]
        
        wv = True if row_cnt > 1 else (row_cnt == 1 and col_cnt >= 2)
        
        # Update state
        new_r0 = [old_ram1, row0_reg[0], row0_reg[1]]
        new_r1 = [old_ram0, row1_reg[0], row1_reg[1]]
        new_r2 = [pixel_in, row2_reg[0], row2_reg[1]]
        row0_reg, row1_reg, row2_reg = new_r0, new_r1, new_r2
        
        line_ram0[ram_idx] = pixel_in
        line_ram1[ram_idx] = old_ram0
        
        if col_cnt == img_w - 1:
            col_cnt = 0
            if row_cnt < K + 1: row_cnt += 1
        else:
            col_cnt += 1
        
        if wv and win_idx < PIX:
            window = [w00, w01, w02, w10, w11, w12, w20, w21, w22]
            for oc in range(OC):
                result = 0
                for kpos in range(9):
                    ky, kx = kpos // 3, kpos % 3
                    pix_val = window[kpos]
                    if unsigned_in:
                        xored = (int(pix_val) & 0xFF) ^ 0x80
                        pix_val = xored if xored < 128 else xored - 256
                    else:
                        raw = int(pix_val) & 0xFF
                        pix_val = raw if raw < 128 else raw - 256
                    
                    w_val = int(weights[oc, ic, ky, kx])
                    prod = int(pix_val) * w_val
                    result += prod
                    
                    if win_idx == 0 and oc == 5:
                        print(f"    PY ENG MAC [oc={oc} ic={ic} w_j={kpos}] in_val={pix_val} w_val={w_val} prod={prod}")
                        
                if ic == 0:
                    psum[win_idx, oc] = int(bias[oc]) + result
                else:
                    psum[win_idx, oc] += result
            win_idx += 1


def _direct_1x1(input_fm, ic, img_w, img_h, weights, bias, psum, OC, pad_val, unsigned_in=False):
    """Direct 1×1 convolution — no line buffer needed, just multiply-accumulate."""
    PIX = img_w * img_h
    for px in range(PIX):
        h = px // img_w
        w = px % img_w
        pixel_in = int(input_fm[h, w, ic])
        if unsigned_in:
            pixel_in = int(np.int8(np.uint8(pixel_in) ^ 0x80))
        for oc in range(OC):
            result = pixel_in * int(weights[oc, ic, 0, 0])
            if ic == 0:
                psum[px, oc] = int(bias[oc]) + result
            else:
                psum[px, oc] += result


# ============================================================================
# Layer weight loader
# ============================================================================
def load_layer_weights(prefix, IC, OC, K):
    """Load weights, bias, M, S, Z for a layer from PE directory."""
    w_path = os.path.join(WEIGHTS_DIR, f"{prefix}_conv_weight_quantized.mem")
    b_path = os.path.join(PE_DIR, f"{prefix}_conv_bias_quantized.mem")
    m_path = os.path.join(PE_DIR, f"{prefix}_conv_weight_multiplier.mem")
    s_path = os.path.join(PE_DIR, f"{prefix}_conv_weight_shift.mem")
    z_path = os.path.join(PE_DIR, f"{prefix}_conv_weight_zero.mem")
    
    weights = read_mem_int8(w_path).reshape(OC, IC, K, K)
    bias = read_mem_hex32(b_path, signed=True)
    mult = read_mem_hex32(m_path)
    shift = read_mem_hex32(s_path)
    zp_raw = read_mem_hex32(z_path)
    zp = [v if v < 128 else v - 256 for v in zp_raw]
    
    return weights, bias, mult, shift, zp


# ============================================================================
# Layer definitions with data flow
# ============================================================================
# Each entry: (name, prefix, IC, OC, K, stride, input_source, output_tag)
# input_source: which previous output to use
# output_tag: tag for this layer's output
#
# C2f logic:
#   cv1 output (OC channels) → split into [first_half, second_half] (OC//2 each)
#   bottleneck input = first_half
#   cv2 input = concat(second_half, all_bottleneck_outputs)

LAYER_PIPELINE = [
    # === L0-L1: simple conv ===
    ("L0",  "model_0",  3,  16, 3, 2, "input",    "L0_out",   None),
    ("L1",  "model_1", 16,  32, 3, 2, "L0_out",   "L1_out",   None),
    
    # === L2 C2f (n=1) ===
    ("L2_cv1",     "model_2_cv1",     32, 32, 1, 1, "L1_out",      "L2_cv1_out",   None),
    ("L2_m0_cv1",  "model_2_m_0_cv1", 16, 16, 3, 1, "L2_cv1_half2","L2_m0_cv1_out",None),  # y[-1] = chunk1
    ("L2_m0_cv2",  "model_2_m_0_cv2", 16, 16, 3, 1, "L2_m0_cv1_out","L2_m0_cv2_out",None),
    ("L2_cv2",     "model_2_cv2",     48, 32, 1, 1, "L2_concat",   "L2_out",       None),  # 48 = 16+16+16
    
    # === L3 ===
    ("L3",  "model_3", 32,  64, 3, 2, "L2_out",    "L3_out",   None),
    
    # === L4 C2f (n=2) ===
    ("L4_cv1",     "model_4_cv1",     64, 64, 1, 1, "L3_out",       "L4_cv1_out",   None),
    ("L4_m0_cv1",  "model_4_m_0_cv1", 32, 32, 3, 1, "L4_cv1_half2", "L4_m0_cv1_out",None),  # y[-1] = chunk1
    ("L4_m0_cv2",  "model_4_m_0_cv2", 32, 32, 3, 1, "L4_m0_cv1_out","L4_m0_cv2_out",None),
    ("L4_m1_cv1",  "model_4_m_1_cv1", 32, 32, 3, 1, "L4_m0_cv2_out","L4_m1_cv1_out",None),
    ("L4_m1_cv2",  "model_4_m_1_cv2", 32, 32, 3, 1, "L4_m1_cv1_out","L4_m1_cv2_out",None),
    ("L4_cv2",     "model_4_cv2",    128, 64, 1, 1, "L4_concat",    "L4_out",       None),  # 128 = 32*4

    # === L5 ===
    ("L5",  "model_5", 64, 128, 3, 2, "L4_out",    "L5_out",   None),
    
    # === L6 C2f (n=2) ===
    ("L6_cv1",     "model_6_cv1",    128,128, 1, 1, "L5_out",       "L6_cv1_out",   None),
    ("L6_m0_cv1",  "model_6_m_0_cv1", 64, 64, 3, 1, "L6_cv1_half2","L6_m0_cv1_out",None),  # y[-1] = chunk1
    ("L6_m0_cv2",  "model_6_m_0_cv2", 64, 64, 3, 1, "L6_m0_cv1_out","L6_m0_cv2_out",None),
    ("L6_m1_cv1",  "model_6_m_1_cv1", 64, 64, 3, 1, "L6_m0_cv2_out","L6_m1_cv1_out",None),
    ("L6_m1_cv2",  "model_6_m_1_cv2", 64, 64, 3, 1, "L6_m1_cv1_out","L6_m1_cv2_out",None),
    ("L6_cv2",     "model_6_cv2",    256,128, 1, 1, "L6_concat",    "L6_out",       None),  # 256 = 64*4

    # === L7 ===
    ("L7",  "model_7",128, 256, 3, 2, "L6_out",    "L7_out",   None),
    
    # === L8 C2f (n=1) ===
    ("L8_cv1",     "model_8_cv1",    256,256, 1, 1, "L7_out",       "L8_cv1_out",   None),
    ("L8_m0_cv1",  "model_8_m_0_cv1",128,128, 3, 1, "L8_cv1_half2","L8_m0_cv1_out",None),  # y[-1] = chunk1
    ("L8_m0_cv2",  "model_8_m_0_cv2",128,128, 3, 1, "L8_m0_cv1_out","L8_m0_cv2_out",None),
    ("L8_cv2",     "model_8_cv2",    384,256, 1, 1, "L8_concat",    "L8_out",       None),

    # === L9: SPPF ===
    ("L9_cv1",     "model_9_cv1",    256,128, 1, 1, "L8_out",       "L9_cv1_out",   None),
    # MaxPool chain + concat handled in pipeline runner (SPPF_concat)
    ("L9_cv2",     "model_9_cv2",    512,256, 1, 1, "SPPF_concat",  "L9_out",       None),

    # === NECK (FPN + PAN) ===
    # L10: Upsample L9_out 10×10→20×20 (256ch) — handled outside conv pipeline
    # L11: Concat(upsample, L6_out) → 384ch — handled outside conv pipeline

    # === L12 C2f (n=1), 20×20, 384→128 ===
    ("L12_cv1",    "model_12_cv1",    384,128, 1, 1, "L12_concat",   "L12_cv1_out",  None),
    ("L12_m0_cv1", "model_12_m_0_cv1", 64, 64, 3, 1, "L12_cv1_half2","L12_m0_cv1_out",None),
    ("L12_m0_cv2", "model_12_m_0_cv2", 64, 64, 3, 1, "L12_m0_cv1_out","L12_m0_cv2_out",None),
    ("L12_cv2",    "model_12_cv2",    192,128, 1, 1, "L12_c2f_concat","L12_out",      None),  # 192 = 64+64+64

    # L13: Upsample L12_out 20×20→40×40 (128ch) — handled outside
    # L14: Concat(upsample, L4_out) → 192ch — handled outside

    # === L15 C2f (n=1), 40×40, 192→64 → P3 ===
    ("L15_cv1",    "model_15_cv1",    192, 64, 1, 1, "L15_concat",   "L15_cv1_out",  None),
    ("L15_m0_cv1", "model_15_m_0_cv1", 32, 32, 3, 1, "L15_cv1_half2","L15_m0_cv1_out",None),
    ("L15_m0_cv2", "model_15_m_0_cv2", 32, 32, 3, 1, "L15_m0_cv1_out","L15_m0_cv2_out",None),
    ("L15_cv2",    "model_15_cv2",     96, 64, 1, 1, "L15_c2f_concat","L15_out",      None),  # 96 = 32+32+32 → P3

    # === L16: Conv 64→64, 3×3, s2, 40→20 ===
    ("L16",  "model_16", 64,  64, 3, 2, "L15_out",   "L16_out",  None),

    # L17: Concat(L16, L12_out) → 192ch — handled outside

    # === L18 C2f (n=1), 20×20, 192→128 → P4 ===
    ("L18_cv1",    "model_18_cv1",    192,128, 1, 1, "L18_concat",   "L18_cv1_out",  None),
    ("L18_m0_cv1", "model_18_m_0_cv1", 64, 64, 3, 1, "L18_cv1_half2","L18_m0_cv1_out",None),
    ("L18_m0_cv2", "model_18_m_0_cv2", 64, 64, 3, 1, "L18_m0_cv1_out","L18_m0_cv2_out",None),
    ("L18_cv2",    "model_18_cv2",    192,128, 1, 1, "L18_c2f_concat","L18_out",      None),  # 192 = 64+64+64 → P4

    # === L19: Conv 128→128, 3×3, s2, 20→10 ===
    ("L19",  "model_19",128, 128, 3, 2, "L18_out",   "L19_out",  None),

    # L20: Concat(L19, L9_out/P5) → 384ch — handled outside

    # === L21 C2f (n=1), 10×10, 384→256 → P5 ===
    ("L21_cv1",    "model_21_cv1",    384,256, 1, 1, "L21_concat",   "L21_cv1_out",  None),
    ("L21_m0_cv1", "model_21_m_0_cv1",128,128, 3, 1, "L21_cv1_half2","L21_m0_cv1_out",None),
    ("L21_m0_cv2", "model_21_m_0_cv2",128,128, 3, 1, "L21_m0_cv1_out","L21_m0_cv2_out",None),
    ("L21_cv2",    "model_21_cv2",    384,256, 1, 1, "L21_c2f_concat","L21_out",      None),  # 384 = 128+128+128 → P5
]

# Neck upsample/concat operations (not conv layers, handled in pipeline runner)


# ============================================================================
# C2f helper: split and concat
# ============================================================================
def c2f_split(fm, half_ch):
    """Split feature map channels into two halves."""
    h, w, c = fm.shape
    assert c == half_ch * 2, f"Expected {half_ch*2} channels, got {c}"
    half1 = fm[:, :, :half_ch].copy()
    half2 = fm[:, :, half_ch:].copy()
    return half1, half2

def c2f_concat(parts):
    """Concatenate feature maps along channel axis."""
    return np.concatenate(parts, axis=2)

def upsample_2x(fm):
    """Nearest-neighbor 2× upsample (HWC format)."""
    h, w, c = fm.shape
    out = np.zeros((h * 2, w * 2, c), dtype=fm.dtype)
    for y in range(h):
        for x in range(w):
            out[2*y,   2*x,   :] = fm[y, x, :]
            out[2*y,   2*x+1, :] = fm[y, x, :]
            out[2*y+1, 2*x,   :] = fm[y, x, :]
            out[2*y+1, 2*x+1, :] = fm[y, x, :]
    return out

def maxpool_5x5_signed(fm):
    """MaxPool 5×5, stride=1, padding=2 (signed INT8 HWC format)."""
    h, w, c = fm.shape
    out = np.zeros_like(fm)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                max_val = np.int8(-128)
                for ky in range(-2, 3):
                    for kx in range(-2, 3):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < h and 0 <= nx < w:
                            val = np.int8(fm[ny, nx, ch])
                            if val > max_val:
                                max_val = val
                out[y, x, ch] = max_val
    return out


# ============================================================================
# Main pipeline
# ============================================================================
def run_pipeline(img_path):
    import cv2
    import onnx
    from onnx import numpy_helper
    
    ONNX_PATH = "/mnt/c/github/kiitmita_tank/pcq_finetune_200/weights/best_int8_pcq.onnx"
    
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    
    # Quantize input image
    print("=" * 60)
    print("  Quantizing input image")
    print("=" * 60)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    model = onnx.load(ONNX_PATH)
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    input_scale, input_zp = 1.0/255.0, -128
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            if node.input[1] in init_map:
                input_scale = float(init_map[node.input[1]].flatten()[0])
            if len(node.input) >= 3 and node.input[2] in init_map:
                input_zp = int(init_map[node.input[2]].flatten()[0])
            break
    
    img_float = img_rgb.astype(np.float32) / 255.0
    img_q = np.clip(np.round(img_float / input_scale) + input_zp, -128, 127).astype(np.int8)
    print(f"  Input: {img_q.shape}, scale={input_scale:.6f}, ZP={input_zp}")
    
    # Feature map storage
    fmaps = {"input": img_q}
    
    # Process each layer
    print("\n" + "=" * 60)
    print("  Processing backbone layers")
    print("=" * 60)
    
    for i, (name, prefix, ic, oc, k, stride, in_src, out_tag, _) in enumerate(LAYER_PIPELINE):
        # Handle C2f split/concat sources
        if in_src.endswith("_half1") or in_src.endswith("_half2"):
            # Split the cv1 output if not already split
            base = in_src.replace("_half1", "_out").replace("_half2", "_out")
            if base in fmaps and in_src not in fmaps:
                half_ch = fmaps[base].shape[2] // 2
                half1, half2 = c2f_split(fmaps[base], half_ch)
                half1_key = base.replace("_out", "_half1")
                half2_key = base.replace("_out", "_half2")
                fmaps[half1_key] = half1
                fmaps[half2_key] = half2
            in_fm = fmaps[in_src]
        elif in_src.endswith("_c2f_concat"):
            # C2f internal concat: half1 + half2 + bottleneck outputs
            c2f_prefix = in_src.replace("_c2f_concat", "")
            half1_key = f"{c2f_prefix}_cv1_half1"
            half2_key = f"{c2f_prefix}_cv1_half2"
            btn_outputs = []
            for j in range(i):
                jname = LAYER_PIPELINE[j][0]
                jtag = LAYER_PIPELINE[j][7]
                if jname.startswith(c2f_prefix + "_m") and jname.endswith("_cv2"):
                    if jtag in fmaps:
                        btn_outputs.append(fmaps[jtag])
            parts = [fmaps[half1_key], fmaps[half2_key]] + btn_outputs
            in_fm = c2f_concat(parts)
            fmaps[in_src] = in_fm
        elif in_src.endswith("_concat"):
            # Neck inter-layer concat (with upsample where needed)
            # SPPF_concat = Concat(cv1, mp1, mp2, mp3) = 512ch
            # L12_concat = Concat(upsample(L9_out), L6_out)
            # L15_concat = Concat(upsample(L12_out), L4_out)
            # L18_concat = Concat(L16_out, L12_out)
            # L21_concat = Concat(L19_out, L9_out)
            if in_src == "SPPF_concat":
                cv1_out = fmaps["L9_cv1_out"]
                mp1 = maxpool_5x5_signed(cv1_out)
                mp2 = maxpool_5x5_signed(mp1)
                mp3 = maxpool_5x5_signed(mp2)
                fmaps["SPPF_mp1"] = mp1
                fmaps["SPPF_mp2"] = mp2
                fmaps["SPPF_mp3"] = mp3
                in_fm = c2f_concat([cv1_out, mp1, mp2, mp3])  # 128*4=512
                print(f"  SPPF MaxPool: mp1 range=[{mp1.min()},{mp1.max()}], mp2 range=[{mp2.min()},{mp2.max()}], mp3 range=[{mp3.min()},{mp3.max()}]")
            elif in_src == "L12_concat":
                ups = upsample_2x(fmaps["L9_out"])
                fmaps["L10_ups"] = ups
                in_fm = c2f_concat([ups, fmaps["L6_out"]])
            elif in_src == "L15_concat":
                ups = upsample_2x(fmaps["L12_out"])
                fmaps["L13_ups"] = ups
                in_fm = c2f_concat([ups, fmaps["L4_out"]])
            elif in_src == "L18_concat":
                in_fm = c2f_concat([fmaps["L16_out"], fmaps["L12_out"]])
            elif in_src == "L21_concat":
                in_fm = c2f_concat([fmaps["L19_out"], fmaps["L9_out"]])
            else:
                # Backbone C2f concat (legacy)
                c2f_prefix = in_src.replace("_concat", "")
                half1_key = f"{c2f_prefix}_cv1_half1"
                half2_key = f"{c2f_prefix}_cv1_half2"
                btn_outputs = []
                for j in range(i):
                    jname = LAYER_PIPELINE[j][0]
                    jtag = LAYER_PIPELINE[j][7]
                    if jname.startswith(c2f_prefix + "_m") and jname.endswith("_cv2"):
                        if jtag in fmaps:
                            btn_outputs.append(fmaps[jtag])
                parts = [fmaps[half1_key], fmaps[half2_key]] + btn_outputs
                in_fm = c2f_concat(parts)
            fmaps[in_src] = in_fm
        else:
            in_fm = fmaps[in_src]
        
        h, w, c = in_fm.shape
        assert c == ic, f"{name}: expected IC={ic}, got {c}"
        
        # Load weights
        try:
            weights, bias, mult, shift, zp = load_layer_weights(prefix, ic, oc, k)
        except Exception as e:
            print(f"  ⚠️  {name}: weight load failed: {e}")
            continue
        
        # Padding value: L0 uses Z_in=-128 (input image), all others Z_in=0 (ReLU output)
        layer_pad_val = PAD_VAL if name == "L0" else 0
        
        # Run RTL-accurate conv
        # NOTE: RTL hardcodes cfg_unsigned_in = 0 for ALL layers
        layer_unsigned_in = False
        print(f"  [{i+1:2d}/{len(LAYER_PIPELINE)}] {name}: {ic}→{oc}, {k}×{k}, s{stride}, {w}×{h} pad={layer_pad_val} ", end="", flush=True)
        out_fm = rtl_conv_golden(in_fm, weights, bias, mult, shift, zp,
                                  ic, oc, k, stride, w, h, pad_val=layer_pad_val, unsigned_in=layer_unsigned_in)
        
        oh, ow, oc_out = out_fm.shape
        print(f"→ {ow}×{oh}×{oc_out}")
        
        # Store output
        fmaps[out_tag] = out_fm
        
        # Save to .mem file
        mem_path = os.path.join(GOLDEN_DIR, f"{name}_golden.mem")
        write_mem_hwc(mem_path, out_fm)
        
        # Also save input .mem for testbench use
        in_mem_path = os.path.join(GOLDEN_DIR, f"{name}_input.mem")
        write_mem_hwc(in_mem_path, in_fm)
    
    print("\n" + "=" * 60)
    print("  🎉 ALL BACKBONE LAYERS COMPLETE!")
    print("=" * 60)
    
    # Summary
    for name, prefix, ic, oc, k, stride, in_src, out_tag, _ in LAYER_PIPELINE:
        if out_tag in fmaps:
            fm = fmaps[out_tag]
            print(f"  {name:15s}: {fm.shape[1]:3d}×{fm.shape[0]:3d}×{fm.shape[2]:3d}  range=[{fm.min():4d},{fm.max():4d}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 golden_all_layers.py <image_path>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
