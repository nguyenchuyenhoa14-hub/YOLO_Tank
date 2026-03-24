#!/bin/bash
# ============================================================================
# YOLOv8 Full RTL Pipeline (SRNet) - VERILATOR SPEED RUN (WSL/LINUX)
# Sequential Pipeline: P3 detect head + IoU NMS (P4/P5 omitted)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$1" ]; then
    IMG_PATH="pic_test/image_s3r2_kiit_402.jpeg"
    echo "No input image given, defaulting to $IMG_PATH"
else
    IMG_PATH="$1"
fi
basename=$(basename "$IMG_PATH" | sed 's/\.[^.]*$//')

echo "╔══════════════════════════════════════════════╗"
echo "║  YOLOv8 SEQUENTIAL PIPELINE (P3 only)      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 0. Run KIIT regeneration script (feature maps + PE weights + bias folding)
echo "▶ [0/5] Running KIIT pipeline: regenerate_all_kiit.py..."
source myenv/bin/activate
python3 regenerate_all_kiit.py "$IMG_PATH"

cd "$SCRIPT_DIR/rtl/yolo_complete"

OBJ_BASE="$HOME/yolo_obj_dir"
OUT_DIR="$SCRIPT_DIR/rtl_output"
mkdir -p $OUT_DIR
VFLAGS_BASE="--binary --timing -Wno-fatal -Wno-TIMESCALEMOD -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-BLKLOOPINIT -Wno-INITIALDLY -Wno-UNOPTFLAT"
VCFLAGS="-CFLAGS -std=c++20 -CFLAGS -O3"

# 1. Run P3 Sequential Simulation
echo "▶ [1/5] Compiling & Running P3 Sequential Detection Head (40x40)..."
OBJ_DIR="$OBJ_BASE/p3"
rm -rf "$OBJ_DIR" && mkdir -p "$OBJ_DIR"
verilator --trace --Mdir $OBJ_DIR $VFLAGS_BASE $VCFLAGS --top-module tb_p3_seq -o p3_seq_test \
    tb/tb_p3_seq.v \
    cnn/head/detect_head_seq.v \
    cnn/head/conv_stage.v \
    cnn/head/cnn_engine_dynamic.v \
    cnn/head/line_buffer.v \
    cnn/head/dfl_accelerator.v \
    cnn/head/seq_divider.v
$OBJ_DIR/p3_seq_test
sync
cp -f yolov8_p3_output.mem "$OUT_DIR/" 2>/dev/null && echo "  ✅ P3 Output saved: $OUT_DIR/yolov8_p3_output.mem" || echo "  ⚠️  P3 output file not found"
echo ""

# P4 SKIPPED — only P3 detect head is implemented in hardware
echo "▶ [—/5] P4 Detection Head SKIPPED (P3-only hardware)"
echo -n "" > yolov8_p4_output.mem
cp -f yolov8_p4_output.mem "$OUT_DIR/" 2>/dev/null
echo ""

# P5 SKIPPED
echo "▶ [—/5] P5 Detection Head SKIPPED"
echo -n "" > yolov8_p5_output.mem
cp -f yolov8_p5_output.mem "$OUT_DIR/" 2>/dev/null
echo ""

# 3. Run Hardware IoU NMS (P3 only)
echo "▶ [3/5] Compiling & Running Hardware IoU NMS..."
OBJ_DIR="$OBJ_BASE/nms"
rm -rf "$OBJ_DIR" && mkdir -p "$OBJ_DIR"
verilator --trace --Mdir $OBJ_DIR $VFLAGS_BASE $VCFLAGS --top-module tb_iou_nms -o iou_nms_sim_verilator \
    tb/tb_iou_nms.v \
    cnn/head/iou_nms_unit.v
$OBJ_DIR/iou_nms_sim_verilator
sync
cp -f tank_centers_iou.mem tank_centers.mem 2>/dev/null
cp -f tank_centers.mem "$OUT_DIR/" 2>/dev/null && echo "  ✅ NMS Output saved: $OUT_DIR/tank_centers.mem" || echo "  ⚠️  NMS output file not found"
echo ""

# 4. Visualize
echo "▶ [4/5] Visualizing Pure RTL Results vs Golden..."
cd "$SCRIPT_DIR"
python3 utils/visualize_hardware_nms.py "$IMG_PATH"

echo ""
echo "🎉 Pipeline Complete! Check 'rtl_detection_result_${basename}.jpg'"
