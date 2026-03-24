#!/bin/bash
cd /mnt/c/github/rtl/yolo_complete

echo "--------------------------------------------------------"
echo "Compiling Per-Layer Cycle Measurement Testbench..."
echo "--------------------------------------------------------"

rm -rf obj_dir_measure

verilator --cc --exe --build --timing -CFLAGS "-std=c++20" -Wno-fatal -Wno-TIMESCALEMOD -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL \
    --public-flat-rw \
    --top-module yolov8_top_core \
    -Mdir obj_dir_measure \
    -o cycle_measure_test \
    tb/tb_cycle_measure.cpp \
    cnn/vivado_src/yolov8_top_core.v \
    cnn/vivado_src/detect_head_seq.v \
    cnn/vivado_src/conv_stage.v \
    cnn/vivado_src/cnn_engine_dynamic.v \
    cnn/vivado_src/line_buffer.v \
    cnn/vivado_src/dfl_accelerator.v \
    cnn/vivado_src/seq_divider.v \
    cnn/vivado_src/iou_nms_unit.v

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "Running Cycle Measurement..."
    echo "--------------------------------------------------------"
    ./obj_dir_measure/cycle_measure_test
else
    echo "Compilation Failed."
fi
