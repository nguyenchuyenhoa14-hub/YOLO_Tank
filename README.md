# 🚀 YOLO-Tank

**A Resource-Efficient FPGA Accelerator for YOLOv8 Detection Head with On-Chip DFL Decoding and Division-Free NMS**

> Submitted to *International Conference on Intelligent Autonomous Agents and Applications (IAAA 2026)*

---

## 📸 Demo — Hardware Detection Result

<p align="center">
  <img src="demo/detection_result.jpg" width="500"/>
</p>

<p align="center"><i>Real-time military vehicle detection on Zynq-7020 FPGA (KIIT-MiTA dataset, 320×320 input)</i></p>

---

## 🏗️ Architecture Overview

```
ARM Cortex-A9 (PS)                    Programmable Logic (PL)
┌──────────────┐    AXI4 burst    ┌─────────────────────────────────────┐
│  DDR Memory  │ ───────────────► │  Master FSM (6-Stage Sequential)    │
│  (Backbone   │                  │      ┌──────────┐  ┌──────────┐    │
│   + Neck     │                  │      │  RAM A   │  │  RAM B   │    │
│   output)    │                  │      └────┬─────┘  └─────┬────┘    │
│              │                  │           │  ping-pong    │         │
│              │                  │      ┌────┴──────────────┴────┐    │
│              │                  │      │   CNN Engine (16× MACs) │    │
│              │                  │      │   512-bit parallel bus  │    │
│              │                  │      └────────────┬───────────┘    │
│              │                  │                   │                 │
│              │                  │      ┌────────────▼───────────┐    │
│              │                  │      │    DFL Accelerator     │    │
│              │                  │      │  (256-entry LUT Softmax)│    │
│              │                  │      └────────────┬───────────┘    │
│              │                  │                   │                 │
│              │  ◄─── Results ── │      ┌────────────▼───────────┐    │
│              │                  │      │   Division-Free NMS    │    │
│              │                  │      │  (5+4 stage pipeline)  │    │
└──────────────┘                  │      └────────────────────────┘    │
                                  └─────────────────────────────────────┘
```

---

## ✨ Key Innovations

| Innovation | Description |
|:-----------|:------------|
| **Resource-multiplexed Conv** | Single configurable convolution engine reused across all 6 head layers via time-division multiplexing with ping-pong BRAM buffering |
| **Hardware DFL Decoder** | 256-entry BRAM LUT replaces floating-point Softmax; sequential integer divider produces Q0.16 probabilities |
| **Division-Free IoU NMS** | Cross-multiplication `inter × D > union × N` eliminates hardware dividers; 5-stage load + 4-stage IoU pipeline at 100 MHz |
| **Three-Phase Quantization** | ReLU Knowledge Distillation → PCQ-Aware Fine-Tuning → INT8 Static Export |

---

## 📊 Implementation Results — Zynq-7020 (xc7z020clg400-1)

| Metric | Value |
|:-------|:------|
| Detection Head Throughput | **31.0 FPS** @ 100 MHz |
| LUT Usage | 15,401 / 53,200 (28.95%) |
| FF Usage | 22,660 / 106,400 (21.30%) |
| BRAM Usage | 126 / 140 (90.00%) |
| DSP48 Usage | 154 / 220 (70.00%) |
| PL Dynamic Power | 0.279 W |
| mAP₅₀ (INT8, KIIT-MiTA) | 67.0% |
| WNS (post P&R) | +0.122 ns |

---

## 📁 Repository Structure

```
YOLO_Tank/
│
├── README.md
│
├── source/                          # RTL source — Detection Head
│   ├── detect_head_seq.v            #   Sequential 6-layer FSM controller
│   ├── conv_stage.v                 #   Configurable convolution stage
│   ├── cnn_engine_dynamic.v         #   16-PE MAC array engine
│   ├── line_buffer.v                #   BRAM-based 3×3 sliding window
│   ├── dfl_accelerator.v            #   5-stage DFL pipeline + LUT Softmax
│   ├── seq_divider.v                #   Sequential restoring divider
│   ├── iou_nms_unit.v               #   Division-free IoU NMS unit
│   ├── yolov8_top_core.v            #   Top-level detection head core
│   ├── yolov8_axi_wrapper_full.v    #   AXI4 bus wrapper
│   ├── ooc_timing.xdc               #   Out-of-context timing constraints
│   ├── ooc_timing_axi.xdc           #   AXI timing constraints
│   └── yolov8_system.xdc            #   Full system constraints
│
├── testbench/                       # Verilator & Verilog testbenches
│   ├── tb_p3_seq.v                  #   P3 scale detection head test
│   ├── tb_p3_system.v               #   P3 system-level test
│   ├── tb_iou_nms.v                 #   IoU NMS unit test
│   ├── tb_centroid_nms.v            #   Centroid NMS test
│   ├── tb_full_pipeline.v           #   Full pipeline integration test
│   ├── tb_axi_wrapper.cpp           #   Verilator AXI wrapper test
│   ├── tb_yolov8_top_core.cpp       #   Verilator top core test
│   └── tb_cycle_measure.cpp         #   Cycle-accurate timing measurement
│
├── weights_and_mem/                 # INT8 quantized weights & LUT data
│   ├── weights_for_verilog_pe/      #   Per-layer weight .mem files
│   │   ├── model_22_cv2_*           #     BBox branch (3 layers × 6 params)
│   │   └── model_22_cv3_*           #     Class branch (3 layers × 6 params)
│   ├── exp_lut_p3.mem               #   DFL exponential lookup table
│   ├── tank_centers.mem             #   NMS detection output
│   └── tank_centers_iou.mem         #   NMS IoU debug output
│
├── python_utils/                    # Python verification & visualization
│   ├── golden_all_layers.py         #   Golden model for all 6 head layers
│   ├── check_rtl_vs_python_accuracy.py  # RTL vs golden comparison
│   ├── visualize_hardware_nms.py    #   NMS result visualization
│   ├── regenerate_all.py            #   Regenerate all .mem from model
│   └── regenerate_all_kiit.py       #   Regenerate for KIIT-MiTA dataset
│
├── scripts/                         # Simulation shell scripts
│   ├── run_all_verilator.sh         #   Run all Verilator simulations
│   └── run_cycle_measure.sh         #   Measure pipeline cycle count
│
└── demo/                            # Demo images & detection results
    ├── image_s3r2_kiit_402.jpeg     #   Input test image (320×320)
    └── detection_result.jpg         #   Hardware detection output
```

---

## 🔧 Quick Start

### Prerequisites

- **Verilator** ≥ 4.0 (RTL simulation)
- **Python 3.8+** with `numpy`, `torch`, `ultralytics`
- **Vivado 2024.2** (synthesis & implementation)

### Run Verilator Simulation

```bash
# 1. Regenerate weight .mem files from trained model
cd python_utils
python3 regenerate_all.py

# 2. Run detection head simulation
cd ../scripts
bash run_all_verilator.sh

# 3. Visualize NMS results
cd ../python_utils
python3 visualize_hardware_nms.py
```

### Vivado Synthesis

```bash
# Source files are in source/
# Constraints: yolov8_system.xdc (full) or ooc_timing.xdc (OOC)
# Top module: yolov8_top_core
```

---

## 📚 Dataset

Trained and evaluated on the [KIIT-MiTA Military Vehicle Dataset](https://doi.org/10.1109/ISACC65211.2025.10969335)
— 2,139 images, single class (military vehicle), 320×320 resolution.

---

## 📝 Citation

```bibtex
@inproceedings{nguyen2026yolotank,
  title   = {YOLO-Tank: A Resource-Efficient FPGA Accelerator for YOLOv8
             Detection Head with On-Chip DFL Decoding and Division-Free NMS},
  author  = {Vo, Hoang Nguyen and Tran, Vo Hai Dang and Vo, Tuan Binh},
  booktitle = {Proc. Int. Conf. Intelligent Autonomous Agents and Applications (IAAA)},
  year    = {2026}
}
```

---

## 📄 License

This project is released for **academic research purposes only**.

---

<p align="center">
  <b>YOLO-Tank</b> — Bringing real-time object detection to the edge with FPGA 🛡️
</p>
