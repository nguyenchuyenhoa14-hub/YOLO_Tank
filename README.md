# YOLO-Tank: FPGA Accelerator for YOLOv8 Detection Head

A resource-efficient FPGA accelerator implementing the complete YOLOv8 detection head — convolution, DFL decoding, and division-free IoU NMS — entirely in hardware on a Zynq-7020.

## Architecture Overview

```
ARM Cortex-A9 (PS)                    Programmable Logic (PL)
┌──────────────┐    AXI4 burst    ┌─────────────────────────────────────┐
│  DDR Memory  │ ───────────────► │  Master FSM (6-Stage Sequential)    │
│  Backbone    │                  │      ┌──────────┐  ┌──────────┐    │
│  + Neck      │                  │      │  RAM A   │  │  RAM B   │    │
│              │                  │      └────┬─────┘  └─────┬────┘    │
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
│              │                  │      ┌────────────▼───────────┐    │
│              │  ◄─── Results ── │      │   Division-Free NMS    │    │
│              │                  │      │  (5+4 stage pipeline)  │    │
└──────────────┘                  │      └────────────────────────┘    │
                                  └─────────────────────────────────────┘
```

## Key Features

- **Resource-multiplexed pipeline**: Single configurable convolution engine reused across all 6 head layers via time-division multiplexing with ping-pong BRAM buffering
- **Hardware DFL decoder**: 256-entry BRAM LUT replaces floating-point Softmax exponentials; sequential integer divider produces Q0.16 probabilities
- **Division-free IoU NMS**: Cross-multiplication eliminates hardware dividers entirely; 5-stage load + 4-stage IoU pipeline achieves 100 MHz timing closure
- **Three-phase quantization**: Knowledge Distillation → PCQ-Aware Fine-Tuning → INT8 export

## Results (Zynq-7020, xc7z020clg400-1)

| Metric | Value |
|--------|-------|
| Detection head throughput | **31.0 FPS** @ 100 MHz |
| LUT | 15,401 (28.95%) |
| FF | 22,660 (21.30%) |
| BRAM | 126 / 140 (90.00%) |
| DSP48 | 154 / 220 (70.00%) |
| PL dynamic power | 0.279 W |
| mAP₅₀ (INT8, KIIT-MiTA) | 67.0% |
| WNS (post place-and-route) | +0.122 ns |

## Repository Structure

```
├── rtl/yolo_complete/
│   ├── cnn/
│   │   ├── head/              # Detection head RTL (main source)
│   │   │   ├── detect_head_seq.v      # Sequential detection head FSM
│   │   │   ├── conv_stage.v           # Configurable convolution stage
│   │   │   ├── cnn_engine_dynamic.v   # Dynamic CNN engine with 16 PEs
│   │   │   ├── line_buffer.v          # BRAM-based 3×3 sliding window
│   │   │   ├── dfl_accelerator.v      # 5-stage DFL pipeline
│   │   │   ├── seq_divider.v          # Sequential restoring divider
│   │   │   ├── iou_nms_unit.v         # Division-free IoU NMS
│   │   │   ├── backbone_seq.v         # Backbone sequential controller
│   │   │   ├── yolov8_top_core.v      # Top-level core
│   │   │   ├── yolov8_axi_wrapper.v   # AXI4 interface wrapper
│   │   │   ├── create_block_design.tcl # Vivado block design script
│   │   │   └── *.xdc                  # Timing constraints
│   │   ├── vivado_src/        # Full system Verilog (backbone + head)
│   │   └── save/              # Archived RTL versions
│   ├── tb/                    # Verilator/Verilog testbenches
│   ├── vitis/                 # Zynq PS bare-metal C code
│   ├── weights_for_verilog_pe/    # INT8 weight .mem files (head)
│   ├── weights_for_verilog_pcq/   # PCQ quantization parameters
│   ├── weights_backbone_pe/       # Backbone weight .mem files
│   ├── exp_lut_p3.mem         # DFL exponential LUT (P3 scale)
│   ├── input_image_320.mem    # Test image (320×320)
│   └── tank_centers.mem       # NMS detection output
├── utils/                     # Python verification & visualization
│   ├── golden_all_layers.py           # Golden model for all layers
│   ├── visualize_hardware_nms.py      # NMS result visualization
│   ├── check_rtl_vs_python_accuracy.py # RTL vs golden comparison
│   └── ...
├── reports/                   # Vivado post-implementation reports
│   ├── utilization.txt
│   ├── utilization_hierarchy.txt
│   ├── timing_summary.txt
│   └── power.txt
├── pic_test/                  # Test images
├── kiitmita_tank/             # Training results & configs
├── regenerate_all.py          # Regenerate all .mem files from model
├── run_full_verilator.sh      # Run head-only Verilator simulation
├── run_all_verilator.sh       # Run all layer verification
└── export_image_to_rtl.py     # Convert test image to .mem format
```

## Quick Start

### Prerequisites
- Verilator (≥ 4.0)
- Python 3.8+ with `numpy`, `torch`, `ultralytics`
- Vivado 2024.2 (for synthesis)

### Run Verilator Simulation
```bash
# Regenerate weight .mem files from trained model
python3 regenerate_all.py

# Run detection head simulation
bash run_full_verilator.sh

# Visualize NMS results
python3 utils/visualize_hardware_nms.py
```

### Vivado Synthesis
```bash
# Source files are in rtl/yolo_complete/cnn/head/
# Use create_block_design.tcl for block design generation
# Constraints: yolov8_system.xdc (full system) or ooc_timing.xdc (OOC)
```

## Dataset

Trained on [KIIT-MiTA Military Vehicle Dataset](https://doi.org/10.1109/ISACC65211.2025.10969335) (2,139 images, single class, 320×320).

## Citation

```bibtex
@inproceedings{nguyen2026yolotank,
  title   = {YOLO-Tank: A Resource-Efficient FPGA Accelerator for YOLOv8
             Detection Head with On-Chip DFL Decoding and Division-Free NMS},
  author  = {Vo, Hoang Nguyen and Tran, Vo Hai Dang and Vo, Tuan Binh},
  booktitle = {Proc. Int. Conf. Intelligent Autonomous Agents and Applications (IAAA)},
  year    = {2026}
}
```

## License

This project is released for academic research purposes.
