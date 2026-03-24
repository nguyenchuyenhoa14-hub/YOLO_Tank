# ================================================================
# Timing Constraints for YOLOv8 AXI4 Full Wrapper
# Target: Xilinx Zynq-7020 (xc7z020clg400-1, speed grade -1)
# Top Module: yolov8_axi_wrapper_full
# ================================================================

# Primary clock: 100 MHz target (10ns period)
# AXI clock from Zynq PS FCLK_CLK0
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports S_AXI_ACLK]

# ================================================================
# I/O Delay Constraints
# ================================================================
# Input delay: AXI signals arrive 2ns after clock edge
set_input_delay  -clock clk 2.0 [get_ports -filter {DIRECTION == IN  && NAME != "S_AXI_ACLK"}]
# Output delay: AXI signals must be stable 2ns before next clock edge
set_output_delay -clock clk 2.0 [get_ports -filter {DIRECTION == OUT}]
