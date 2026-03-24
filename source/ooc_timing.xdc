# ================================================================
# Timing Constraints for YOLOv8 Top Core
# Target: Xilinx Zynq-7020 (xc7z020clg400-1, speed grade -1)
# ================================================================

# Primary clock: 100 MHz target (10ns period)
create_clock -period 9.524 -name clk -waveform {0.000 4.762} [get_ports clk]

# ================================================================
# I/O Delay Constraints
# ================================================================
# Input delay: data arrives 2ns after clock edge
set_input_delay  -clock clk 2.0 [get_ports -filter {DIRECTION == IN  && NAME != "clk"}]
# Output delay: data must be stable 2ns before next clock edge
set_output_delay -clock clk 2.0 [get_ports -filter {DIRECTION == OUT}]

# ================================================================
# Multicycle Path Exceptions (if needed)
# ================================================================
# Weight loading is not timing-critical (PS writes at slow rate)
# set_multicycle_path 2 -setup -from [get_ports ext_w_rd_data*]
# set_multicycle_path 1 -hold  -from [get_ports ext_w_rd_data*]

# ================================================================
# Physical Optimization Directives
# ================================================================
# Instruct Vivado to perform extra timing-driven placement
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# Phys_opt: aggressive optimization after placement
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# Place: extra timing optimization
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt [get_runs impl_1]

# Route: aggressive exploration for better QoR
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# Post-route phys_opt (additional optimization pass)
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
