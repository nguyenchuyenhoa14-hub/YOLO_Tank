/**
 * tb_p3_seq.v — P3 Sequential Detect Head Testbench
 *
 * Uses detect_head_seq module which runs 6 conv_stage instances sequentially.
 * Loads input, runs pipeline, then does DFL decode + threshold filtering.
 */
`timescale 1ns/1ps

module tb_p3_seq();

localparam GRID_W   = 40;
localparam GRID_H   = 40;
localparam IN_CH    = 64;
localparam MID_CH   = 64;
localparam BBOX_CH  = 64;
localparam NUM_CLASS = 1;
localparam PIX      = GRID_W * GRID_H;
localparam TOTAL_IN = PIX * IN_CH;
localparam CONF_THRESH = -128;
localparam DFL_BINS = 16;
localparam NUM_COORDS = 4;
localparam STRIDE   = 8;

// Clock & Reset
reg clk, rst_n;
initial begin clk = 0; forever #5 clk = ~clk; end

// DUT signals
reg        dut_start;
wire       dut_done, dut_busy;
reg  [7:0] ext_wr_data;
reg  [31:0] ext_wr_addr;
reg        ext_wr_en;
reg  [31:0] ext_rd_addr;
wire [7:0] ext_rd_bbox_data, ext_rd_cls_data;
wire [31:0] perf_counter;

detect_head_seq #(
    .GRID_W(GRID_W), .GRID_H(GRID_H), .IN_CH(IN_CH),
    .MID_CH(MID_CH), .BBOX_CH(BBOX_CH), .NUM_CLASS(NUM_CLASS),
    .PARALLEL(16), .CONF_THRESH(CONF_THRESH), .STRIDE(STRIDE),
    .CV2_0_W("weights_for_verilog_pe/model_22_cv2_0_0_conv_weight_quantized_pe.mem"),
    .CV2_0_B("weights_for_verilog_pe/model_22_cv2_0_0_conv_bias_quantized.mem"),
    .CV2_0_M("weights_for_verilog_pe/model_22_cv2_0_0_multiplier.mem"),
    .CV2_0_S("weights_for_verilog_pe/model_22_cv2_0_0_shift.mem"),
    .CV2_1_W("weights_for_verilog_pe/model_22_cv2_0_1_conv_weight_quantized_pe.mem"),
    .CV2_1_B("weights_for_verilog_pe/model_22_cv2_0_1_conv_bias_quantized.mem"),
    .CV2_1_M("weights_for_verilog_pe/model_22_cv2_0_1_multiplier.mem"),
    .CV2_1_S("weights_for_verilog_pe/model_22_cv2_0_1_shift.mem"),
    .CV2_2_W("weights_for_verilog_pe/model_22_cv2_0_2_conv_weight_quantized_pe.mem"),
    .CV2_2_B("weights_for_verilog_pe/model_22_cv2_0_2_conv_bias_quantized.mem"),
    .CV2_2_M("weights_for_verilog_pe/model_22_cv2_0_2_multiplier.mem"),
    .CV2_2_S("weights_for_verilog_pe/model_22_cv2_0_2_shift.mem"),
    .CV3_0_W("weights_for_verilog_pe/model_22_cv3_0_0_conv_weight_quantized_pe.mem"),
    .CV3_0_B("weights_for_verilog_pe/model_22_cv3_0_0_conv_bias_quantized.mem"),
    .CV3_0_M("weights_for_verilog_pe/model_22_cv3_0_0_multiplier.mem"),
    .CV3_0_S("weights_for_verilog_pe/model_22_cv3_0_0_shift.mem"),
    .CV3_1_W("weights_for_verilog_pe/model_22_cv3_0_1_conv_weight_quantized_pe.mem"),
    .CV3_1_B("weights_for_verilog_pe/model_22_cv3_0_1_conv_bias_quantized.mem"),
    .CV3_1_M("weights_for_verilog_pe/model_22_cv3_0_1_multiplier.mem"),
    .CV3_1_S("weights_for_verilog_pe/model_22_cv3_0_1_shift.mem"),
    .CV3_2_W("weights_for_verilog_pe/model_22_cv3_0_2_conv_weight_quantized_pe.mem"),
    .CV3_2_B("weights_for_verilog_pe/model_22_cv3_0_2_conv_bias_quantized.mem"),
    .CV3_2_M("weights_for_verilog_pe/model_22_cv3_0_2_multiplier.mem"),
    .CV3_2_S("weights_for_verilog_pe/model_22_cv3_0_2_shift.mem"),
    .CV2_0_Z("weights_for_verilog_pe/model_22_cv2_0_0_zero_point.mem"),
    .CV2_1_Z("weights_for_verilog_pe/model_22_cv2_0_1_zero_point.mem"),
    .CV2_2_Z("weights_for_verilog_pe/model_22_cv2_0_2_zero_point.mem"),
    .CV3_0_Z("weights_for_verilog_pe/model_22_cv3_0_0_zero_point.mem"),
    .CV3_1_Z("weights_for_verilog_pe/model_22_cv3_0_1_zero_point.mem"),
    .CV3_2_Z("weights_for_verilog_pe/model_22_cv3_0_2_zero_point.mem")
) dut (
    .clk(clk), .rst_n(rst_n),
    .start(dut_start), .done(dut_done), .busy(dut_busy),
    .ext_wr_data(ext_wr_data), .ext_wr_addr(ext_wr_addr), .ext_wr_en(ext_wr_en),
    .ext_rd_addr(ext_rd_addr),
    .ext_rd_bbox_data(ext_rd_bbox_data),
    .ext_rd_cls_data(ext_rd_cls_data),
    .ext_w_rd_data(8'd0),
    .ext_b_rd_data(32'd0),
    .ext_m_rd_data(32'd0),
    .ext_s_rd_data(5'd0),
    .ext_z_rd_data(8'd0),
    .perf_counter(perf_counter)
);

// DFL Accelerator
reg         dfl_start;
wire        dfl_done, dfl_busy;
reg  [127:0] dfl_logits;
wire signed [15:0] dfl_coord;

dfl_accelerator #(
    .LUT_FILE("exp_lut_p3.mem")
) dfl_inst (
    .clk(clk), .rst_n(rst_n),
    .start(dfl_start), .done(dfl_done), .busy(dfl_busy),
    .logits_flat(dfl_logits), .coord_out(dfl_coord)
);

// Main test
integer i, scan_val;
integer input_fd, output_fd;
integer cell_idx, coord;
integer det_count, class_idx;
reg signed [7:0] conf, best_conf, max_conf_seen;
reg [7:0] best_class;
reg signed [7:0] bbox_val;
reg signed [15:0] decoded_coords [0:3];
reg [5:0] gx_out, gy_out;  // 6-bit grid coords for output packing

initial begin
    max_conf_seen = -128;
    rst_n = 0;
    dut_start = 0;
    ext_wr_en = 0;
    ext_wr_addr = 0;
    ext_wr_data = 0;
    ext_rd_addr = 0;
    dfl_start = 0;

    #100; rst_n = 1; #100;

    $display("╔══════════════════════════════════════════╗");
    $display("║  P3 Sequential Pipeline Testbench       ║");
    $display("║  Grid: 40x40  IN_CH: 64  Stride: 8     ║");
    $display("╚══════════════════════════════════════════╝\n");

    $dumpfile("p3_seq.vcd");
    $dumpvars(0, tb_p3_seq);


    // Load input feature map into DUT's input_ram
    $display("Loading layer15_output.mem...");
    input_fd = $fopen("layer15_output.mem", "r");
    if (input_fd == 0) begin $display("ERROR: Cannot open layer15_output.mem"); $finish; end
    for (i = 0; i < TOTAL_IN; i = i + 1) begin
        if ($fscanf(input_fd, "%h\n", scan_val) != 1) begin
            $display("WARNING: EOF at i=%0d", i); i = TOTAL_IN;
        end else begin
            @(posedge clk);
            ext_wr_addr = i;
            ext_wr_data = scan_val[7:0];
            ext_wr_en   = 1;
        end
    end
    @(posedge clk); ext_wr_en = 0;
    $fclose(input_fd);
    $display("  ✅ Loaded %0d values\n", TOTAL_IN);

    // Start pipeline
    @(negedge clk);
    dut_start = 1;
    @(negedge clk);
    dut_start = 0;

    $display("  [P3-SEQ] Pipeline started...");

    // Wait for completion
    fork
        begin
            wait(dut_done);
            $display("  ✅ Pipeline complete! cycles=%0d", perf_counter);
            $display("  [R3] Est. FPS @100MHz = %0d FPS", 100000000 / (perf_counter + 1));
            $display("  [R3] Est. FPS @150MHz = %0d FPS", 150000000 / (perf_counter + 1));
        end
        begin
            #(64'd80000000); // 80ms timeout
            if (!dut_done) begin 
                $display("  ❌ TIMEOUT: perf_counter=%0d", perf_counter);
                $display("  DEBUG: fsm_state=%0d active_stage=%0d rd_pix=%0d rd_pass=%0d conv_pix_valid=%0d conv_pix_ready=%0d conv_act_valid=%0d conv_act_ready=%0d", 
                    dut.fsm_state, dut.active_stage, dut.rd_pix, dut.rd_pass, dut.conv_pix_valid, dut.conv_pix_ready, dut.conv_act_valid, dut.conv_act_ready);
                $finish; 
            end
        end
    join_any
    #50;



    // ── DFL Decode + Threshold Filter ──
    $display("\n  [P3-SEQ] Running DFL decode on %0d cells...", PIX);
    det_count = 0;
    output_fd = $fopen("yolov8_p3_output.mem", "w");

    for (cell_idx = 0; cell_idx < PIX; cell_idx = cell_idx + 1) begin
        best_conf = -128;
        best_class = 0;
        
        for (class_idx = 0; class_idx < NUM_CLASS; class_idx = class_idx + 1) begin
            ext_rd_addr = cell_idx * NUM_CLASS + class_idx;
            @(posedge clk); 
            @(posedge clk); #1; // Wait 1 more cycle for synchronous BRAM read
            conf = $signed(ext_rd_cls_data);
            if (conf > best_conf) begin
                best_conf = conf;
                best_class = class_idx;
            end
            if (conf > max_conf_seen) begin
                max_conf_seen = conf;
            end
        end

        if (1) begin
            // Decode 4 bbox coordinates using DFL
            for (coord = 0; coord < NUM_COORDS; coord = coord + 1) begin
                for (i = 0; i < DFL_BINS; i = i + 1) begin
                    ext_rd_addr = cell_idx * BBOX_CH + coord * DFL_BINS + i;
                    @(posedge clk); 
                    @(posedge clk); #1; // Wait 1 more cycle for synchronous BRAM read
                    dfl_logits[(i*8)+7 -: 8] = ext_rd_bbox_data;
                end
                
                if (det_count < 10) begin
                    $display("  DFL Det#%0d Cell(%0d,%0d) Coord%0d(%s): %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        det_count, cell_idx % GRID_W, cell_idx / GRID_W, coord,
                        (coord==0) ? "L" : (coord==1) ? "T" : (coord==2) ? "R" : "B",
                        $signed(dfl_logits[7:0]), $signed(dfl_logits[15:8]), $signed(dfl_logits[23:16]), $signed(dfl_logits[31:24]),
                        $signed(dfl_logits[39:32]), $signed(dfl_logits[47:40]), $signed(dfl_logits[55:48]), $signed(dfl_logits[63:56]),
                        $signed(dfl_logits[71:64]), $signed(dfl_logits[79:72]), $signed(dfl_logits[87:80]), $signed(dfl_logits[95:88]),
                        $signed(dfl_logits[103:96]), $signed(dfl_logits[111:104]), $signed(dfl_logits[119:112]), $signed(dfl_logits[127:120])
                    );
                end
                @(negedge clk); dfl_start = 1;
                @(negedge clk); dfl_start = 0;
                wait(dfl_done);
                decoded_coords[coord] = dfl_coord;
                if (det_count < 3)
                    $display("    DFL coord_out[%0d] = %0d (0x%04x)", coord, $signed(dfl_coord), dfl_coord);
                @(posedge clk);
            end

            // ── Weighted Neighbor Anchor Correction ──
            // Read left neighbor's confidence and encode it into Word 4
            // for the NMS unit to perform the correction in hardware.
            begin : neighbor_check
                reg signed [7:0] left_conf;
                left_conf = -128;  // default: no neighbor
                
                if ((cell_idx % GRID_W) > 0) begin
                    ext_rd_addr = (cell_idx - 1) * NUM_CLASS + best_class;
                    @(posedge clk);
                    @(posedge clk); #1;
                    left_conf = $signed(ext_rd_cls_data);
                end
                
                $fwrite(output_fd, "%08x\n", {16'd0, decoded_coords[0]});
                $fwrite(output_fd, "%08x\n", {16'd0, decoded_coords[1]});
                $fwrite(output_fd, "%08x\n", {16'd0, decoded_coords[2]});
                $fwrite(output_fd, "%08x\n", {16'd0, decoded_coords[3]});
                // Word 4: [31:24] = grid_y, [23:16] = grid_x
                //          [15:8] = left_neighbor_conf (signed INT8)
                //          [7:0]  = confidence
                gx_out = cell_idx % GRID_W;
                gy_out = cell_idx / GRID_W;
                $fwrite(output_fd, "%08x\n", {2'b0, gy_out, 2'b0, gx_out, left_conf, best_conf});
                if (det_count < 10 && left_conf > 0)
                    $display("    ↰ Left neighbor conf=%0d (packed into word4[15:8])", $signed(left_conf));
            end

            if (det_count < 30)
                $display("  Det#%0d: Cell[%0d] Grid(%0d,%0d) cls=%0d conf=%0d",
                    det_count, cell_idx, cell_idx % GRID_W, cell_idx / GRID_W, best_class, $signed(best_conf));
            det_count = det_count + 1;
        end
    end

    $fclose(output_fd);
    $display("\n  Total detections: %0d", det_count);
    $display("  ✅ Saved yolov8_p3_output.mem");
    $display("  [P3-SEQ] MAX CONF SEEN GLOBALLY: %0d", max_conf_seen);

    $display("\n╔══════════════════════════════════════════╗");
    $display("║  P3 Sequential Pipeline: COMPLETE       ║");
    $display("╚══════════════════════════════════════════╝");
    $finish;
end




// Timeout watchdog
initial begin
    #(64'd18000000000000);
    $display("TIMEOUT");
    $finish;
end

endmodule
