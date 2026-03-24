/**
 * tb_full_pipeline.v — Full Integrated Pipeline Testbench
 *
 * Tests the complete Conv → DFL → NMS pipeline in yolov8_top_core.
 * Loads input feature map, starts pipeline, waits for done,
 * reads NMS results from shared memory.
 */
`timescale 1ns/1ps

module tb_full_pipeline();

/* verilator lint_off UNUSEDPARAM */
parameter CONF_THRESH = 67;
/* verilator lint_on UNUSEDPARAM */

localparam GRID_W   = 40;
localparam GRID_H   = 40;
localparam IN_CH    = 64;
localparam PIX      = GRID_W * GRID_H;
localparam TOTAL_IN = PIX * IN_CH;

localparam P3_BASE   = 22'h010000;
localparam OUT_BASE  = 22'h030000;
localparam MEM_SIZE  = 22'h040000;

// Shared memory model
reg [31:0] memory [0:MEM_SIZE-1];

// Clock & Reset
reg clk = 0;
always #5 clk = ~clk;
reg rst_n;

// DUT signals
reg        dut_start;
wire       dut_done, dut_busy, need_reload;
reg  [7:0] ext_wr_data;
reg  [31:0] ext_wr_addr;
reg        ext_wr_en;

// Shared memory interface
wire [21:0] mem_wr_addr, mem_rd_addr;
wire        mem_wr_en, mem_rd_en;
wire [31:0] mem_wr_data;

// Performance outputs
wire [31:0] perf_det, perf_nms;

yolov8_top_core #(
    .GRID_W(GRID_W), .GRID_H(GRID_H),
    .IN_CH(IN_CH), .MID_CH(64), .BBOX_CH(64),
    .NUM_CLASS(1), .PARALLEL(16), .STRIDE(8),
    .CONF_THRESH_DET(CONF_THRESH),
    .CONF_THRESH_DFL(CONF_THRESH),
    .DFL_BINS(16),
    .MAX_P3_DETS(48), .MAX_TOTAL(64), .MAX_OUT_DETS(20),
    .IOU_THRESH_NUM(9), .IOU_THRESH_DEN(20),
    .CONF_THRESH_NMS(CONF_THRESH), .FRAC_BITS(12)
) dut (
    .clk(clk), .rst_n(rst_n),
    .start(dut_start), .done(dut_done), .busy(dut_busy),
    .need_reload(need_reload),
    .ext_wr_en(ext_wr_en), .ext_wr_addr(ext_wr_addr), .ext_wr_data(ext_wr_data),
    .ext_w_rd_data(8'd0), .ext_b_rd_data(32'd0),
    .ext_m_rd_data(32'd0), .ext_s_rd_data(5'd0), .ext_z_rd_data(8'd0),
    .nms_mem_wr_addr(mem_wr_addr), .nms_mem_wr_en(mem_wr_en), .nms_mem_wr_data(mem_wr_data),
    .nms_mem_rd_addr(mem_rd_addr), .nms_mem_rd_en(mem_rd_en), .nms_mem_rd_data(memory[mem_rd_addr]),
    .p3_det_count(11'd0),   // Unused — DFL computes internally
    .p4_det_count(9'd0),
    .p3_base_addr(P3_BASE),
    .p4_base_addr(22'h020000),
    .nms_out_base_addr(OUT_BASE),
    .perf_det_cycles(perf_det),
    .perf_nms_cycles(perf_nms)
);

// Shared memory write
always @(posedge clk) begin
    if (mem_wr_en) begin
        memory[mem_wr_addr] <= mem_wr_data;
        // Debug: print DFL writes (first 5 detections × 5 words = 25 writes)
        if (mem_wr_addr >= P3_BASE && mem_wr_addr < P3_BASE + 25)
            $display("  [DFL-WR] addr=%06x data=%08x (det#%0d word%0d)",
                mem_wr_addr, mem_wr_data,
                (mem_wr_addr - P3_BASE) / 5, (mem_wr_addr - P3_BASE) % 5);
    end
end

// Performance counter — total cycles from start to done
reg [31:0] total_cycles;
always @(posedge clk) begin
    if (dut_start) total_cycles <= 0;
    else if (dut_busy) total_cycles <= total_cycles + 1;
end

integer i, scan_val, input_fd, fd;
reg [31:0] w0, w1, w2;
reg [9:0] cx_v, cy_v, x1_v, y1_v, x2_v, y2_v;

initial begin
    rst_n = 0; dut_start = 0;
    ext_wr_en = 0; ext_wr_addr = 0; ext_wr_data = 0;
    for (i = 0; i < MEM_SIZE; i = i + 1) memory[i] = 32'h0;

    #100; rst_n = 1; #100;

    $display("╔══════════════════════════════════════════════════╗");
    $display("║  FULL PIPELINE: Conv → DFL → NMS               ║");
    $display("║  Grid: 40x40  IN_CH: 64  Stride: 8            ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    // Load input feature map
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

    // Start full pipeline
    @(negedge clk); dut_start = 1;
    @(negedge clk); dut_start = 0;

    $display("  [PIPELINE] Started... Conv → DFL → NMS");

    // Wait for completion
    fork
        begin
            wait(dut_done);
            $display("\n  ════════════════════════════════════════════");
            $display("  ✅ PIPELINE COMPLETE!");
            $display("  ════════════════════════════════════════════");
            $display("  Total cycles     = %0d", total_cycles);
            $display("  Conv cycles      = %0d", perf_det);
            $display("  NMS cycles       = %0d", perf_nms);
            $display("  DFL+scan cycles  = %0d", total_cycles - perf_det - perf_nms);
            $display("  DFL detections   = %0d", dut.df_det_count);
            $display("  ────────────────────────────────────────────");
            $display("  Latency @100MHz  = %0d.%02d ms", total_cycles / 100000, (total_cycles % 100000) / 1000);
            $display("  FPS @100MHz      = %0d", 100000000 / (total_cycles + 1));
            $display("  FPS @150MHz      = %0d", 150000000 / (total_cycles + 1));
            $display("  ════════════════════════════════════════════\n");
        end
        begin
            #(64'd100000000); // 100ms timeout
            if (!dut_done) begin
                $display("  ❌ TIMEOUT at %0d cycles", total_cycles);
                $display("  top_state=%0d df_state=%0d df_cell=%0d df_coord=%0d df_logit=%0d df_det=%0d",
                    dut.top_state, dut.df_state, dut.df_cell_idx, dut.df_coord_idx, dut.df_logit_idx, dut.df_det_count);
                $finish;
            end
        end
    join_any
    #50;

    // Print results
    $display("╔══════════════════════════════════════════════════════════╗");
    $display("║  TANK DETECTIONS (IoU NMS) — Center + BBox             ║");
    $display("╚══════════════════════════════════════════════════════════╝");
    $display("  # |   cx  |   cy  |  x1  |  y1  |  x2  |  y2  ");
    $display("----+-------+-------+------+------+------+------");
    for (i = 0; i < 40; i = i + 1) begin
        w0 = memory[OUT_BASE + i*3];
        if (w0 === 32'hDEADBEEF) i = 100;
        else begin
            w1 = memory[OUT_BASE + i*3 + 1];
            w2 = memory[OUT_BASE + i*3 + 2];
            cx_v = w0[25:16]; cy_v = w0[9:0];
            x1_v = w1[25:16]; y1_v = w1[9:0];
            x2_v = w2[25:16]; y2_v = w2[9:0];
            $display("  %0d | %4d  | %4d  | %3d  | %3d  | %3d  | %3d",
                     i, cx_v, cy_v, x1_v, y1_v, x2_v, y2_v);
        end
    end

    // Save results
    fd = $fopen("tank_centers_integrated.mem", "w");
    for (i = 0; i < 40; i = i + 1) begin
        w0 = memory[OUT_BASE + i*3];
        if (w0 === 32'hDEADBEEF) begin
            $fwrite(fd, "DEADBEEF\n"); i = 100;
        end else begin
            w1 = memory[OUT_BASE + i*3 + 1];
            w2 = memory[OUT_BASE + i*3 + 2];
            $fwrite(fd, "%08x %08x %08x\n", w0, w1, w2);
        end
    end
    $fclose(fd);
    $display("\n  ✅ Saved → tank_centers_integrated.mem");
    $finish;
end

// Timeout watchdog
initial begin
    #(64'd18000000000000);
    $display("ULTRA TIMEOUT"); $finish;
end

endmodule
