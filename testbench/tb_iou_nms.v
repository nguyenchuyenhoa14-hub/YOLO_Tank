/**
 * tb_iou_nms.v
 * Testbench for iou_nms_unit.v (with bbox pass-through)
 * 3-word output: centroid + bbox per detection
 */
`timescale 1ns/1ps

module tb_iou_nms;

localparam P3_BASE   = 22'h010000;
localparam P4_BASE   = 22'h020000;
localparam P5_BASE   = 22'h025000;
localparam OUT_BASE  = 22'h030000;
localparam MEM_SIZE  = 22'h040000;

reg [31:0] memory [0:MEM_SIZE-1];

reg  clk, rst_n, start;
wire done, busy;
wire [21:0] mem_rd_addr, mem_wr_addr;
wire        mem_rd_en,   mem_wr_en;
wire [31:0] mem_wr_data;
wire [31:0] nms_cycles;
reg  [10:0] p3_count;
reg  [8:0]  p4_count, p5_count;

iou_nms_unit #(
    .MAX_TOTAL      (64),
    .MAX_OUT_DETS   (40),
    .IOU_THRESH_NUM (9),
    .IOU_THRESH_DEN (20),
    .CONF_THRESH    (80),
    .BBOX_PAD       (0)
) dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .done(done), .busy(busy),
    .p3_base_addr(P3_BASE), .p3_count(p3_count),
    .p4_base_addr(P4_BASE), .p4_count(p4_count),
    .out_base_addr(OUT_BASE),
    .mem_rd_addr(mem_rd_addr), .mem_rd_en(mem_rd_en),
    .mem_rd_data(memory[mem_rd_addr]),
    .mem_wr_addr(mem_wr_addr), .mem_wr_en(mem_wr_en),
    .mem_wr_data(mem_wr_data)
);

always @(posedge clk) begin
    if (mem_wr_en) memory[mem_wr_addr] <= mem_wr_data;
end

always #5 clk = ~clk;

integer fd, ret, i, val;
integer p3_n, p4_n, p5_n;
reg [31:0] w0, w1, w2;
reg [9:0]  cx_v, cy_v, x1_v, y1_v, x2_v, y2_v;

initial begin
    clk = 0; rst_n = 0; start = 0;
    p3_count = 0; p4_count = 0; p5_count = 0;
    for (i = 0; i < MEM_SIZE; i = i + 1) memory[i] = 32'h0;

    // Load P3 only (P4/P5 not implemented in hardware)
    $display("Loading yolov8_p3_output.mem...");
    fd = $fopen("yolov8_p3_output.mem", "r");
    if (fd == 0) begin $display("  ❌ P3 not found."); $finish; end
    p3_n = 0;
    while (!$feof(fd)) begin
        ret = $fscanf(fd, "%h\n", val);
        if (ret == 1) begin memory[P3_BASE + p3_n] = val; p3_n = p3_n + 1; end
    end
    $fclose(fd);
    p3_count = p3_n / 5;
    p4_count = 0;
    p5_count = 0;
    $display("  ✅ Loaded %0d words → %0d P3 detections (P4/P5 disabled)", p3_n, p3_count);

    // Run DUT
    #20; rst_n = 1; #20;
    start = 1; #10; start = 0;
    wait(done); #20;

    $display("  NMS completed in %0d cycles", dut.nms_cycle_count);
    $display("  total_dets loaded = %0d", dut.total_dets);
    // Print first 5 loaded detections
    for (i = 0; i < 5 && i < dut.total_dets; i = i + 1) begin
        $display("    Det[%0d]: cx=%0d cy=%0d x1=%0d y1=%0d x2=%0d y2=%0d conf=%0d valid=%0b",
            i, dut.cx_buf[i], dut.cy_buf[i],
            dut.x1_buf[i], dut.y1_buf[i], dut.x2_buf[i], dut.y2_buf[i],
            $signed(dut.conf_buf[i]), dut.valid[i]);
    end
    // Count final valid
    begin : count_valid
        integer v_cnt;
        v_cnt = 0;
        for (i = 0; i < dut.total_dets; i = i + 1)
            if (dut.valid[i]) v_cnt = v_cnt + 1;
        $display("  Final valid detections after NMS: %0d", v_cnt);
    end
    $display("  Est. NMS time @100MHz = %0d us", dut.nms_cycle_count / 100);

    // Print results — 3 words per detection
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

    // Save to file
    fd = $fopen("tank_centers_iou.mem", "w");
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
    $display("\n  ✅ Saved → tank_centers_iou.mem");
    $finish;
end

endmodule
