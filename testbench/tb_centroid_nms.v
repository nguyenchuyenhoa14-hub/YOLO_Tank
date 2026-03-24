/**
 * tb_centroid_nms.v
 * Testbench for centroid_nms_unit.v
 * Loads the existing sparse P3/P4/P5 output .mem files,
 * counts detections, runs the centroid NMS module,
 * and prints the final tank center coordinates.
 */
`timescale 1ns/1ps

module tb_centroid_nms;

// -------------------------------------------------------------------
// Memory layout (shared 22-bit address space)
// -------------------------------------------------------------------
localparam P3_BASE   = 22'h010000;
localparam P4_BASE   = 22'h020000;
localparam P5_BASE   = 22'h025000;
localparam OUT_BASE  = 22'h030000;
localparam MEM_SIZE  = 22'h040000;

reg [31:0] memory [0:MEM_SIZE-1];

// -------------------------------------------------------------------
// DUT signals
// -------------------------------------------------------------------
reg  clk, rst_n, start;
wire done, busy;
wire [21:0] mem_rd_addr, mem_wr_addr;
wire        mem_rd_en,   mem_wr_en;
wire [31:0] mem_wr_data;
reg  [10:0] p3_count;
reg  [8:0]  p4_count, p5_count;

// DUT
centroid_nms_unit #(
    .MAX_P3_DETS    (820),
    .MAX_P4_DETS    (100),
    .MAX_P5_DETS    (20),
    .MAX_TOTAL      (940),
    .MAX_OUT_DETS   (40),
    .P3_GRID        (40), .P4_GRID (20), .P5_GRID (10),
    .P3_STRIDE      (8),  .P4_STRIDE (16), .P5_STRIDE (32),
    .FRAC_BITS      (12),
    .DIST_THRESH_SQ (1024),
    .CONF_THRESH    (-17)
) dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (start),
    .done          (done),
    .busy          (busy),
    .p3_base_addr  (P3_BASE),
    .p3_count      (p3_count),
    .p4_base_addr  (P4_BASE),
    .p4_count      (p4_count),
    .p5_base_addr  (P5_BASE),
    .p5_count      (p5_count),
    .out_base_addr (OUT_BASE),
    .mem_rd_addr   (mem_rd_addr),
    .mem_rd_en     (mem_rd_en),
    .mem_rd_data   (memory[mem_rd_addr]),
    .mem_wr_addr   (mem_wr_addr),
    .mem_wr_en     (mem_wr_en),
    .mem_wr_data   (mem_wr_data)
);

// Memory write
always @(posedge clk) begin
    if (mem_wr_en) memory[mem_wr_addr] <= mem_wr_data;
end

// Clock
always #5 clk = ~clk;

// -------------------------------------------------------------------
// Main
// -------------------------------------------------------------------
integer fd, ret, i, val;
integer p3_n, p4_n, p5_n;

initial begin
    clk   = 0; rst_n = 0; start = 0;
    p3_count = 0; p4_count = 0; p5_count = 0;

    // Initialize memory to 0
    for (i = 0; i < MEM_SIZE; i = i + 1) memory[i] = 32'h0;

    // ---------------------------------------------------------------
    // Load P3 sparse bbox output
    // ---------------------------------------------------------------
    $display("Loading yolov8_p3_output.mem...");
    fd = $fopen("yolov8_p3_output.mem", "r");
    if (fd == 0) begin
        $display("  ❌ yolov8_p3_output.mem not found. Run P3 sim first."); $finish;
    end
    p3_n = 0;
    while (!$feof(fd)) begin
        ret = $fscanf(fd, "%h\n", val);
        if (ret == 1) begin
            memory[P3_BASE + p3_n] = val;
            p3_n = p3_n + 1;
        end
    end
    $fclose(fd);
    p3_count = p3_n / 5;
    $display("  ✅ Loaded %0d words → %0d P3 detections", p3_n, p3_count);

    // ---------------------------------------------------------------
    // Load P4 sparse bbox output
    // ---------------------------------------------------------------
    $display("Loading yolov8_full_output.mem...");
    fd = $fopen("yolov8_full_output.mem", "r");
    if (fd == 0) begin
        $display("  ❌ yolov8_full_output.mem not found. Run P4 sim first."); $finish;
    end
    p4_n = 0;
    while (!$feof(fd)) begin
        ret = $fscanf(fd, "%h\n", val);
        if (ret == 1) begin
            memory[P4_BASE + p4_n] = val;
            p4_n = p4_n + 1;
        end
    end
    $fclose(fd);
    p4_count = p4_n / 5;
    $display("  ✅ Loaded %0d words → %0d P4 detections", p4_n, p4_count);

    // ---------------------------------------------------------------
    // Load P5 sparse bbox output (optional — skip if not present)
    // ---------------------------------------------------------------
    $display("Loading yolov8_p5_output.mem...");
    fd = $fopen("yolov8_p5_output.mem", "r");
    if (fd == 0) begin
        $display("  ⚠️  yolov8_p5_output.mem not found. Skipping P5.");
        p5_count = 0;
    end else begin
        p5_n = 0;
        while (!$feof(fd)) begin
            ret = $fscanf(fd, "%h\n", val);
            if (ret == 1) begin
                memory[P5_BASE + p5_n] = val;
                p5_n = p5_n + 1;
            end
        end
        $fclose(fd);
        p5_count = p5_n / 5;
        $display("  ✅ Loaded %0d words → %0d P5 detections", p5_n, p5_count);
    end
    $display("");

    // ---------------------------------------------------------------
    // Run DUT
    // ---------------------------------------------------------------
    #20; rst_n = 1; #20;
    start = 1; #10; start = 0;

    wait(done);
    #20;

    // ---------------------------------------------------------------
    // Print results
    // ---------------------------------------------------------------
    $display("╔══════════════════════════════════════════╗");
    $display("║  TANK CENTER COORDINATES (Pixel 320×320) ║");
    $display("╚══════════════════════════════════════════╝");
    $display("  # |   cx  |   cy  ");
    $display("----+--------+--------");
    for (i = 0; i < 40; i = i + 1) begin
        if (memory[OUT_BASE + i] === 32'hDEADBEEF) begin
            i = 100;  // break
        end else begin
            $display("  %0d |  %4d  |  %4d  ",
                     i,
                     memory[OUT_BASE + i][25:16],  // cx upper 10 bits
                     memory[OUT_BASE + i][9:0]);    // cy lower 10 bits
        end
    end

    // Save to file
    fd = $fopen("tank_centers.mem", "w");
    for (i = 0; i < 40; i = i + 1) begin
        if (memory[OUT_BASE + i] === 32'hDEADBEEF) begin
            $fwrite(fd, "DEADBEEF\n");
            i = 100;
        end else begin
            $fwrite(fd, "%08x\n", memory[OUT_BASE + i]);
        end
    end
    $fclose(fd);
    $display("\n  ✅ Saved → tank_centers.mem");

    $finish;
end

endmodule
