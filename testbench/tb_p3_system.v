/**
 * P3 Detection Head Testbench
 *
 * Loads layer15_output.mem (40x40x64 P3 feature map)
 * Runs yolov8_detect_head_p3
 * Saves output to yolov8_p3_output.mem (40x40x5 = 8000 values)
 */
`timescale 1ns/1ps

module tb_p3_system();

// Parameters
localparam GRID_W    = 40;
localparam GRID_H    = 40;
localparam IN_CH     = 64;
localparam OUT_CH    = 5;
localparam TOTAL_IN  = GRID_W * GRID_H * IN_CH;  // 102400
localparam TOTAL_OUT = GRID_W * GRID_H * OUT_CH;  // 8000

localparam INPUT_BASE  = 22'h000000;
localparam OUTPUT_BASE = 22'h100000;

// Clock & Reset
reg clk;
reg rst_n;
reg start;
wire done, busy;

// Memory model (8MB)
reg [31:0] memory [0:2097151];

// DUT signals
wire [21:0] mem_rd_addr, mem_wr_addr;
wire mem_rd_en, mem_wr_en;
wire [31:0] mem_rd_data, mem_wr_data;

assign mem_rd_data = memory[mem_rd_addr];
always @(posedge clk) begin
    if (mem_wr_en) memory[mem_wr_addr] <= mem_wr_data;
end

// DUT: P3 detect head
yolov8_detect_head_p3 dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .done(done), .busy(busy),
    .input_base_addr(INPUT_BASE),
    .output_base_addr(OUTPUT_BASE),
    .mem_rd_addr(mem_rd_addr), .mem_rd_en(mem_rd_en), .mem_rd_data(mem_rd_data),
    .mem_wr_addr(mem_wr_addr), .mem_wr_en(mem_wr_en), .mem_wr_data(mem_wr_data)
);

// Clock: 100MHz
initial begin clk = 0; forever #5 clk = ~clk; end

integer i, scan_val;
integer input_fd, output_fd;
reg signed [31:0] px_val;
reg signed [7:0]  conf_s8;
integer best_idx;
reg signed [7:0] best_conf;
integer total_cells;

initial begin
    // Init
    rst_n = 0; start = 0;
    // Clear Input Region
    for (i = 0; i < TOTAL_IN; i = i + 1) memory[INPUT_BASE+i] = 32'b0;
    // Set Output Region to INVALID (DEADBEEF)
    for (i = 0; i < TOTAL_OUT; i = i + 1) memory[OUTPUT_BASE+i] = 32'hDEADBEEF;
    
    #100; rst_n = 1; #100;

    $display("╔══════════════════════════════════════════╗");
    $display("║  YOLOv8 P3 Detection Head Testbench     ║");
    $display("║  Grid: 40x40  IN_CH: 64  Stride: 8      ║");
    $display("╚══════════════════════════════════════════╝\n");

    // ─── Load input feature map ───
    $display("Loading layer15_output.mem (%0d values)...", TOTAL_IN);
    input_fd = $fopen("layer15_output.mem", "r");
    if (input_fd == 0) begin
        $display("ERROR: Cannot open layer15_output.mem"); $finish;
    end
    for (i = 0; i < TOTAL_IN; i = i + 1) begin
        if (i % 20000 == 0) $display("  ... read progress: %0d / %0d", i, TOTAL_IN);
        if ($fscanf(input_fd, "%h\n", scan_val) != 1) begin
            $display("WARNING: EOF at i=%0d", i); i = TOTAL_IN;
        end else begin
            memory[INPUT_BASE + i] = scan_val & 32'hFF;
        end
    end
    $fclose(input_fd);
    $display("  ✅ Loaded %0d values from layer15_output.mem\n", TOTAL_IN);

    // ─── Run simulation ───
    start = 1; #10; start = 0;
    wait(done); #50;

    // ─── Save output (Sparse) ───
    $display("\nSaving yolov8_p3_output.mem...");
    output_fd = $fopen("yolov8_p3_output.mem", "w");
    // Write until DEADBEEF or max
    for (i = 0; i < TOTAL_OUT; i = i + 1) begin
        px_val = $signed(memory[OUTPUT_BASE + i]);
        if (memory[OUTPUT_BASE + i] === 32'hDEADBEEF) begin
            // Stop writing at marker
            $display("  [Info] Hit DEADBEEF at index %0d, stopping save.", i);
            i = TOTAL_OUT; // break
        end else begin
            $fwrite(output_fd, "%08x\n", px_val);
        end
    end
    $fclose(output_fd);
    $display("  ✅ Saved valid detections\n");

    // ─── Analysis (Sparse) ───
    $display("=== P3 VALID DETECTIONS (Hardware Filtered) ===");
    best_conf = -8'sd128; best_idx = 0;
    
    // Iterate through valid blocks of 5 words
    for (i = 0; i < TOTAL_OUT/OUT_CH; i = i + 1) begin
        if (memory[OUTPUT_BASE + i*OUT_CH] === 32'hDEADBEEF) begin
             i = TOTAL_OUT; // break loop
        end else begin
            // Decode Packed Info
            // Word 4: {cell_idx[15:0], 8'b0, conf[7:0]}
            scan_val = memory[OUTPUT_BASE + i*OUT_CH + 4];
            conf_s8 = $signed(scan_val[7:0]);
            
            // Extract Cell Index from upper 16 bits
            // If sign extended, mask it. RTL put cell_idx in [31:16] literally? 
            // In RTL: mem_wr_data <= {wr_cell[15:0], ...}
            // So top 16 bits are cell_idx.
            // If wr_cell is small positive, bit 31 is 0.
            
            best_idx = (scan_val >> 16) & 16'hFFFF; 
            
            $display("  Det#%0d: Cell[%0d] Grid(%0d,%0d) conf=%0d  l=%0d t=%0d r=%0d b=%0d",
                i, best_idx, best_idx%GRID_W, best_idx/GRID_W, $signed(conf_s8),
                $signed(memory[OUTPUT_BASE + i*OUT_CH + 0]),
                $signed(memory[OUTPUT_BASE + i*OUT_CH + 1]),
                $signed(memory[OUTPUT_BASE + i*OUT_CH + 2]),
                $signed(memory[OUTPUT_BASE + i*OUT_CH + 3]));
        end
    end

    $display("\n╔══════════════════════════════════════════╗");
    $display("║  P3 Simulation Complete!                 ║");
    $display("╚══════════════════════════════════════════╝");
    $finish;
end

endmodule
