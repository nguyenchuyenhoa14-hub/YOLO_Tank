`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module: Line Buffer (RAM Based, Dynamic Width)
// Description: Sliding window buffer for 2D convolution
//              Uses block RAMs/Distributed RAMs instead of registers for efficiency
//              Supports runtime configurable image width
//////////////////////////////////////////////////////////////////////////////////

/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off PINCONNECTEMPTY */
module line_buffer #(
    parameter MAX_WIDTH = 640,  // Maximum supported width
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire soft_reset,
    input wire enable,
    input wire [DATA_WIDTH-1:0] padding_val, // Dynamic padding value
    input wire [DATA_WIDTH-1:0] pixel_in,
    input wire pixel_valid,
    input wire [9:0] current_width,
    input wire padding_en,  // 1=enable padding (Conv), 0=disable (MaxPool) // Dynamic width (e.g. 192, 96, 48...)
    
    // 3x3 window output
    // Window layout:
    // 0_0 0_1 0_2 (Top Row) - delayed by 2 lines
    // 1_0 1_1 1_2 (Mid Row) - delayed by 1 line
    // 2_0 2_1 2_2 (Bot Row) - current stream
    output wire [DATA_WIDTH-1:0] window_0_0, window_0_1, window_0_2,
    output wire [DATA_WIDTH-1:0] window_1_0, window_1_1, window_1_2,
    output wire [DATA_WIDTH-1:0] window_2_0, window_2_1, window_2_2,
    output wire window_valid
);

    // RAMs for line storage
    reg [DATA_WIDTH-1:0] line_ram0 [0:MAX_WIDTH-1];
    reg [DATA_WIDTH-1:0] line_ram1 [0:MAX_WIDTH-1];
    
    integer i;
    
    initial begin
        for (i=0; i<MAX_WIDTH; i=i+1) begin
            line_ram0[i] = 0;
            line_ram1[i] = 0;
        end
    end
    
    // Read data from RAMs
    reg [DATA_WIDTH-1:0] rdata0, rdata1;
    
    // Shift registers for 3x3 window (3 taps per row)
    // Row 0: From RAM1 (Oldest)
    // Row 1: From RAM0 (Mid)
    // Row 2: From Input (Newest)
    reg [DATA_WIDTH-1:0] row0_reg [0:KERNEL_SIZE-1];
    reg [DATA_WIDTH-1:0] row1_reg [0:KERNEL_SIZE-1];
    reg [DATA_WIDTH-1:0] row2_reg [0:KERNEL_SIZE-1];
    
    // Counters
    reg [9:0] col_cnt;
    reg [9:0] row_cnt;
    
    reg [DATA_WIDTH-1:0] pixel_in_d; // Delay reg
    
    // Initialize shift registers to prevent X values in simulation
    initial begin
        for (i=0; i<KERNEL_SIZE; i=i+1) begin
            row0_reg[i] = 0;
            row1_reg[i] = 0;
            row2_reg[i] = 0;
        end
        col_cnt = 0;
        row_cnt = 0;
        pixel_in_d = 0;
    end
    
    // Explicit Read-Before-Write Logic to prevents "Transparent Write" issues
    // We read the OLD value from RAMs into wires before the clock edge
    wire [DATA_WIDTH-1:0] old_ram0_val = line_ram0[col_cnt[5:0]];
    wire [DATA_WIDTH-1:0] old_ram1_val = line_ram1[col_cnt[5:0]];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt <= 0;
            row_cnt <= 0;
            // Clear shift registers
            for (i=0; i<KERNEL_SIZE; i=i+1) begin
                row0_reg[i] <= padding_val;
                row1_reg[i] <= padding_val;
                row2_reg[i] <= padding_val;
            end
            // CRITICAL FIX: Clear RAM arrays to padding_val (zero-point)
            // Previously cleared to 0 which contaminated first 2 rows 
            // when zero-point is -128 (0x80)
            for (i=0; i<MAX_WIDTH; i=i+1) begin
                line_ram0[i] <= padding_val;
                line_ram1[i] <= padding_val;
            end
        end else if (soft_reset) begin
            col_cnt <= 0;
            row_cnt <= 0;
            // Clear shift registers
            for (i=0; i<KERNEL_SIZE; i=i+1) begin
                row0_reg[i] <= padding_val;
                row1_reg[i] <= padding_val;
                row2_reg[i] <= padding_val;
            end
            // Clear RAMs to padding_val on soft reset
            for (i=0; i<MAX_WIDTH; i=i+1) begin
                line_ram0[i] <= padding_val;
                line_ram1[i] <= padding_val;
            end
        end else if (enable && pixel_valid) begin
            // 1. Shift Window Registers (Move previous values)
            // Using loop from 1 to KERNEL_SIZE-1 is fine with non-blocking assignments
            for (i = 1; i < KERNEL_SIZE; i = i + 1) begin
                row2_reg[i] <= row2_reg[i-1];
                row1_reg[i] <= row1_reg[i-1];
                row0_reg[i] <= row0_reg[i-1];
            end
            
            // 2. Load New Values into Head of Window (Index 0)
            // Row 2 (Bottom): Current Pixel
            row2_reg[0] <= pixel_in;
            // Row 1 (Mid): Old value from RAM0 (guaranteed old by wire read)
            row1_reg[0] <= old_ram0_val;
            // Row 0 (Top): Old value from RAM1
            row0_reg[0] <= old_ram1_val;

            // 3. Update RAMs for next pass (Cascade)
            line_ram0[col_cnt[5:0]] <= pixel_in;     // Store current pixel in RAM0
            line_ram1[col_cnt[5:0]] <= old_ram0_val; // Cascade old RAM0 value to RAM1
            
            // 4. Update Counters
            // DEBUG TRACE removed
                     
            if (col_cnt == current_width - 1) begin
                col_cnt <= 0;
                if (row_cnt < KERNEL_SIZE + 1) // Count up to enough rows
                    row_cnt <= row_cnt + 1;
            end else begin
                col_cnt <= col_cnt + 1;
            end
        end
    end
    
    // Handle RAM read latency?
    // If strict RAM:
    // Cycle 1: Addr setup (col_cnt).
    // Cycle 2: Data valid.
    // If Distributed RAM (LUTRAM): Data valid same cycle (async read).
    // Let's assume Distributed RAM for small buffers (640x8 is small).
    // Async read: assign rdata = mem[addr];
    // This allows single cycle update.
    
    // SAME Padding Logic (Combinatorial)
    // BUGFIX: Window center = reg[1]. With non-blocking assignments, reg[1] has the pixel
    // from 2 cycles ago (pixel at col_cnt-2). For the center to be at x=0 (first column),
    // we need col_cnt to be 2. But the RIGHT neighbor reg[0] also needs to have advanced:
    // reg[0] has pixel from col_cnt-1. So at col_cnt=3: reg[1]=pixel[1], reg[0]=pixel[2].
    // At col_cnt=2: reg[1]=pixel[0], reg[0]=pixel[1]. This should be correct.
    // The issue: pad_left should fire when center is at x=0 → col_cnt=2.
    // pad_right should fire when center is at x=W-1 → col_cnt=W-1+2=W+1 → wraps → col_cnt=1.
    wire pad_top = (row_cnt == 1) || (row_cnt == 2 && col_cnt < 2);
    wire pad_left = (col_cnt == 2);
    wire pad_right = (col_cnt == 1);
 
    // window_valid is now registered in the pipeline stage below
    // ================================================================
    // PIPELINE STAGE: Register all window outputs (TIMING FIX)
    // Breaks critical path: BRAM/pixel_in → [REG] → DSP48 multiply
    // Without: BRAM cascade(3.3ns) + mux + pixel_in → DSP48(3.7ns setup) = 12.8ns
    // With:    Path1 ≈ 6ns, Path2 ≈ 5.5ns → 100MHz achievable
    // ================================================================
    
    // Pre-padding combinational values
    wire [DATA_WIDTH-1:0] w00_pre = (padding_en && (pad_top || pad_left)) ? padding_val : row0_reg[2];
    wire [DATA_WIDTH-1:0] w01_pre = (padding_en && pad_top) ? padding_val : row0_reg[1];
    wire [DATA_WIDTH-1:0] w02_pre = (padding_en && (pad_top || pad_right)) ? padding_val : old_ram1_val;
    wire [DATA_WIDTH-1:0] w10_pre = (padding_en && pad_left) ? padding_val : row1_reg[2];
    wire [DATA_WIDTH-1:0] w11_pre = row1_reg[1];
    wire [DATA_WIDTH-1:0] w12_pre = (padding_en && pad_right) ? padding_val : old_ram0_val;
    wire [DATA_WIDTH-1:0] w20_pre = (padding_en && pad_left) ? padding_val : row2_reg[1];
    wire [DATA_WIDTH-1:0] w21_pre = row2_reg[0];
    wire [DATA_WIDTH-1:0] w22_pre = (padding_en && pad_right) ? padding_val : pixel_in;
    wire                  wv_pre  = (row_cnt > 1) ? 1'b1 : (row_cnt == 1 && col_cnt >= 2);

    // Registered output stage
    reg [DATA_WIDTH-1:0] win_r00, win_r01, win_r02;
    reg [DATA_WIDTH-1:0] win_r10, win_r11, win_r12;
    reg [DATA_WIDTH-1:0] win_r20, win_r21, win_r22;
    reg                  win_rv;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_r00 <= 0; win_r01 <= 0; win_r02 <= 0;
            win_r10 <= 0; win_r11 <= 0; win_r12 <= 0;
            win_r20 <= 0; win_r21 <= 0; win_r22 <= 0;
            win_rv  <= 0;
        end else if (enable && pixel_valid) begin
            win_r00 <= w00_pre; win_r01 <= w01_pre; win_r02 <= w02_pre;
            win_r10 <= w10_pre; win_r11 <= w11_pre; win_r12 <= w12_pre;
            win_r20 <= w20_pre; win_r21 <= w21_pre; win_r22 <= w22_pre;
            win_rv  <= wv_pre;
        end else begin
            win_rv  <= 0;  // only valid for 1 cycle per pixel
        end
    end

    assign window_0_0 = win_r00;
    assign window_0_1 = win_r01;
    assign window_0_2 = win_r02;
    assign window_1_0 = win_r10;
    assign window_1_1 = win_r11;
    assign window_1_2 = win_r12;
    assign window_2_0 = win_r20;
    assign window_2_1 = win_r21;
    assign window_2_2 = win_r22;
    assign window_valid = win_rv;

endmodule
