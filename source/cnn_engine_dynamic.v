`timescale 1ns/1ps
////////////////////////////////////////////////////////////////////////////////
// Module: Dynamic CNN Engine (PARALLEL BUS OUTPUT - No Serialization)
// Description: Generalized CNN engine for YOLOv8n execution
//              - Support dynamic dimensions (192, 96, ..., 6)
//              - "Input Channel Stationary" architecture
//              - PARALLEL_OUT parallel Processing Engines
//              - External weight interface
//              - Dynamic Kernel Size Support (1x1 and 3x3)
//
// KEY OPTIMIZATION: Eliminates ST_OUTPUT serialization bottleneck.
//   Old Path: Compute(1 clk) → Serialize 16 results(16+ clk) → Repeat
//   New Path: Compute(1 clk) → Fire wide bus(1 clk) → Repeat immediately
//
// Interface change: result_out is now a WIDE BUS [32*PARALLEL_OUT-1:0]
//   and result_valid_fix is a 1-cycle pulse (no ACK needed from conv_stage).
////////////////////////////////////////////////////////////////////////////////

/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off PINCONNECTEMPTY */
module cnn_engine_dynamic #(
    parameter MAX_WIDTH    = 320,
    parameter PARALLEL_OUT = 16,
    parameter KERNEL       = 3,
    parameter UNSIGNED_IN  = 0
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire frame_start,
    input wire [7:0] padding_val,

    // Configuration
    input wire [9:0] img_width,
    input wire [9:0] img_height,
    input wire [9:0] in_channels,
    input wire [9:0] out_channels,
    input wire [1:0] stride,
    input wire [2:0] kernel_size,
    input wire       unsigned_in_cfg,

    input  wire [7:0] pixel_in,
    input  wire       pixel_valid_fix,
    output wire       pixel_ready_fix,
    input  wire       eng_ready,

    // Weight Interface
    input wire [7:0] weight_in,
    input wire       weight_wr,
    input wire [9:0]  weight_addr,
    input wire [31:0] bias_in,
    input wire        bias_wr,
    input wire [5:0]  bias_addr,

    // * PARALLEL RESULT BUS *
    // All PARALLEL_OUT results are valid simultaneously for 1 clock cycle
    // result_valid_fix is a 1-cycle pulse — no ACK handshake needed
    output reg signed [32*PARALLEL_OUT-1:0] result_bus,
    output reg        result_valid_fix,

    output reg done,
    input wire [9:0] out_blk_cnt_dbg,
    input wire [9:0] in_ch_cnt_dbg
);

    //--------------------------------------------------------------------------
    // 1. Line Buffer
    //--------------------------------------------------------------------------
    wire [7:0] win[0:8];
    wire [7:0] w00, w01, w02;
    wire [7:0] w10, w11, w12;
    wire [7:0] w20, w21, w22;
    wire lb_win_valid;

    assign win[0] = w00; assign win[1] = w01; assign win[2] = w02;
    assign win[3] = w10; assign win[4] = w11; assign win[5] = w12;
    assign win[6] = w20; assign win[7] = w21; assign win[8] = w22;

    reg [9:0] col_cnt, row_cnt;
    reg valid_last_col;


    wire frame_done = (kernel_size != 1) ? (row_cnt == img_height + 1 && col_cnt == 1 && pixel_valid_fix)
                                         : (row_cnt == img_height - 1 && col_cnt == img_width - 1 && pixel_valid_fix);
    wire lb_soft_reset = start || frame_done;

    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb_inst (
        .clk(clk), .rst_n(rst_n),
        .pixel_in(pixel_in), .pixel_valid(pixel_valid_fix),
        .current_width(img_width),
        .padding_en(1'b1),
        .padding_val(padding_val),
        .enable(eng_ready),
        .soft_reset(lb_soft_reset),

        .window_0_0(w00), .window_0_1(w01), .window_0_2(w02),
        .window_1_0(w10), .window_1_1(w11), .window_1_2(w12),
        .window_2_0(w20), .window_2_1(w21), .window_2_2(w22),
        .window_valid(lb_win_valid)
    );

    wire win_valid_3x3 = lb_win_valid;

    // ---- Pixel input register (TIMING FIX for 1x1 kernel path) ----
    // For 1x1 kernels, pixel_in bypasses the line buffer and goes directly
    // to the DSP48 multiply. This register breaks that critical path.
    // For 3x3, the line_buffer's window output register handles the fix.
    reg [7:0] pixel_in_r;
    reg       pixel_valid_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_in_r    <= 0;
            pixel_valid_r <= 0;
        end else begin
            pixel_in_r    <= pixel_in;
            pixel_valid_r <= pixel_valid_fix;
        end
    end

    // For 1x1: use registered valid (matches registered pixel_in_r data)
    // For 3x3: use lb_win_valid (from line_buffer, already registered)
    wire win_valid = (kernel_size == 1) ? pixel_valid_r : win_valid_3x3;


    //--------------------------------------------------------------------------
    // 2. Weight Storage + Fan-out Reduction Pipeline
    //    Pipeline registers + MAX_FANOUT force Vivado to auto-replicate
    //    the data register, reducing routing delay from 8.7ns to ~3ns.
    //    The simulator sees a plain scalar (bit-exact simulation).
    //--------------------------------------------------------------------------
    reg signed [7:0] weights [0:PARALLEL_OUT-1][0:2][0:8];
    reg signed [31:0] biases [0:255] /* verilator public */;

    // ── Pipeline registers with fanout control ──
    // MAX_FANOUT forces Vivado to replicate w_pipe_data into multiple copies
    // (DONT_TOUCH must NOT be used here, as it forbids replication)
    (* MAX_FANOUT = 10 *) reg signed [7:0] w_pipe_data;
    reg               w_pipe_wr;
    reg [9:0]         w_pipe_addr;
    reg [2:0]         w_pipe_ksize;
    (* MAX_FANOUT = 16 *) reg signed [31:0] b_pipe_data;
    reg               b_pipe_wr;
    reg [5:0]         b_pipe_addr;
    reg [9:0]         b_pipe_oblk;

    integer w_i, w_j;

    always @(posedge clk) begin
        // ── Stage 1: Capture inputs (1 cycle delay) ──
        w_pipe_data  <= $signed(weight_in);
        w_pipe_wr    <= weight_wr;
        w_pipe_addr  <= weight_addr;
        w_pipe_ksize <= kernel_size;
        b_pipe_data  <= $signed(bias_in);
        b_pipe_wr    <= bias_wr;
        b_pipe_addr  <= bias_addr;
        b_pipe_oblk  <= out_blk_cnt_dbg;

        // ── Stage 2: Write to arrays from pipelined registers ──
        if (w_pipe_wr) begin
            if (w_pipe_ksize == 1) begin
                if (w_pipe_addr < PARALLEL_OUT)
                    weights[w_pipe_addr[5:0]][0][4] <= w_pipe_data;
            end else begin
                if (w_pipe_addr < PARALLEL_OUT * 9)
                    weights[(w_pipe_addr / 9)][0][(w_pipe_addr % 9)] <= w_pipe_data;
            end
        end
        if (b_pipe_wr)
            biases[b_pipe_oblk * PARALLEL_OUT + b_pipe_addr] <= b_pipe_data;
    end

    initial begin
        for (w_i=0; w_i<PARALLEL_OUT; w_i=w_i+1) begin
            for (w_j=0; w_j<9; w_j=w_j+1) begin
                weights[w_i][0][w_j] = 0;
                weights[w_i][1][w_j] = 0;
                weights[w_i][2][w_j] = 0;
            end
            biases[w_i] = 0;
        end
    end

    //--------------------------------------------------------------------------
    // 3. Processing Engines + State Machine (3-Stage Pipelined MAC)
    //    Stage C1:  Register multiply products  (8×8 → 16-bit registered)
    //    Stage C2a: Partial adder tree           (3 groups of 3 → 3 sums)
    //    Stage C2b: Final sum + bias             (sum of 3 partial + bias → result_bus)
    //
    //    CRITICAL: pixel_ready_fix is ONLY HIGH in ST_STREAM.
    //    ST_FLUSH1/2/3 drain the 3-deep pipeline without consuming pixels.
    //--------------------------------------------------------------------------
    
    localparam ST_WAIT   = 3'd0;
    localparam ST_STREAM = 3'd1;
    localparam ST_FLUSH1 = 3'd2;
    localparam ST_FLUSH2 = 3'd3;
    localparam ST_FLUSH3 = 3'd4;
    reg [2:0] state;

    reg stride_aligned;

    // *** BUG FIX ***: Only accept pixels in ST_STREAM.
    assign pixel_ready_fix = (state == ST_STREAM) ? 1'b1 : 1'b0;

    // --- Pipeline Stage C1: multiply products ---
    reg signed [15:0] c1_prod [0:PARALLEL_OUT-1][0:8];
    reg signed [31:0] c1_bias [0:PARALLEL_OUT-1];
    reg               c1_valid;
    reg               c1_done;

    // --- Pipeline Stage C2a: partial sums (3 groups of 3) ---
    reg signed [31:0] c2a_psum [0:PARALLEL_OUT-1][0:2];  // 3 partial sums per PE
    reg signed [31:0] c2a_bias [0:PARALLEL_OUT-1];
    reg               c2a_valid;
    reg               c2a_done;

    // Temporary variables for C1 combinational logic (blocking assigns)
    reg signed [15:0] tmp_p, tmp_w;

    reg bias_enable;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) bias_enable <= 0;
        else if (start) bias_enable <= frame_start;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt          <= 0;
            row_cnt          <= 0;
            result_valid_fix <= 0;
            done             <= 0;
            state            <= ST_WAIT;
            c1_valid         <= 0;
            c1_done          <= 0;
            c2a_valid        <= 0;
            c2a_done         <= 0;
            result_bus       <= 0;

        end else begin
            // Compute stride alignment (combinational)
            if (stride == 1) begin
                stride_aligned = 1;
            end else begin
                stride_aligned = (row_cnt[0] == 1) && (col_cnt[0] == 0);
            end

            // Default: pipeline advances, clear pulses
            c1_valid         <= 0;
            c1_done          <= 0;
            c2a_valid        <= c1_valid;
            c2a_done         <= c1_done;
            result_valid_fix <= c2a_valid;
            done             <= c2a_done;

            // ---- Stage C2a: Partial adder tree (3 groups of 3 products) ----
            if (c1_valid) begin
                for (w_i = 0; w_i < PARALLEL_OUT; w_i = w_i + 1) begin
                    // Group 0: products [0]+[1]+[2]
                    c2a_psum[w_i][0] <=
                        {{16{c1_prod[w_i][0][15]}}, c1_prod[w_i][0]} +
                        {{16{c1_prod[w_i][1][15]}}, c1_prod[w_i][1]} +
                        {{16{c1_prod[w_i][2][15]}}, c1_prod[w_i][2]};
                    // Group 1: products [3]+[4]+[5]
                    c2a_psum[w_i][1] <=
                        {{16{c1_prod[w_i][3][15]}}, c1_prod[w_i][3]} +
                        {{16{c1_prod[w_i][4][15]}}, c1_prod[w_i][4]} +
                        {{16{c1_prod[w_i][5][15]}}, c1_prod[w_i][5]};
                    // Group 2: products [6]+[7]+[8]
                    c2a_psum[w_i][2] <=
                        {{16{c1_prod[w_i][6][15]}}, c1_prod[w_i][6]} +
                        {{16{c1_prod[w_i][7][15]}}, c1_prod[w_i][7]} +
                        {{16{c1_prod[w_i][8][15]}}, c1_prod[w_i][8]};
                    // Pass bias through
                    c2a_bias[w_i] <= c1_bias[w_i];


                end
            end

            // ---- Stage C2b: Final sum of 3 partial sums + bias → result_bus ----
            if (c2a_valid) begin
                for (w_i = 0; w_i < PARALLEL_OUT; w_i = w_i + 1) begin
                    result_bus[(w_i*32)+31 -: 32] <= (bias_enable ? c2a_bias[w_i] : 32'sd0) +
                        c2a_psum[w_i][0] +
                        c2a_psum[w_i][1] +
                        c2a_psum[w_i][2];
                end
            end


            // ---- Main FSM ----
            case (state)
                ST_WAIT: begin
                    if (start) begin
                        state   <= ST_STREAM;
                        col_cnt <= 0;
                        row_cnt <= 0;
                    end
                end

                ST_STREAM: begin

                    // Update counters on valid pixel
                    if (pixel_valid_fix && eng_ready) begin
                        if (col_cnt == img_width - 1) begin
                            col_cnt <= 0;
                            if (kernel_size == 1 && row_cnt == img_height - 1) begin
                                row_cnt <= 0;
                                c1_done <= 1;
                                state   <= ST_FLUSH1;
                            end else begin
                                row_cnt <= row_cnt + 1;
                            end
                        end else if (kernel_size != 1 && row_cnt == img_height + 1 && col_cnt == 1) begin
                            row_cnt <= 0;
                            col_cnt <= 0;
                            c1_done <= 1;
                            state   <= ST_FLUSH1;
                        end else begin
                            col_cnt <= col_cnt + 1;
                        end
                    end

                    // ---- Stage C1: Register multiply products ----
                    if (pixel_valid_fix && win_valid && stride_aligned && eng_ready) begin
                        c1_valid <= 1;


                        for (w_i = 0; w_i < PARALLEL_OUT; w_i = w_i + 1) begin
                            // Bias (pass-through, registered)
                            if (in_ch_cnt_dbg == 0) begin
                                c1_bias[w_i] = biases[out_blk_cnt_dbg * PARALLEL_OUT + w_i];

                            end else
                                c1_bias[w_i] = 0;

                            // Multiply products (blocking for Verilator compatibility)
                            for (w_j = 0; w_j < 9; w_j = w_j + 1) begin
                                if (kernel_size == 1) begin
                                    if (w_j == 4) begin
                                        tmp_p = unsigned_in_cfg ? $signed(pixel_in_r ^ 8'h80)
                                                            : $signed(pixel_in_r);
                                        tmp_w = $signed(weights[w_i][0][4]);
                                    end else begin
                                        tmp_p = 0;
                                        tmp_w = 0;
                                    end
                                end else begin
                                    tmp_p = unsigned_in_cfg ? $signed(win[w_j] ^ 8'h80)
                                                        : $signed(win[w_j]);
                                    tmp_w = $signed(weights[w_i][0][w_j]);
                                end
                                
                                c1_prod[w_i][w_j] = tmp_p * tmp_w;

                            end
                        end
                    end
                end

                ST_FLUSH1: begin
                    // Pipeline drain cycle 1: C1 → C2a
                    state <= ST_FLUSH2;
                end

                ST_FLUSH2: begin
                    // Pipeline drain cycle 2: C2a → C2b
                    state <= ST_FLUSH3;
                end

                ST_FLUSH3: begin
                    // Pipeline drain cycle 3: C2b → output
                    state <= ST_WAIT;
                end

                default: state <= ST_WAIT;
            endcase
        end
    end

endmodule

