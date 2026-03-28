/**
 * conv_stage.v — Pipelined Conv2D Stage (HARDWARE REUSE VERSION)
 *
 * REFACTORED: Runtime-configurable single instance.
 *   - Weight ROMs externalized (addresses output, data input)
 *   - Simulation: zero-latency combinational reads from parent
 *   - Synthesis: 1-cycle latency from external memory
 */

`timescale 1ns/1ps

/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off PINCONNECTEMPTY */
module conv_stage #(
    parameter IMG_W      = 40,
    parameter IMG_H      = 40,
    parameter MAX_IN_CH  = 128,
    parameter MAX_OUT_CH = 64,
    parameter PARALLEL   = 16
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output reg  done,

    // Runtime configuration
    input  wire [9:0]  cfg_in_ch,
    input  wire [9:0]  cfg_out_ch,
    input  wire [2:0]  cfg_kernel,
    input  wire        cfg_relu_en,
    input  wire        cfg_unsigned_in,
    input  wire [7:0]  cfg_padding_val,
    input  wire [9:0]  cfg_img_w,
    input  wire [9:0]  cfg_img_h,

    // Input pixel stream
    input  wire [7:0]  pix_in,
    input  wire        pix_valid,
    output wire        pix_ready,

    // Output activation stream
    output wire [7:0]  act_out,
    output wire        act_valid,
    input  wire        act_ready,

    // External Weight ROM interface
    output wire [31:0] w_rd_addr,
    input  wire [7:0]  w_rd_data,

    // External Bias/M/S/Z ROM interface
    output wire [9:0]  bmsz_rd_addr,
    input  wire signed [31:0] b_rd_data,
    input  wire signed [31:0] m_rd_data,
    input  wire [4:0]         s_rd_data,
    input  wire signed [7:0]  z_rd_data,
    output wire        done_tick,
    output wire [9:0]  in_ch_cnt_dbg
);

localparam PIX = IMG_W * IMG_H;

reg [9:0] r_in_ch, r_out_ch;
reg [2:0] r_kernel;
reg       r_relu_en;
reg       r_unsigned_in;
reg [9:0] r_w_max;
reg [9:0] r_n_out_blks;
// Deleted unused cfg registers

// Pixel counters for Line Buffer
reg [10:0] rd_pix;
reg [10:0] q_pix;
reg [10:0] q_pix_reg;
// ============================================================
// FIFO
// ============================================================
localparam FIFO_DEPTH = 32;
localparam FIFO_BITS  = 5;
reg [7:0]  fifo_mem [0:FIFO_DEPTH-1];
reg [FIFO_BITS-1:0] fifo_wr_ptr, fifo_rd_ptr;
reg [FIFO_BITS:0]   fifo_cnt;
wire fifo_full  = (fifo_cnt == FIFO_DEPTH);
wire fifo_empty = (fifo_cnt == 0);

assign act_valid = !fifo_empty;
assign act_out   = fifo_mem[fifo_rd_ptr];

// ============================================================
// FSM States
// ============================================================
localparam S_IDLE       = 4'd0;
localparam S_LOAD_B     = 4'd1;
localparam S_LOAD_W     = 4'd2;
localparam S_ENG_START  = 4'd3;
localparam S_STREAM     = 4'd4;
localparam S_CAP_BURST  = 4'd5;
localparam S_QUANT_RD   = 4'd6;
localparam S_QUANT_PIPE = 4'd7;
localparam S_QUANT_DRAIN= 4'd8;
localparam S_DONE       = 4'd9;
localparam S_LOAD_W_SETTLE = 4'd10; // Fan-out tree: weight pipeline settle
localparam S_LOAD_B_SETTLE = 4'd11; // Fan-out tree: bias pipeline settle

reg [3:0] state;

wire fifo_pop  = (act_valid && act_ready);

// ============================================================
// 4-Stage Pipelined Quantization (TIMING FIX)
// Stage Q0: Register psum bit-select + M value (breaks BRAM→DSP48 path)
// Stage Q1: 32x32 multiply (registered → DSP48 MREG inference)
// Stage Q2: barrel shift + add Z (registered)
// Stage Q3: clip + ReLU → FIFO write
// ============================================================

// Pre-loaded quantization parameters (filled during S_LOAD_B)
reg signed [31:0] m_local [0:PARALLEL-1];
reg [4:0]         s_local [0:PARALLEL-1];
reg signed [7:0]  z_local [0:PARALLEL-1];

// Stage Q0: Register bit-selected psum and M scale factor
// Moved to bottom logically to prevent used-before-declared warnings
reg signed [31:0] psum_val_r;
reg signed [31:0] m_data_r;
reg [4:0]  s_q0_r;
reg signed [7:0] z_q0_r;
reg        q0_valid;

// Stage Q1: Multiply from registered inputs
(* use_dsp = "yes" *) reg signed [63:0] mul_res_r;
reg [4:0]         s_data_r;
reg signed [7:0]  z_data_r;
reg               q1_valid;  // Pipeline valid for stage Q2
reg [9:0]         q_oc_r;    // Pipeline q_oc for FIFO address tracking

always @(posedge clk) begin
    mul_res_r <= $signed(psum_val_r) * $signed(m_data_r);
    s_data_r  <= s_q0_r;
    z_data_r  <= z_q0_r;
    q1_valid  <= q0_valid;
    q_pix_reg <= q_pix;
end

// Stage Q2: Shift + add Z (registered to break critical path)
// BUGFIX: Verilog rule - if shift amount is unsigned, the whole expression becomes unsigned!
// We must explicitly sign extend the intermediate shift to maintain arithmetic shift.
wire signed [63:0] mul_shifted_31 = mul_res_r >>> 31;
wire signed [31:0] qs_w = $signed(mul_shifted_31 >>> s_data_r) + $signed(z_data_r);

reg signed [31:0] qs_r;        // Registered shift+add result
reg signed [7:0]  z_data_r2;   // Pipeline Z for ReLU compare in Q3
reg               q2_valid;    // Pipeline valid for stage Q3

always @(posedge clk) begin
    qs_r      <= qs_w;
    z_data_r2 <= z_data_r;
    q2_valid  <= q1_valid;
end

// Stage Q3: Clip + ReLU (combinational, written to FIFO)
wire signed [7:0] qc_w_raw = (qs_r > 127) ? 8'sd127 : ((qs_r < -128) ? -8'sd128 : qs_r[7:0]);
// BUGFIX: Quantized ReLU must clip to the zero-point (Z), not 0.
wire signed [7:0] qc_w = (r_relu_en && $signed(qc_w_raw) < $signed(z_data_r2)) ? z_data_r2 : qc_w_raw;

wire fifo_push = q2_valid && !fifo_full;

always @(posedge clk) begin
    if (fifo_push) begin
        fifo_mem[fifo_wr_ptr] <= qc_w;
        // DEBUG
        if (q_oc_r == 0 && out_blk == 0 && fifo_cnt < 10) 
          $display("RTL FIFO PUSH cv3_0: psum_val_r=%d, m_data=%d, shift_31=%d, qs_r=%d, qc_w_raw=%d, qc_w=%d", 
                   psum_val_r, m_data_r, mul_shifted_31, qs_r, qc_w_raw, qc_w);
    end
end


always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fifo_wr_ptr <= 0; fifo_rd_ptr <= 0; fifo_cnt <= 0;
    end else begin
        case ({fifo_push, fifo_pop})
            2'b10: begin fifo_wr_ptr <= fifo_wr_ptr + 1; fifo_cnt <= fifo_cnt + 1; end
            2'b01: begin fifo_rd_ptr <= fifo_rd_ptr + 1; fifo_cnt <= fifo_cnt - 1; end
            2'b11: begin fifo_wr_ptr <= fifo_wr_ptr + 1; fifo_rd_ptr <= fifo_rd_ptr + 1; end
            default: ;
        endcase
    end
end

// ============================================================
// CNN Engine
// ============================================================
reg        eng_start;
wire       eng_done;
reg  [7:0] eng_w_data;
reg        eng_w_wr;
reg  [9:0] eng_w_addr;
reg [31:0] eng_b_data;
reg        eng_b_wr;
reg  [5:0] eng_b_addr;
(* max_fanout = 16 *) reg  [9:0] eng_out_blk_dbg;
(* max_fanout = 16 *) reg  [9:0] eng_in_ch_dbg;

wire signed [32*PARALLEL-1:0] eng_result_bus;
wire                          eng_result_valid;

// Removed duplicate FSM states
reg [9:0]  out_blk;
reg [9:0]  in_ch;
reg [9:0]  w_cnt;
reg [9:0]  b_cnt;
reg [10:0] pix_res;
reg [1:0]  burst_st;
reg [9:0]  q_oc;
reg [2:0]  q_st;
reg        eng_done_latch;
reg [31:0] w_base_addr;
reg [9:0]  active_ocs;
reg [2:0]  pipe_drain;

// Pipeline registers for psum accumulate (breaks ~13ns → ~7ns critical path)
reg [32*PARALLEL-1:0] pipe_eng_data;    // captured engine result
reg [32*PARALLEL-1:0] pipe_psum_rdata;  // captured pre-read psum
reg [10:0]            pipe_addr;        // write-back address
reg                   pipe_first_ch;    // in_ch == 0 flag
reg                   pipe_valid;       // stage 2 active




wire eng_pix_ready;
assign pix_ready = (state == S_STREAM) ? eng_pix_ready : 1'b0;

cnn_engine_dynamic #(
    .MAX_WIDTH   (IMG_W),
    .PARALLEL_OUT(PARALLEL),
    .UNSIGNED_IN (0)
) engine (
    .clk(clk), .rst_n(rst_n),
    .start(eng_start),
    .frame_start(in_ch == 0),
    .padding_val(cfg_padding_val),
    .img_width (cfg_img_w), .img_height(cfg_img_h),
    .in_channels (r_in_ch),
    .out_channels(r_out_ch),
    .stride(2'd1), .kernel_size(r_kernel),
    .unsigned_in_cfg(r_unsigned_in),
    .pixel_in         (pix_in),
    .pixel_valid_fix  (pix_valid && state == S_STREAM),
    .pixel_ready_fix  (eng_pix_ready),
    .eng_ready        (pix_ready),
    .weight_in (eng_w_data), .weight_wr(eng_w_wr), .weight_addr(eng_w_addr),
    .bias_in   (eng_b_data), .bias_wr  (eng_b_wr), .bias_addr  (eng_b_addr),
    .result_bus      (eng_result_bus),
    .result_valid_fix(eng_result_valid),
    .done(eng_done),
    .out_blk_cnt_dbg(eng_out_blk_dbg),
    .in_ch_cnt_dbg  (eng_in_ch_dbg)
);

// ============================================================
// Partial Sum Buffer
// ============================================================
(* ram_style = "block" *) reg [32*PARALLEL-1:0] psum_wide [0:PIX-1];
reg [10:0] psum_wide_waddr;
reg       psum_wide_we;
reg [32*PARALLEL-1:0] psum_wide_wdata, psum_wide_rdata_r;

wire [10:0] psum_wide_raddr_w = (state == S_ENG_START) ? 0 :
                                (state == S_STREAM) ? (eng_result_valid ? (pix_res + 1) : pix_res) :
                                (state == S_QUANT_RD) ? q_pix :
                                (state == S_QUANT_PIPE) ? q_pix :
                                (state == S_QUANT_DRAIN) ? (q_pix + 1) : 0;

always @(posedge clk) begin
    if (psum_wide_we) psum_wide[psum_wide_waddr] <= psum_wide_wdata;
    psum_wide_rdata_r <= psum_wide[psum_wide_raddr_w];
end

// Stage Q0 (Moved here from top): Register bit-selected psum and M scale factor
wire signed [31:0] psum_val_w = $signed(psum_wide_rdata_r[(q_oc*32)+31 -: 32]);
wire q0_feed = (state == S_QUANT_PIPE) && (q_oc < active_ocs) && (fifo_cnt < FIFO_DEPTH - 6);

always @(posedge clk) begin
    if (q0_feed) begin
        psum_val_r <= psum_val_w;
        m_data_r   <= m_local[q_oc];
        s_q0_r     <= s_local[q_oc];
        z_q0_r     <= z_local[q_oc];
    end
    q0_valid <= q0_feed;
end

// result_latch removed (was unused dead code)

// FSM States moved up
// ============================================================
// Combinational Address Outputs
// ============================================================
// w_rd_addr: base is pre-computed in registered transition,
// only simple add remains on combinational path
assign w_rd_addr = w_base_addr + w_cnt;

assign bmsz_rd_addr = (state == S_LOAD_B) ? (out_blk * PARALLEL + b_cnt) :
                                            (out_blk * PARALLEL + q_oc);

integer pi;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state      <= S_IDLE; done <= 0;
        eng_start  <= 0; eng_w_wr <= 0; eng_b_wr <= 0;
        psum_wide_we <= 0;
        out_blk    <= 0; in_ch  <= 0; w_cnt  <= 0; b_cnt  <= 0;
        pix_res    <= 0; burst_st <= 0;
        q_pix      <= 0; q_oc   <= 0; q_st   <= 0;
        eng_done_latch <= 0;
        pipe_valid     <= 0;
        r_in_ch <= 0; r_out_ch <= 0; r_kernel <= 3; r_relu_en <= 0;
        r_w_max <= 0; r_n_out_blks <= 0;

    end else begin
        eng_start <= 0; eng_w_wr <= 0; eng_b_wr <= 0; psum_wide_we <= 0; done <= 0;


        case (state)

        S_IDLE: begin
            if (start) begin
                r_in_ch      <= cfg_in_ch;
                r_out_ch     <= cfg_out_ch;
                r_kernel     <= cfg_kernel;
                r_relu_en    <= cfg_relu_en;
                r_unsigned_in<= cfg_unsigned_in;
                r_w_max      <= (cfg_kernel == 3) ? (PARALLEL * 9) : PARALLEL;
                r_n_out_blks <= (cfg_out_ch + PARALLEL - 1) / PARALLEL;
                out_blk <= 0; in_ch <= 0; b_cnt <= 0;
                eng_out_blk_dbg <= 0;
                eng_in_ch_dbg <= 0;
                state <= S_LOAD_B;
            end
        end

        // ---- Load biases ----
        // bmsz_rd_addr = out_blk*PARALLEL + b_cnt (combinational)
        // b_rd_data available same cycle (combinational reads in simulation)
        S_LOAD_B: begin
            if (b_cnt < PARALLEL && (out_blk*PARALLEL + b_cnt) < r_out_ch) begin
                eng_b_wr   <= 1;
                eng_b_addr <= b_cnt[5:0];
                eng_b_data <= b_rd_data;
                m_local[b_cnt] <= m_rd_data;
                s_local[b_cnt] <= s_rd_data[4:0];
                z_local[b_cnt] <= z_rd_data;
                b_cnt <= b_cnt + 1;
            end else begin
                if (out_blk == r_n_out_blks - 1 && (r_out_ch % PARALLEL) != 0)
                    active_ocs <= r_out_ch % PARALLEL;
                else
                    active_ocs <= PARALLEL;
                b_cnt <= 0; w_cnt <= 0;
                eng_out_blk_dbg <= out_blk;
                eng_in_ch_dbg   <= in_ch;
                if (r_kernel == 3)
                    w_base_addr <= out_blk * r_in_ch * (PARALLEL * 9) + in_ch * (PARALLEL * 9);
                else
                    w_base_addr <= out_blk * r_in_ch * PARALLEL + in_ch * PARALLEL;
                state <= S_LOAD_B_SETTLE;
            end
        end

        // ---- Fan-out tree: bias pipeline settle ----
        S_LOAD_B_SETTLE: begin
            state <= S_LOAD_W;
        end

        // ---- Load weights ----
        // w_rd_addr computed combinationally, w_rd_data available same cycle
        S_LOAD_W: begin
            if (w_cnt < r_w_max) begin
                eng_w_wr   <= 1;
                eng_w_addr <= w_cnt[9:0];
                eng_w_data <= w_rd_data;
                w_cnt <= w_cnt + 1;
            end else begin
                w_cnt <= 0;
                state <= S_LOAD_W_SETTLE;
            end
        end

        // ---- Fan-out tree: weight pipeline settle ----
        S_LOAD_W_SETTLE: begin
            state <= S_ENG_START;
        end

        S_ENG_START: begin
            eng_start <= 1;
            pix_res   <= 0;
            state     <= S_STREAM;
        end

        // ---- PIPELINED S_STREAM (2-stage psum accumulate) ----
        // Stage 1: Capture engine result + pre-read psum into pipeline regs
        // Stage 2: Compute 32-bit add, write to psum BRAM
        // Engine outputs every ~10 cycles, so no back-to-back hazard.
        S_STREAM: begin
            // --- Stage 2: Write accumulated result to BRAM ---
            if (pipe_valid) begin
                psum_wide_waddr <= pipe_addr;
                for (pi = 0; pi < PARALLEL; pi = pi + 1) begin
                    psum_wide_wdata[(pi*32)+31 -: 32] <= pipe_first_ch ?
                        pipe_eng_data[(pi*32)+31 -: 32] :
                        ($signed(pipe_psum_rdata[(pi*32)+31 -: 32]) +
                         $signed(pipe_eng_data[(pi*32)+31 -: 32]));
                end
                psum_wide_we <= 1;
                pipe_valid   <= 0;
            end

            // --- Stage 1: Capture engine output ---
            if (eng_result_valid) begin
                pipe_eng_data  <= eng_result_bus;
                pipe_psum_rdata <= psum_wide_rdata_r;
                pipe_addr      <= pix_res;
                pipe_first_ch  <= (in_ch == 0);
                pipe_valid     <= 1;
                pix_res        <= pix_res + 1;

                if (eng_done)
                    eng_done_latch <= 1;
            end
            else if (!pipe_valid) begin
                // Only check done when pipe is drained
                if (eng_done || eng_done_latch) begin
                    eng_done_latch <= 0;
                    if (in_ch < r_in_ch - 1) begin
                        in_ch <= in_ch + 1;
                        eng_in_ch_dbg <= in_ch + 1;
                        w_cnt <= 0;
                        // Pre-compute weight base address for next channel
                        if (r_kernel == 3)
                            w_base_addr <= out_blk * r_in_ch * (PARALLEL * 9) + (in_ch + 1) * (PARALLEL * 9);
                        else
                            w_base_addr <= out_blk * r_in_ch * PARALLEL + (in_ch + 1) * PARALLEL;
                        state <= S_LOAD_W;
                    end else begin
                        in_ch  <= 0;
                        q_pix  <= 0;
                        q_oc   <= 0;
                        q_st   <= 0;
                        state  <= S_QUANT_RD;
                    end
                end
            end
        end

        // S_CAP_BURST eliminated — logic inlined into S_STREAM above

        S_QUANT_RD: begin
            if (fifo_cnt < FIFO_DEPTH - 6) begin
                q_oc <= 0;
                pipe_drain <= 0;
                state <= S_QUANT_PIPE;
            end
        end

        S_QUANT_PIPE: begin
            if (q0_feed) begin
                if (q_oc < active_ocs - 1) begin
                    q_oc <= q_oc + 1;
                end else begin
                    state <= S_QUANT_DRAIN;
                    pipe_drain <= 0;
                end
            end
        end

        S_QUANT_DRAIN: begin
            pipe_drain <= pipe_drain + 1;
            if (pipe_drain == 4) begin
                if (q_pix < PIX - 1) begin
                    q_pix <= q_pix + 1;
                    state <= S_QUANT_RD;
                end else begin
                    q_pix <= 0;
                    if (out_blk == r_n_out_blks - 1) begin
                        done  <= 1;
                        state <= S_IDLE;
                    end else begin
                        out_blk <= out_blk + 1;
                        eng_out_blk_dbg <= out_blk + 1;
                        in_ch   <= 0; b_cnt <= 0;
                        eng_in_ch_dbg <= 0;
                        state   <= S_LOAD_B;
                    end
                end
            end
        end

        S_DONE: begin
            done    <= 1;
            out_blk <= 0; in_ch <= 0;
            state <= S_IDLE;
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
