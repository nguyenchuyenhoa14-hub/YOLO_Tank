/**
 * detect_head_seq.v — Sequential Detect Head (HARDWARE REUSE VERSION)
 *
 * REFACTORED: Uses ONE conv_stage instance, runtime-reconfigured per layer.
 *   cv2 branch: conv0(IN→64,3x3) → conv1(64→64,3x3) → conv2(64→64,1x1)
 *   cv3 branch: conv0(IN→64,3x3) → conv1(64→64,3x3) → conv2(64→1,1x1)
 *
 * Weight Architecture:
 *   Simulation: On-chip ROM arrays loaded via $readmemh
 *   Synthesis:  External DDR memory via input ports (no on-chip weight storage)
 */
`timescale 1ns/1ps

/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off PINCONNECTEMPTY */
module detect_head_seq #(
    parameter GRID_W   = 40,
    parameter GRID_H   = 40,
    parameter IN_CH    = 64,
    parameter MID_CH   = 64,
    parameter BBOX_CH  = 64,
    parameter NUM_CLASS = 1,
    parameter PARALLEL = 16,
    /* verilator lint_off UNUSEDPARAM */
    parameter CONF_THRESH = -40,
    parameter STRIDE   = 8,
    /* verilator lint_on UNUSEDPARAM */
    parameter CV2_0_W = "V:/weights_for_verilog_pe/model_22_cv2_0_0_conv_weight_quantized_pe.mem", parameter CV2_0_B = "V:/weights_for_verilog_pe/model_22_cv2_0_0_conv_bias_quantized.mem", parameter CV2_0_M = "V:/weights_for_verilog_pe/model_22_cv2_0_0_multiplier.mem", parameter CV2_0_S = "V:/weights_for_verilog_pe/model_22_cv2_0_0_shift.mem",
    parameter CV2_1_W = "V:/weights_for_verilog_pe/model_22_cv2_0_1_conv_weight_quantized_pe.mem", parameter CV2_1_B = "V:/weights_for_verilog_pe/model_22_cv2_0_1_conv_bias_quantized.mem", parameter CV2_1_M = "V:/weights_for_verilog_pe/model_22_cv2_0_1_multiplier.mem", parameter CV2_1_S = "V:/weights_for_verilog_pe/model_22_cv2_0_1_shift.mem",
    parameter CV2_2_W = "V:/weights_for_verilog_pe/model_22_cv2_0_2_conv_weight_quantized_pe.mem", parameter CV2_2_B = "V:/weights_for_verilog_pe/model_22_cv2_0_2_conv_bias_quantized.mem", parameter CV2_2_M = "V:/weights_for_verilog_pe/model_22_cv2_0_2_multiplier.mem", parameter CV2_2_S = "V:/weights_for_verilog_pe/model_22_cv2_0_2_shift.mem",
    parameter CV3_0_W = "V:/weights_for_verilog_pe/model_22_cv3_0_0_conv_weight_quantized_pe.mem", parameter CV3_0_B = "V:/weights_for_verilog_pe/model_22_cv3_0_0_conv_bias_quantized.mem", parameter CV3_0_M = "V:/weights_for_verilog_pe/model_22_cv3_0_0_multiplier.mem", parameter CV3_0_S = "V:/weights_for_verilog_pe/model_22_cv3_0_0_shift.mem",
    parameter CV3_1_W = "V:/weights_for_verilog_pe/model_22_cv3_0_1_conv_weight_quantized_pe.mem", parameter CV3_1_B = "V:/weights_for_verilog_pe/model_22_cv3_0_1_conv_bias_quantized.mem", parameter CV3_1_M = "V:/weights_for_verilog_pe/model_22_cv3_0_1_multiplier.mem", parameter CV3_1_S = "V:/weights_for_verilog_pe/model_22_cv3_0_1_shift.mem",
    parameter CV3_2_W = "V:/weights_for_verilog_pe/model_22_cv3_0_2_conv_weight_quantized_pe.mem", parameter CV3_2_B = "V:/weights_for_verilog_pe/model_22_cv3_0_2_conv_bias_quantized.mem", parameter CV3_2_M = "V:/weights_for_verilog_pe/model_22_cv3_0_2_multiplier.mem", parameter CV3_2_S = "V:/weights_for_verilog_pe/model_22_cv3_0_2_shift.mem",
    parameter CV2_0_Z = "", parameter CV2_1_Z = "", parameter CV2_2_Z = "",
    parameter CV3_0_Z = "", parameter CV3_1_Z = "", parameter CV3_2_Z = ""
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  busy,

    // External write port: load input FM
    input  wire        ext_wr_en,
    input  wire [31:0] ext_wr_addr,
    input  wire [7:0]  ext_wr_data,

    // External read port: read bbox/cls results
    input  wire [31:0] ext_rd_addr,
    output wire [7:0]  ext_rd_bbox_data,
    output reg  [7:0]  ext_rd_cls_data,

    // External weight memory interface (from DDR via AXI)
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [7:0]         ext_w_rd_data,
    input  wire signed [31:0] ext_b_rd_data,
    input  wire signed [31:0] ext_m_rd_data,
    input  wire [4:0]         ext_s_rd_data,
    input  wire signed [7:0]  ext_z_rd_data,
    /* verilator lint_on UNUSEDSIGNAL */

    output reg [31:0] perf_counter
);

localparam PIX     = GRID_W * GRID_H;
localparam MAX_CH  = (IN_CH > MID_CH) ? IN_CH : MID_CH;
localparam RAM_SIZE = PIX * MAX_CH;

// ========================================================================
// Data RAMs (BBOX RAM has been eliminated and merged into RAM B)
(* ram_style = "block" *) reg signed [7:0] ram_a [0:RAM_SIZE-1];
(* ram_style = "block" *) reg signed [7:0] ram_b [0:RAM_SIZE-1];
(* ram_style = "block" *) reg signed [7:0] ram_c [0:RAM_SIZE-1];
(* ram_style = "block" *) reg signed [7:0] cls_ram  [0:PIX*NUM_CLASS-1];

// External read ports (Synchronous for BRAM Inference)
always @(posedge clk) begin
    ext_rd_cls_data  <= cls_ram[ext_rd_addr];
end

assign ext_rd_bbox_data = bram_rdata_b;

// ========================================================================
// Main FSM States & Config
// ========================================================================
localparam S_IDLE         = 4'd0;
localparam S_START_STAGE  = 4'd1;
localparam S_FEED         = 4'd2;
localparam S_COLLECT      = 4'd3;
localparam S_NEXT         = 4'd5;
localparam S_DONE         = 4'd6;

reg [3:0] fsm_state;

reg [7:0] delay_cnt;



// ========================================================================
// Data RAM Read — Synchronous BRAM Inference MUX
// ========================================================================
// ========================================================================
wire [31:0] internal_rd_addr = rd_pix * stage_in_ch_cfg[active_stage] + ({22'd0, rd_pass[9:0]} % {22'd0, stage_in_ch_cfg[active_stage]});
/* verilator lint_off UNUSEDSIGNAL */
wire [31:0] rd_addr = (fsm_state == S_IDLE) ? ext_rd_addr : internal_rd_addr;
/* verilator lint_on UNUSEDSIGNAL */
wire rd_pix_oob = (rd_pix >= PIX);

reg signed [7:0] bram_rdata_a, bram_rdata_b, bram_rdata_c; // Renamed from ram_a_dout, etc.
reg rd_pix_oob_d1;
reg [1:0] rd_src_cur_d1;

// --- 250MHz READ FAN-OUT PIPELINE EXTENSION (+3 cycles to match BRAM) ---
(* max_fanout = 16 *) reg [31:0] rd_addr_p1, rd_addr_p2;
reg valid_p1, valid_p2, valid_p3;

// Combinational wire ensuring pixel valid latching aligns EXACTLY with rd_pix fetches
wire valid_cmd = (fsm_state == S_START_STAGE) || 
                 (fsm_state == S_FEED && skid_count < 4 && rd_pass < rd_total_passes);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_p1 <= 0;
        valid_p2 <= 0;
        valid_p3 <= 0;
        rd_addr_p1 <= 0;
        rd_addr_p2 <= 0;
    end else begin
        rd_addr_p1 <= rd_addr;
        rd_addr_p2 <= rd_addr_p1;

        if (fsm_state == S_IDLE || fsm_state == S_COLLECT || fsm_state == S_NEXT || fsm_state == S_DONE) begin
            // ONLY flush the pipeline if we are NOT in active fetch cycles
            valid_p1 <= 0;
            valid_p2 <= 0;
            valid_p3 <= 0;
        end else begin
            // In S_START_STAGE and S_FEED, properly pipeline the fetch flag!
            // Pixel 0 is evaluated in S_START_STAGE, so valid_cmd MUST flow into valid_p1!
            valid_p1 <= valid_cmd; 
            valid_p2 <= valid_p1;
            valid_p3 <= valid_p2; 
        end
    end
end

always @(posedge clk) begin
    rd_pix_oob_d1 <= rd_pix_oob;
    rd_src_cur_d1 <= stage_src[active_stage];
    bram_rdata_a <= ram_a[rd_addr_p2];
    bram_rdata_b <= ram_b[rd_addr_p2];
    bram_rdata_c <= ram_c[rd_addr_p2];
end

// ========================================================================
// Weight Memory
// ========================================================================
/* verilator lint_off UNUSEDSIGNAL */
wire [31:0] conv_w_rd_addr;
wire [9:0]  conv_bmsz_rd_addr;
/* verilator lint_on UNUSEDSIGNAL */

reg [7:0]  w_data_mux;
reg signed [31:0] b_data_mux;
reg signed [31:0] m_data_mux;
reg [4:0]         s_data_mux;
reg signed [7:0]  z_data_mux;

reg [2:0] active_stage;   // 0..5

`ifndef SYNTHESIS
// ---- SIMULATION MODE: On-chip weight ROMs ----
localparam OUT_BLKS_MID = (MID_CH + PARALLEL - 1) / PARALLEL;
localparam OUT_BLKS_BBOX = (BBOX_CH + PARALLEL - 1) / PARALLEL;
localparam OUT_BLKS_CLS = (NUM_CLASS + PARALLEL - 1) / PARALLEL;

localparam W_DEPTH_0 = OUT_BLKS_MID * PARALLEL * IN_CH * 9;
reg signed [7:0]  w_rom_0 [0:W_DEPTH_0-1];
reg signed [31:0] b_rom_0 [0:MID_CH-1];
reg signed [31:0] m_rom_0 [0:MID_CH-1];
reg [4:0]         s_rom_0 [0:MID_CH-1];
reg signed [7:0]  z_rom_0 [0:MID_CH-1];

localparam W_DEPTH_1 = OUT_BLKS_MID * PARALLEL * MID_CH * 9;
reg signed [7:0]  w_rom_1 [0:W_DEPTH_1-1];
reg signed [31:0] b_rom_1 [0:MID_CH-1];
reg signed [31:0] m_rom_1 [0:MID_CH-1];
reg [4:0]         s_rom_1 [0:MID_CH-1];
reg signed [7:0]  z_rom_1 [0:MID_CH-1];

localparam W_DEPTH_2 = OUT_BLKS_BBOX * PARALLEL * MID_CH;
reg signed [7:0]  w_rom_2 [0:W_DEPTH_2-1];
reg signed [31:0] b_rom_2 [0:BBOX_CH-1];
reg signed [31:0] m_rom_2 [0:BBOX_CH-1];
reg [4:0]         s_rom_2 [0:BBOX_CH-1];
reg signed [7:0]  z_rom_2 [0:BBOX_CH-1];

localparam W_DEPTH_3 = OUT_BLKS_MID * PARALLEL * IN_CH * 9;
reg signed [7:0]  w_rom_3 [0:W_DEPTH_3-1];
reg signed [31:0] b_rom_3 [0:MID_CH-1];
reg signed [31:0] m_rom_3 [0:MID_CH-1];
reg [4:0]         s_rom_3 [0:MID_CH-1];
reg signed [7:0]  z_rom_3 [0:MID_CH-1];

localparam W_DEPTH_4 = OUT_BLKS_MID * PARALLEL * MID_CH * 9;
reg signed [7:0]  w_rom_4 [0:W_DEPTH_4-1];
reg signed [31:0] b_rom_4 [0:MID_CH-1];
reg signed [31:0] m_rom_4 [0:MID_CH-1];
reg [4:0]         s_rom_4 [0:MID_CH-1];
reg signed [7:0]  z_rom_4 [0:MID_CH-1];

localparam W_DEPTH_5 = OUT_BLKS_CLS * PARALLEL * MID_CH;
reg signed [7:0]  w_rom_5 [0:W_DEPTH_5-1];
reg signed [31:0] b_rom_5 [0:NUM_CLASS-1];
reg signed [31:0] m_rom_5 [0:NUM_CLASS-1];
reg [4:0]         s_rom_5 [0:NUM_CLASS-1];
reg signed [7:0]  z_rom_5 [0:NUM_CLASS-1];

initial begin : blk_init_weights
    integer zi;
    if (CV2_0_W != "") $readmemh(CV2_0_W, w_rom_0);
    if (CV2_0_B != "") $readmemh(CV2_0_B, b_rom_0);
    if (CV2_0_M != "") $readmemh(CV2_0_M, m_rom_0);
    if (CV2_0_S != "") $readmemh(CV2_0_S, s_rom_0);
    if (CV2_0_Z != "") $readmemh(CV2_0_Z, z_rom_0);
    else for (zi = 0; zi < MID_CH; zi = zi + 1) z_rom_0[zi] = 8'sd0;

    if (CV2_1_W != "") $readmemh(CV2_1_W, w_rom_1);
    if (CV2_1_B != "") $readmemh(CV2_1_B, b_rom_1);
    if (CV2_1_M != "") $readmemh(CV2_1_M, m_rom_1);
    if (CV2_1_S != "") $readmemh(CV2_1_S, s_rom_1);
    if (CV2_1_Z != "") $readmemh(CV2_1_Z, z_rom_1);
    else for (zi = 0; zi < MID_CH; zi = zi + 1) z_rom_1[zi] = 8'sd0;

    if (CV2_2_W != "") $readmemh(CV2_2_W, w_rom_2);
    if (CV2_2_B != "") $readmemh(CV2_2_B, b_rom_2);
    if (CV2_2_M != "") $readmemh(CV2_2_M, m_rom_2);
    if (CV2_2_S != "") $readmemh(CV2_2_S, s_rom_2);
    if (CV2_2_Z != "") $readmemh(CV2_2_Z, z_rom_2);
    else for (zi = 0; zi < BBOX_CH; zi = zi + 1) z_rom_2[zi] = 8'sd0;

    if (CV3_0_W != "") $readmemh(CV3_0_W, w_rom_3);
    if (CV3_0_B != "") $readmemh(CV3_0_B, b_rom_3);
    if (CV3_0_M != "") $readmemh(CV3_0_M, m_rom_3);
    if (CV3_0_S != "") $readmemh(CV3_0_S, s_rom_3);
    if (CV3_0_Z != "") $readmemh(CV3_0_Z, z_rom_3);
    else for (zi = 0; zi < MID_CH; zi = zi + 1) z_rom_3[zi] = 8'sd0;

    if (CV3_1_W != "") $readmemh(CV3_1_W, w_rom_4);
    if (CV3_1_B != "") $readmemh(CV3_1_B, b_rom_4);
    if (CV3_1_M != "") $readmemh(CV3_1_M, m_rom_4);
    if (CV3_1_S != "") $readmemh(CV3_1_S, s_rom_4);
    if (CV3_1_Z != "") $readmemh(CV3_1_Z, z_rom_4);
    else for (zi = 0; zi < MID_CH; zi = zi + 1) z_rom_4[zi] = 8'sd0;

    if (CV3_2_W != "") $readmemh(CV3_2_W, w_rom_5);
    if (CV3_2_B != "") $readmemh(CV3_2_B, b_rom_5);
    if (CV3_2_M != "") $readmemh(CV3_2_M, m_rom_5);
    if (CV3_2_S != "") $readmemh(CV3_2_S, s_rom_5);
    if (CV3_2_Z != "") $readmemh(CV3_2_Z, z_rom_5);
    else for (zi = 0; zi < NUM_CLASS; zi = zi + 1) z_rom_5[zi] = 8'sd0;
end

// Combinational reads for simulation (zero-latency)
always @(*) begin
    case (active_stage)
        3'd0: begin w_data_mux = w_rom_0[conv_w_rd_addr[17:0]]; b_data_mux = b_rom_0[conv_bmsz_rd_addr[5:0]]; m_data_mux = m_rom_0[conv_bmsz_rd_addr[5:0]]; s_data_mux = s_rom_0[conv_bmsz_rd_addr[5:0]]; z_data_mux = z_rom_0[conv_bmsz_rd_addr[5:0]]; end
        3'd1: begin w_data_mux = w_rom_1[conv_w_rd_addr[17:0]]; b_data_mux = b_rom_1[conv_bmsz_rd_addr[5:0]]; m_data_mux = m_rom_1[conv_bmsz_rd_addr[5:0]]; s_data_mux = s_rom_1[conv_bmsz_rd_addr[5:0]]; z_data_mux = z_rom_1[conv_bmsz_rd_addr[5:0]]; end
        3'd2: begin w_data_mux = w_rom_2[conv_w_rd_addr[17:0]]; b_data_mux = b_rom_2[conv_bmsz_rd_addr[5:0]]; m_data_mux = m_rom_2[conv_bmsz_rd_addr[5:0]]; s_data_mux = s_rom_2[conv_bmsz_rd_addr[5:0]]; z_data_mux = z_rom_2[conv_bmsz_rd_addr[5:0]]; end
        3'd3: begin w_data_mux = w_rom_3[conv_w_rd_addr[17:0]]; b_data_mux = b_rom_3[conv_bmsz_rd_addr[5:0]]; m_data_mux = m_rom_3[conv_bmsz_rd_addr[5:0]]; s_data_mux = s_rom_3[conv_bmsz_rd_addr[5:0]]; z_data_mux = z_rom_3[conv_bmsz_rd_addr[5:0]]; end
        3'd4: begin w_data_mux = w_rom_4[conv_w_rd_addr[17:0]]; b_data_mux = b_rom_4[conv_bmsz_rd_addr[5:0]]; m_data_mux = m_rom_4[conv_bmsz_rd_addr[5:0]]; s_data_mux = s_rom_4[conv_bmsz_rd_addr[5:0]]; z_data_mux = z_rom_4[conv_bmsz_rd_addr[5:0]]; end
        3'd5: begin w_data_mux = w_rom_5[conv_w_rd_addr[10:0]]; b_data_mux = b_rom_5[conv_bmsz_rd_addr[6:0]]; m_data_mux = m_rom_5[conv_bmsz_rd_addr[6:0]]; s_data_mux = s_rom_5[conv_bmsz_rd_addr[6:0]]; z_data_mux = z_rom_5[conv_bmsz_rd_addr[6:0]]; end
        default: begin w_data_mux = 0; b_data_mux = 0; m_data_mux = 0; s_data_mux = 0; z_data_mux = 0; end
    endcase
end

`else
// ---- SYNTHESIS MODE ----
// No on-chip weight storage. Weight data comes via ext_*_rd_data input ports.
always @(*) begin
    w_data_mux = ext_w_rd_data;
    b_data_mux = ext_b_rd_data;
    m_data_mux = ext_m_rd_data;
    s_data_mux = ext_s_rd_data;
    z_data_mux = ext_z_rd_data;
end
`endif

// ========================================================================
// Shared Conv Stage Instance (1 only!)
// ========================================================================
reg        conv_start;
wire       conv_done;
reg  [9:0] conv_cfg_in_ch;
reg  [9:0] conv_cfg_out_ch;
reg  [2:0] conv_cfg_kernel;
reg        conv_cfg_relu_en;
reg        conv_cfg_unsigned_in;
reg  [7:0] conv_cfg_padding_val;

wire [7:0] conv_pix_in_bram = rd_pix_oob_d1 ? ((rd_src_cur_d1 == 2'd0) ? 8'h80 : 8'h00) :
                         (rd_src_cur_d1 == 2'd0) ? bram_rdata_a :
                         (rd_src_cur_d1 == 2'd1) ? bram_rdata_b :
                                                   bram_rdata_c;

reg        conv_pix_valid;
wire       conv_pix_ready;

// --- 8-Deep FWFT Synchronous FIFO to absorb 3-cycle Pipelined BRAM Latency ---
reg [7:0] skid_fifo_data [0:7];
reg [2:0] skid_wr_ptr;
reg [2:0] skid_rd_ptr;
reg [3:0] skid_count;

wire skid_full  = (skid_count >= 8);
wire skid_empty = (skid_count == 0);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        skid_wr_ptr <= 0;
        skid_rd_ptr <= 0;
        skid_count  <= 0;
    end else begin
        if (fsm_state == S_START_STAGE) begin
            skid_wr_ptr <= 0;
            skid_rd_ptr <= 0;
            skid_count  <= 0;
        end else begin
            case ({valid_p3, (conv_pix_ready && !skid_empty)})
                2'b10: begin // Write only
                    skid_fifo_data[skid_wr_ptr] <= conv_pix_in_bram;
                    skid_wr_ptr <= skid_wr_ptr + 1;
                    skid_count  <= skid_count + 1;
                end
                2'b01: begin // Read only
                    skid_rd_ptr <= skid_rd_ptr + 1;
                    skid_count  <= skid_count - 1;
                end
                2'b11: begin // Simultaneous Write and Read
                    skid_fifo_data[skid_wr_ptr] <= conv_pix_in_bram;
                    skid_wr_ptr <= skid_wr_ptr + 1;
                    skid_rd_ptr <= skid_rd_ptr + 1;
                end
                default: ; // Do nothing
            endcase
        end
    end
end

wire [7:0] conv_pix_in = skid_fifo_data[skid_rd_ptr];
wire       conv_pix_valid_skid = !skid_empty;
wire [7:0] conv_act_out;
wire       conv_act_valid;
reg        conv_act_ready;

conv_stage #(
    .IMG_W(GRID_W), .IMG_H(GRID_H),
    .MAX_IN_CH(MAX_CH), .MAX_OUT_CH(MAX_CH),
    .PARALLEL(PARALLEL)
) shared_conv (
    .clk(clk), .rst_n(rst_n),
    .start(conv_start), .done(conv_done),
    .cfg_in_ch(conv_cfg_in_ch),
    .cfg_out_ch(conv_cfg_out_ch),
    .cfg_kernel(conv_cfg_kernel),
    .cfg_relu_en(conv_cfg_relu_en),
    .cfg_unsigned_in(conv_cfg_unsigned_in),
    .cfg_padding_val(conv_cfg_padding_val),
    .cfg_img_w(GRID_W[9:0]),
    .cfg_img_h(GRID_H[9:0]),
    .pix_in(conv_pix_in), .pix_valid(conv_pix_valid_skid), .pix_ready(conv_pix_ready),
    .act_out(conv_act_out), .act_valid(conv_act_valid), .act_ready(conv_act_ready),
    .w_rd_addr(conv_w_rd_addr), .w_rd_data(w_data_mux),
    .bmsz_rd_addr(conv_bmsz_rd_addr),
    .b_rd_data(b_data_mux), .m_rd_data(m_data_mux),
    .s_rd_data(s_data_mux), .z_rd_data(z_data_mux),
    /* verilator lint_off PINCONNECTEMPTY */
    .done_tick(), .in_ch_cnt_dbg()
    /* verilator lint_on PINCONNECTEMPTY */
);

// Debug: Trace first 50 pixels consumed by conv_stage for stage 0, IC=0


// Stage configuration
reg [9:0] stage_in_ch_cfg  [0:5];
reg [9:0] stage_out_ch_cfg [0:5];
reg [1:0] stage_src        [0:5];
reg [2:0] stage_dst        [0:5];
reg       stage_is_final_cv2 [0:5];
reg       stage_is_final_cv3 [0:5];
reg [2:0] stage_kernel       [0:5];
reg       stage_relu_en      [0:5];
reg       stage_unsigned_in  [0:5];
reg [7:0] stage_padding_val  [0:5];

// RAM Reader state
reg [31:0] rd_pass;
reg [31:0] rd_pix;
reg [31:0] rd_total_passes;
reg [31:0] rd_pix_max;

// RAM Writer state
reg [31:0] wr_idx;
reg [31:0] wr_blk;
reg [31:0] wr_pix;
reg [31:0] wr_sub;
reg [9:0]  wr_par;
reg [9:0]  wr_out_ch;

// ========================================================================
// Main FSM
// ========================================================================
// (moved up)

initial begin
    // Init Routing Table
    // BBOX Data is written to RAM B. CV3 branch uses C and A to free up B.
    stage_in_ch_cfg[0]  = IN_CH;  stage_out_ch_cfg[0]  = MID_CH;
    stage_src[0] = 0; stage_dst[0] = 1;  // read ram_a, write ram_b (cv2_0)
    stage_is_final_cv2[0] = 0; stage_is_final_cv3[0] = 0;
    stage_kernel[0] = 3; stage_relu_en[0] = 1; stage_unsigned_in[0] = 0; stage_padding_val[0] = 8'h80;

    stage_in_ch_cfg[1]  = MID_CH; stage_out_ch_cfg[1]  = MID_CH;
    stage_src[1] = 1; stage_dst[1] = 2;  // read ram_b, write ram_c (cv2_1)
    stage_is_final_cv2[1] = 0; stage_is_final_cv3[1] = 0;
    stage_kernel[1] = 3; stage_relu_en[1] = 1; stage_unsigned_in[1] = 0; stage_padding_val[1] = 8'h80;

    stage_in_ch_cfg[2]  = MID_CH; stage_out_ch_cfg[2]  = BBOX_CH;
    stage_src[2] = 2; stage_dst[2] = 1;  // read ram_c, write ram_b (cv2_2 -> Bbox is now preserved in B)
    stage_is_final_cv2[2] = 1; stage_is_final_cv3[2] = 0;
    stage_kernel[2] = 1; stage_relu_en[2] = 0; stage_unsigned_in[2] = 0; stage_padding_val[2] = 8'h80;

    stage_in_ch_cfg[3]  = IN_CH;  stage_out_ch_cfg[3]  = MID_CH;
    stage_src[3] = 0; stage_dst[3] = 2;  // read ram_a, write ram_c (cv3_0 -> A is preserved from input)
    stage_is_final_cv2[3] = 0; stage_is_final_cv3[3] = 0;
    stage_kernel[3] = 3; stage_relu_en[3] = 1; stage_unsigned_in[3] = 0; stage_padding_val[3] = 8'h80;

    stage_in_ch_cfg[4]  = MID_CH; stage_out_ch_cfg[4]  = MID_CH;
    stage_src[4] = 2; stage_dst[4] = 0;  // read ram_c, write ram_a (cv3_1 -> A is overwritten by CV3)
    stage_is_final_cv2[4] = 0; stage_is_final_cv3[4] = 0;
    stage_kernel[4] = 3; stage_relu_en[4] = 1; stage_unsigned_in[4] = 0; stage_padding_val[4] = 8'h80;

    stage_in_ch_cfg[5]  = MID_CH; stage_out_ch_cfg[5]  = NUM_CLASS;
    stage_src[5] = 0; stage_dst[5] = 4;  // read ram_a, write cls   (cv3_2 -> outputs to cls_ram)
    stage_is_final_cv2[5] = 0; stage_is_final_cv3[5] = 1;
    stage_kernel[5] = 1; stage_relu_en[5] = 0; stage_unsigned_in[5] = 0; stage_padding_val[5] = 8'h80;
end

// ========================================================================
// Dedicated BRAM Write Ports — Registered Pipeline (TIMING FIX)
// ========================================================================
// Stage 0 (combinational): compute write address and enables
wire conv_wr_en = (fsm_state == S_FEED || fsm_state == S_COLLECT) && conv_act_valid && conv_act_ready;
wire [31:0] wr_addr_ab_comb = wr_pix * wr_out_ch + wr_blk * PARALLEL + wr_sub;
wire [31:0] wr_addr_cls_comb = wr_pix * NUM_CLASS + wr_sub;

wire ram_a_we_comb = (conv_wr_en && !stage_is_final_cv2[active_stage] && !stage_is_final_cv3[active_stage] && stage_dst[active_stage] == 0);
wire ram_b_we_comb = conv_wr_en && !stage_is_final_cv3[active_stage] && stage_dst[active_stage] == 1;
wire ram_c_we_comb = conv_wr_en && !stage_is_final_cv3[active_stage] && stage_dst[active_stage] == 2;
wire cls_ram_we_comb = conv_wr_en && stage_is_final_cv3[active_stage];

// Stage 1 (registered): delay address, data, and enables by 1 cycle
reg [31:0]       wr_addr_ab_r;
reg [31:0]       wr_addr_cls_r;
reg signed [7:0] wr_data_r;
reg              ram_a_we_r, ram_b_we_r, ram_c_we_r, cls_ram_we_r;

// ext_wr pipeline registers
reg              ext_wr_en_r;
reg [31:0]       ext_wr_addr_r;
reg signed [7:0] ext_wr_data_r;

// --- 250MHz FAN-OUT PIPELINE EXTENSION (Delay = +2) ---
(* max_fanout = 16 *) reg [31:0]       wr_addr_ab_p1, wr_addr_ab_p2;
(* max_fanout = 16 *) reg [31:0]       wr_addr_cls_p1, wr_addr_cls_p2;
(* max_fanout = 16 *) reg signed [7:0] wr_data_p1, wr_data_p2;
(* max_fanout = 16 *) reg              ram_a_we_p1, ram_a_we_p2;
(* max_fanout = 16 *) reg              ram_b_we_p1, ram_b_we_p2;
(* max_fanout = 16 *) reg              ram_c_we_p1, ram_c_we_p2;
(* max_fanout = 16 *) reg              cls_ram_we_p1, cls_ram_we_p2;
(* max_fanout = 16 *) reg              ext_wr_en_p1, ext_wr_en_p2;
(* max_fanout = 16 *) reg [31:0]       ext_wr_addr_p1, ext_wr_addr_p2;
(* max_fanout = 16 *) reg signed [7:0] ext_wr_data_p1, ext_wr_data_p2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_addr_ab_r <= 0; wr_addr_cls_r <= 0; wr_data_r <= 0;
        ext_wr_addr_r <= 0; ext_wr_data_r <= 0;
        ram_a_we_r   <= 0; ram_b_we_r   <= 0; ram_c_we_r   <= 0; cls_ram_we_r <= 0; ext_wr_en_r  <= 0;
        
        wr_addr_ab_p1 <= 0; wr_addr_cls_p1 <= 0; wr_data_p1 <= 0;
        ext_wr_addr_p1 <= 0; ext_wr_data_p1 <= 0;
        ram_a_we_p1  <= 0; ram_b_we_p1  <= 0; ram_c_we_p1  <= 0; cls_ram_we_p1 <= 0; ext_wr_en_p1 <= 0;
        
        wr_addr_ab_p2 <= 0; wr_addr_cls_p2 <= 0; wr_data_p2 <= 0;
        ext_wr_addr_p2 <= 0; ext_wr_data_p2 <= 0;
        ram_a_we_p2  <= 0; ram_b_we_p2  <= 0; ram_c_we_p2  <= 0; cls_ram_we_p2 <= 0; ext_wr_en_p2 <= 0;
    end else begin
        wr_addr_ab_r  <= wr_addr_ab_comb;
        wr_addr_cls_r <= wr_addr_cls_comb;
        wr_data_r     <= conv_act_out;
        ram_a_we_r    <= ram_a_we_comb;
        ram_b_we_r    <= ram_b_we_comb;
        ram_c_we_r    <= ram_c_we_comb;
        cls_ram_we_r  <= cls_ram_we_comb;
        ext_wr_en_r   <= ext_wr_en;
        ext_wr_addr_r <= ext_wr_addr;
        ext_wr_data_r <= ext_wr_data;

        wr_addr_ab_p1  <= wr_addr_ab_r;
        wr_addr_cls_p1 <= wr_addr_cls_r;
        wr_data_p1     <= wr_data_r;
        ram_a_we_p1    <= ram_a_we_r;
        ram_b_we_p1    <= ram_b_we_r;
        ram_c_we_p1    <= ram_c_we_r;
        cls_ram_we_p1  <= cls_ram_we_r;
        ext_wr_en_p1   <= ext_wr_en_r;
        ext_wr_addr_p1 <= ext_wr_addr_r;
        ext_wr_data_p1 <= ext_wr_data_r;

        wr_addr_ab_p2  <= wr_addr_ab_p1;
        wr_addr_cls_p2 <= wr_addr_cls_p1;
        wr_data_p2     <= wr_data_p1;
        ram_a_we_p2    <= ram_a_we_p1;
        ram_b_we_p2    <= ram_b_we_p1;
        ram_c_we_p2    <= ram_c_we_p1;
        cls_ram_we_p2  <= cls_ram_we_p1;
        ext_wr_en_p2   <= ext_wr_en_p1;
        ext_wr_addr_p2 <= ext_wr_addr_p1;
        ext_wr_data_p2 <= ext_wr_data_p1;
    end
end

// Stage 2: BRAM writes using registered pipeline values
// RAM A Write MUX
wire final_ram_a_we = ext_wr_en_p2 | ram_a_we_p2;
/* verilator lint_off UNUSEDSIGNAL */
wire [31:0] final_ram_a_addr = ext_wr_en_p2 ? ext_wr_addr_p2 : wr_addr_ab_p2;
/* verilator lint_on UNUSEDSIGNAL */
wire signed [7:0] final_ram_a_data = ext_wr_en_p2 ? ext_wr_data_p2 : wr_data_p2;

always @(posedge clk) begin
    if (final_ram_a_we)
        ram_a[final_ram_a_addr] <= final_ram_a_data;
end

// RAM B Write
always @(posedge clk) begin
    if (ram_b_we_p2)
        ram_b[wr_addr_ab_p2] <= wr_data_p2;
end

// RAM C Write
always @(posedge clk) begin
    if (ram_c_we_p2)
        ram_c[wr_addr_ab_p2] <= wr_data_p2;
end

// CLS RAM Write (final output)
always @(posedge clk) begin
    if (cls_ram_we_p2)
        cls_ram[wr_addr_cls_p2] <= wr_data_p2;
end

// ========================================================================
// FSM Logic
// ========================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fsm_state <= S_IDLE;
        active_stage <= 0;
        done <= 0; busy <= 0;
        conv_start <= 0;
        perf_counter <= 0;
        rd_pass <= 0; rd_pix <= 0; conv_pix_valid <= 0;
        wr_idx <= 0; wr_blk <= 0; wr_pix <= 0; wr_sub <= 0;
        conv_act_ready <= 0;
    end else begin
        conv_start <= 0;
        conv_pix_valid <= 0;
        done <= 0;

        if (busy) perf_counter <= perf_counter + 1;

        case (fsm_state)

        S_IDLE: begin
            if (start) begin
                fsm_state <= S_START_STAGE;
                active_stage <= 0;
                busy <= 1;
                perf_counter <= 0;
            end
        end

        S_START_STAGE: begin
            conv_cfg_in_ch      <= stage_in_ch_cfg[active_stage];
            conv_cfg_out_ch     <= stage_out_ch_cfg[active_stage];
            conv_cfg_kernel     <= stage_kernel[active_stage];
            conv_cfg_relu_en    <= stage_relu_en[active_stage];
            conv_cfg_unsigned_in <= stage_unsigned_in[active_stage];
            conv_cfg_padding_val <= stage_padding_val[active_stage];

            conv_start <= 1;
            rd_pass <= 0;
            rd_pix <= (stage_kernel[active_stage] == 1) ? 32'd1 : 32'd0;
            rd_total_passes <= ((stage_out_ch_cfg[active_stage] + PARALLEL - 1) / PARALLEL) * stage_in_ch_cfg[active_stage];
            if (stage_kernel[active_stage] == 3)
                rd_pix_max <= PIX + GRID_W + 1;
            else
                rd_pix_max <= PIX - 1;
            wr_idx <= 0;
            wr_blk <= 0; wr_pix <= 0; wr_sub <= 0;
            // Fix: handle last block where out_ch < PARALLEL
            if (stage_out_ch_cfg[active_stage] < PARALLEL) begin
                wr_par <= stage_out_ch_cfg[active_stage];
            end else begin
                wr_par <= PARALLEL;
            end
            wr_out_ch <= stage_out_ch_cfg[active_stage];
            conv_act_ready <= 1;
            conv_pix_valid <= 0;  // Not used for data flow anymore (FIFO driven)
            fsm_state <= S_FEED;
        end

        S_FEED: begin
            conv_act_ready <= 1;

            if (skid_count < 4 && rd_pass < rd_total_passes) begin
                if (rd_pix == rd_pix_max) begin
                    rd_pix <= 0;
                    rd_pass <= rd_pass + 1;
                end else begin
                    rd_pix <= rd_pix + 1;
                end
            end

            if (conv_act_valid && conv_act_ready) begin
                wr_idx <= wr_idx + 1;
                if (wr_sub < wr_par - 1) begin
                    wr_sub <= wr_sub + 1;
                end else begin
                    wr_sub <= 0;
                    if (wr_pix < PIX - 1) begin
                        wr_pix <= wr_pix + 1;
                    end else begin
                        wr_pix <= 0;
                        wr_blk <= wr_blk + 1;
                    end
                end
            end

            if (conv_done) begin
                fsm_state <= S_COLLECT;
            end
        end

        S_COLLECT: begin
            conv_act_ready <= 1;
            if (conv_act_valid) begin
                wr_idx <= wr_idx + 1;
                if (wr_sub < wr_par - 1) begin
                    wr_sub <= wr_sub + 1;
                end else begin
                    wr_sub <= 0;
                    if (wr_pix < PIX - 1) begin
                        wr_pix <= wr_pix + 1;
                    end else begin
                        wr_pix <= 0;
                        wr_blk <= wr_blk + 1;
                    end
                end
            end else begin
                conv_act_ready <= 0;
                fsm_state <= S_NEXT;
            end
        end

        S_NEXT: begin
            if (active_stage < 5) begin
                active_stage <= active_stage + 1;
                rd_pix <= 0; // Reset rd_pix so pre-assert BRAM read in S_START_STAGE uses address 0
                fsm_state <= S_START_STAGE;
            end else begin
                fsm_state <= S_DONE;
            end
        end

        S_DONE: begin
            conv_act_ready <= 0;
            done <= 1;
            busy <= 0;
            fsm_state <= S_IDLE;
        end

        default: fsm_state <= S_IDLE;
        endcase
    end
end

endmodule
