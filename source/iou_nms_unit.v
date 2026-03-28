/**
 * iou_nms_unit.v
 * ============================================================
 * Reads sparse P3/P4/P5 bbox outputs from shared RAM, computes
 * the (cx, cy) pixel centroid of each detected object, applies
 * hardware IoU-based NMS, and writes a compact list of
 * surviving detections to the output RAM.
 *
 * Output format — 3 words (32-bit) per detection:
 *   Word 0: [31:16] = cx, [15:0] = cy  (centroid)
 *   Word 1: [31:16] = x1, [15:0] = y1  (top-left)
 *   Word 2: [31:16] = x2, [15:0] = y2  (bottom-right)
 * List ends with sentinel 32'hDEADBEEF.
 *
 * NMS strategy: suppress box with lower confidence when
 * IoU > threshold. Division-free: inter*DEN > union*NUM.
 *
 * 4-STAGE PIPELINED NMS for 100 MHz timing:
 *   ST_NMS_RD:   Read LUTRAM → register coords, conf, valid
 *   ST_NMS_CALC: Compute intersection box (min/max) + widths
 *   ST_NMS_AREA: Area multiplies via DSP48 inference (registered)
 *   ST_NMS_CMP:  Union + threshold compare + suppress
 *
 * Box merge: when IoU > threshold, the surviving box is updated
 *   to the average of the two overlapping boxes.
 * ============================================================
 */
`timescale 1ns/1ps

module iou_nms_unit #(
    parameter MAX_P3_DETS    = 820,
    parameter MAX_P4_DETS    = 100,
    parameter MAX_P5_DETS    = 20,
    parameter MAX_TOTAL      = 64,
    parameter MAX_OUT_DETS   = 20,
    parameter P3_GRID        = 40,
    parameter P4_GRID        = 20,
    parameter P5_GRID        = 10,
    parameter P3_STRIDE      = 8,
    parameter P4_STRIDE      = 16,
    parameter P5_STRIDE      = 32,
    parameter FRAC_BITS      = 12,
    parameter IOU_THRESH_NUM = 9,   // IoU threshold = NUM/DEN
    parameter IOU_THRESH_DEN = 20,  // default: 9/20 = 0.45
    parameter CONF_THRESH    = -35,
    parameter BBOX_PAD       = 0    // expand bbox by N pixels per side
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  busy,

    // P3 detection buffer
    input wire [21:0] p3_base_addr,
    input wire [10:0] p3_count,

    // P4 detection buffer
    input wire [21:0] p4_base_addr,
    input wire [8:0]  p4_count,

    // P5 detection buffer
    input wire [21:0] p5_base_addr,
    input wire [8:0]  p5_count,

    // Output buffer address
    input wire [21:0] out_base_addr,

    // Shared memory bus (read)
    output reg  [21:0] mem_rd_addr,
    output reg         mem_rd_en,
    input  wire [31:0] mem_rd_data,

    // Shared memory bus (write)
    output reg  [21:0] mem_wr_addr,
    output reg         mem_wr_en,
    output reg  [31:0] mem_wr_data,

    // Performance counter
    output reg [31:0] nms_cycle_count
);

// ================================================================
// States — 5-bit encoding (17 states)
// ================================================================
localparam ST_IDLE      = 5'd0;
localparam ST_LOAD_RD   = 5'd1;
localparam ST_LOAD_LAT  = 5'd2;
localparam ST_LOAD_WR   = 5'd3;
localparam ST_NMS_RD    = 5'd4;   // Stage 1: read LUTRAM
localparam ST_NMS_CALC  = 5'd5;   // Stage 2: intersection bounds + widths
localparam ST_NMS_AREA  = 5'd6;   // Stage 3: area multiplies (DSP48)
localparam ST_NMS_CMP   = 5'd7;   // Stage 4: union + threshold + suppress
localparam ST_WRITE     = 5'd8;
localparam ST_WRITE_B1  = 5'd9;
localparam ST_WRITE_B2  = 5'd10;
localparam ST_MARK_END  = 5'd11;
localparam ST_DONE      = 5'd12;
localparam ST_LOAD_CORR = 5'd13;  // Pipeline stage 4: pixel convert + centroid
localparam ST_NMS_MERGE = 5'd14;  // NMS Stage 5: write merge results to LUTRAM
localparam ST_LOAD_LAT2 = 5'd15;  // Pipeline stage 3: apply WNAC correction
localparam ST_LOAD_WNAC = 5'd16;  // Pipeline stage 2: compute corr_shift
localparam ST_LOAD_CORR_A = 5'd17; // Pipeline stage 4a: pixel shift + anchor

reg [4:0] state;

// ================================================================
// Internal buffers (LUTRAM)
// ================================================================
(* ram_style = "distributed" *) reg [9:0]        cx_buf  [0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg [9:0]        cy_buf  [0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg signed [7:0] conf_buf[0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg [9:0]        x1_buf  [0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg [9:0]        y1_buf  [0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg [9:0]        x2_buf  [0:MAX_TOTAL-1];
(* ram_style = "distributed" *) reg [9:0]        y2_buf  [0:MAX_TOTAL-1];
reg [MAX_TOTAL-1:0] valid;

reg [10:0] total_dets;

// ================================================================
// Load-phase registers
// ================================================================
reg        loading_p4;
reg        loading_p5;
reg [21:0] cur_base;
reg [10:0] cur_count;
reg [10:0] load_idx;
reg [2:0]  word_idx;

// 5-word accumulator
reg [15:0] tmp_l, tmp_t, tmp_r, tmp_b;

// WNAC (Weighted Neighbor Anchor Correction) temporaries
reg signed [7:0] nb_conf_s;
reg [7:0] nb_u, best_u;
reg [8:0] sum_u;
reg [15:0] corr_shift;
reg [15:0] adj_l, adj_r;

// Pipeline registers: ST_LOAD_LAT → ST_LOAD_WNAC (raw BRAM latch)
reg [15:0] lat1_l, lat1_r, lat1_t, lat1_b; // raw L/R/T/B
reg [5:0]  lat1_cx_g, lat1_cy_g;            // grid coordinates
reg        lat1_is_p4, lat1_is_p5;          // scale flags
reg signed [7:0] lat1_nb_conf;              // neighbor confidence
reg [7:0]  lat1_best_u;                     // best confidence (unsigned)
reg signed [7:0] lat1_conf;                 // raw detection confidence

// Pipeline registers: ST_LOAD_WNAC → ST_LOAD_LAT2
reg [15:0] lat2_l, lat2_r;           // raw L/R from word accumulator
reg [15:0] lat2_corr;                // computed corr_shift
reg [15:0] lat2_t, lat2_b;           // pass-through T/B
reg [5:0]  lat2_cx_g, lat2_cy_g;     // grid coordinates
reg        lat2_is_p4, lat2_is_p5;   // scale flags
reg signed [7:0] lat2_conf;          // raw confidence

// Pipeline registers between ST_LOAD_LAT and ST_LOAD_CORR
reg [15:0] adj_l_r, adj_r_r;         // registered WNAC-adjusted L/R
reg [15:0] tmp_t_r, tmp_b_r;         // registered T/B (pass-through)
reg [5:0]  cx_g_r, cy_g_r;           // registered grid coordinates
reg        is_p4_r, is_p5_r;         // registered scale flags
reg signed [7:0] conf_raw_r;         // registered raw confidence

// Intermediate centroid computation
reg [5:0]         tmp_cx_g, tmp_cy_g;
reg [15:0] tmp_l_pix, tmp_t_pix, tmp_r_pix, tmp_b_pix;
reg signed [15:0] tmp_anchor_x, tmp_anchor_y;
reg signed [15:0] tmp_box_cx, tmp_box_cy;
reg signed [15:0] tmp_box_x1, tmp_box_y1, tmp_box_x2, tmp_box_y2;

// Pipeline registers: ST_LOAD_CORR_A → ST_LOAD_CORR (Stage B)
reg [15:0] s5a_l_pix, s5a_t_pix, s5a_r_pix, s5a_b_pix;
reg signed [15:0] s5a_anchor_x, s5a_anchor_y;
reg signed [7:0] s5a_conf;

// Registered pipeline outputs from ST_LOAD_LAT → ST_LOAD_WR
reg               ld_pass_r;
reg [9:0]         ld_cx_r, ld_cy_r;
reg signed [7:0]  ld_conf_r;
reg [9:0]         ld_x1_r, ld_y1_r, ld_x2_r, ld_y2_r;

// ================================================================
// NMS registers — 4-stage pipeline
// ================================================================
reg [10:0] nms_i, nms_j;

// Stage 1 (RD) outputs: latched LUTRAM values
reg [9:0]  r_x1_i, r_y1_i, r_x2_i, r_y2_i;
reg [9:0]  r_x1_j, r_y1_j, r_x2_j, r_y2_j;
reg [9:0]  r_cx_i, r_cy_i, r_cx_j, r_cy_j;
reg signed [7:0] r_conf_i, r_conf_j;
reg        r_valid_i, r_valid_j;
reg [10:0] s1_nms_i, s1_nms_j;

// Stage 2 (CALC) outputs: intersection bounds + widths (registered)
reg [9:0]  s2_inter_w, s2_inter_h;
reg [9:0]  s2_w_i, s2_h_i, s2_w_j, s2_h_j;
reg        s2_both_valid;
reg signed [7:0] s2_conf_i, s2_conf_j;
reg [9:0]  s2_cx_i, s2_cy_i, s2_cx_j, s2_cy_j;
reg [9:0]  s2_x1_i, s2_y1_i, s2_x2_i, s2_y2_i;
reg [9:0]  s2_x1_j, s2_y1_j, s2_x2_j, s2_y2_j;
reg [10:0] s2_nms_i, s2_nms_j;

// Stage 3 (AREA) outputs: areas via DSP48 multiply (registered)
(* use_dsp = "yes" *) reg [19:0] s3_inter_area;
(* use_dsp = "yes" *) reg [19:0] s3_area_i;
(* use_dsp = "yes" *) reg [19:0] s3_area_j;
reg        s3_both_valid;
reg signed [7:0] s3_conf_i, s3_conf_j;
reg [9:0]  s3_cx_i, s3_cy_i, s3_cx_j, s3_cy_j;
reg [9:0]  s3_x1_i, s3_y1_i, s3_x2_i, s3_y2_i;
reg [9:0]  s3_x1_j, s3_y1_j, s3_x2_j, s3_y2_j;
reg [10:0] s3_nms_i, s3_nms_j;

// Stage 4 (MERGE) outputs: registered compare + merge results
reg        s4_do_merge;            // IoU exceeded threshold
reg        s4_keep_i;              // 1=keep i, 0=keep j
reg [10:0] s4_nms_i, s4_nms_j;
reg [9:0]  s4_avg_cx, s4_avg_cy;
reg [9:0]  s4_avg_x1, s4_avg_y1, s4_avg_x2, s4_avg_y2;

// ================================================================
// Write registers
// ================================================================
reg [10:0] wr_scan;
reg [5:0] wr_out;

// Loop var for reset
integer gi;

// ================================================================
// FSM
// ================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state     <= ST_IDLE;
        done      <= 0;
        busy      <= 0;
        mem_rd_en <= 0;
        mem_wr_en <= 0;
        total_dets <= 0;
        valid <= 0;
        nms_cycle_count <= 0;
    end else begin
        mem_rd_en <= 0;
        mem_wr_en <= 0;

        // Performance counter
        if (busy) nms_cycle_count <= nms_cycle_count + 1;

        case (state)

        // ----------------------------------------------------------
        ST_IDLE: begin
            done <= 0;
            if (start) begin
                busy       <= 1;
                load_idx   <= 0;
                word_idx   <= 0;
                total_dets <= 0;
                loading_p4 <= 0;
                loading_p5 <= 0;
                cur_base   <= p3_base_addr;
                cur_count  <= p3_count;
                nms_cycle_count <= 0;
                for (gi=0; gi<MAX_TOTAL; gi=gi+1) begin
                    valid[gi] <= 1'b0;
                end
                state <= ST_LOAD_RD;
            end
        end

        // ----------------------------------------------------------
        ST_LOAD_RD: begin
            if (load_idx < cur_count) begin
                mem_rd_addr <= cur_base + (load_idx * 5) + {19'b0, word_idx};
                mem_rd_en   <= 1;
                state <= ST_LOAD_LAT;
            end else begin
                if (!loading_p4 && !loading_p5 && p4_count > 0) begin
                    loading_p4 <= 1;
                    cur_base   <= p4_base_addr;
                    cur_count  <= p4_count;
                    load_idx   <= 0;
                    word_idx   <= 0;
                    state <= ST_LOAD_RD;
                end else if (!loading_p5 && p5_count > 0) begin
                    loading_p4 <= 0;
                    loading_p5 <= 1;
                    cur_base   <= p5_base_addr;
                    cur_count  <= p5_count;
                    load_idx   <= 0;
                    word_idx   <= 0;
                    state <= ST_LOAD_RD;
                end else begin
                    if (total_dets > 1) begin
                        nms_i <= 0;
                        nms_j <= 1;
                        state <= ST_NMS_RD;
                    end else begin
                        state <= (total_dets > 0) ? ST_WRITE : ST_MARK_END;
                        wr_scan <= 0;
                        wr_out  <= 0;
                    end
                end
            end
        end

        // ----------------------------------------------------------
        // Pipeline Stage 1: Latch raw BRAM data — ZERO combinational
        //   Only register mem_rd_data fields. No computation.
        //   Critical path: BRAM read → register ≈ 2.5ns (well within 10ns)
        // ----------------------------------------------------------
        ST_LOAD_LAT: begin
            case (word_idx)
                3'd0: tmp_l <= mem_rd_data[15:0];
                3'd1: tmp_t <= mem_rd_data[15:0];
                3'd2: tmp_r <= mem_rd_data[15:0];
                3'd3: tmp_b <= mem_rd_data[15:0];
                3'd4: begin
                    // Just register raw fields — no computation
                    lat1_l       <= tmp_l;
                    lat1_r       <= tmp_r;
                    lat1_t       <= tmp_t;
                    lat1_b       <= tmp_b;
                    lat1_cx_g    <= mem_rd_data[21:16];
                    lat1_cy_g    <= mem_rd_data[29:24];
                    lat1_nb_conf <= $signed(mem_rd_data[15:8]);
                    lat1_best_u  <= mem_rd_data[7:0];
                    lat1_conf    <= $signed(mem_rd_data[7:0]);
                    lat1_is_p4   <= loading_p4;
                    lat1_is_p5   <= loading_p5;

                    load_idx <= load_idx + 1;
                end
                default: ;
            endcase

            if (word_idx < 4) begin
                word_idx <= word_idx + 1;
                state <= ST_LOAD_RD;
            end else begin
                state <= ST_LOAD_WNAC;  // → pipeline stage 2: compute corr_shift
            end
        end

        // ----------------------------------------------------------
        // Pipeline Stage 2: Compute corr_shift from registered lat1_*
        //   All inputs are registered — no BRAM on critical path.
        //   Critical path: 8-bit add + cross-multiply + clamp ≈ 8 LUT levels
        //   Fits within 10ns budget since BRAM delay is eliminated.
        // ----------------------------------------------------------
        ST_LOAD_WNAC: begin
            nb_conf_s = lat1_nb_conf;
            corr_shift = 16'd0;

            if (nb_conf_s > 0 && !lat1_is_p4 && !lat1_is_p5) begin
                nb_u  = nb_conf_s[7:0];
                best_u = lat1_best_u;
                sum_u = {1'b0, nb_u} + {1'b0, best_u};

                // Division-free ratio estimation via cross-multiply
                if ({nb_u, 2'b00} >= {sum_u, 1'b0} + sum_u)
                    corr_shift = 16'd6144;
                else if ({nb_u, 1'b0} >= sum_u)
                    corr_shift = 16'd4096;
                else if ({nb_u, 2'b00} >= sum_u)
                    corr_shift = 16'd2048;
                else
                    corr_shift = 16'd1024;

                if (corr_shift > lat1_r)
                    corr_shift = lat1_r;
            end

            // Register for Stage 3
            lat2_l     <= lat1_l;
            lat2_r     <= lat1_r;
            lat2_corr  <= corr_shift;
            lat2_t     <= lat1_t;
            lat2_b     <= lat1_b;
            lat2_cx_g  <= lat1_cx_g;
            lat2_cy_g  <= lat1_cy_g;
            lat2_is_p4 <= lat1_is_p4;
            lat2_is_p5 <= lat1_is_p5;
            lat2_conf  <= lat1_conf;
            state <= ST_LOAD_LAT2;  // → pipeline stage 3: apply correction
        end

        // ----------------------------------------------------------
        // Pipeline Stage 3: Apply WNAC correction (registered)
        //   Uses lat2_* from ST_LOAD_WNAC. Only simple add/sub here.
        //   Critical path: 16-bit add ≈ 4 LUT levels (via CARRY4)
        // ----------------------------------------------------------
        ST_LOAD_LAT2: begin
            adj_l_r    <= lat2_l + lat2_corr;
            adj_r_r    <= lat2_r - lat2_corr;
            tmp_t_r    <= lat2_t;
            tmp_b_r    <= lat2_b;
            cx_g_r     <= lat2_cx_g;
            cy_g_r     <= lat2_cy_g;
            is_p4_r    <= lat2_is_p4;
            is_p5_r    <= lat2_is_p5;
            conf_raw_r <= lat2_conf;
            state <= ST_LOAD_CORR_A;  // → pipeline stage 4a: pixel shift + anchor
        end

        // ----------------------------------------------------------
        // Pipeline Stage 4a: Pixel shift + anchor coordinate multiply
        //   Uses registered adj_l_r/adj_r_r/tmp_b_r from ST_LOAD_LAT2
        //   Critical path: multiply (grid * STRIDE) ≈ 5 logic levels
        // ----------------------------------------------------------
        ST_LOAD_CORR_A: begin
            // Pixel conversion (shift only — very fast)
            if (!is_p4_r && !is_p5_r) begin
                // Scale bbox dimensions down by 0.8125 (multiply by 13, shift by 13 instead of 9)
                s5a_l_pix <= ((adj_l_r * 13) + 4096) >> 13;
                s5a_t_pix <= ((tmp_t_r * 13) + 4096) >> 13;
                s5a_r_pix <= ((adj_r_r * 13) + 4096) >> 13;
                s5a_b_pix <= ((tmp_b_r * 13) + 4096) >> 13;
                s5a_anchor_x <= $signed({10'b0, cx_g_r}) * P3_STRIDE + (P3_STRIDE >> 1);
                s5a_anchor_y <= $signed({10'b0, cy_g_r}) * P3_STRIDE + (P3_STRIDE >> 1);
            end else if (is_p4_r && !is_p5_r) begin
                s5a_l_pix <= ((adj_l_r * 13) + 2048) >> 12;
                s5a_t_pix <= ((tmp_t_r * 13) + 2048) >> 12;
                s5a_r_pix <= ((adj_r_r * 13) + 2048) >> 12;
                s5a_b_pix <= ((tmp_b_r * 13) + 2048) >> 12;
                s5a_anchor_x <= $signed({9'b0, cx_g_r}) * P4_STRIDE + (P4_STRIDE >> 1);
                s5a_anchor_y <= $signed({9'b0, cy_g_r}) * P4_STRIDE + (P4_STRIDE >> 1);
            end else begin
                s5a_l_pix <= adj_l_r >> 7;
                s5a_t_pix <= tmp_t_r >> 7;
                s5a_r_pix <= adj_r_r >> 7;
                s5a_b_pix <= tmp_b_r >> 7;
                s5a_anchor_x <= $signed({8'b0, cx_g_r}) * P5_STRIDE + (P5_STRIDE >> 1);
                s5a_anchor_y <= $signed({8'b0, cy_g_r}) * P5_STRIDE + (P5_STRIDE >> 1);
            end
            s5a_conf <= conf_raw_r;
            state <= ST_LOAD_CORR;
        end

        // ----------------------------------------------------------
        // Pipeline Stage 4b: Centroid + clamp + boundary filter
        //   Uses registered s5a_* from ST_LOAD_CORR_A
        //   Critical path: add + compare + clamp ≈ 6 logic levels
        // ----------------------------------------------------------
        ST_LOAD_CORR: begin
            tmp_box_cx = s5a_anchor_x + $signed({1'b0, s5a_r_pix} - {1'b0, s5a_l_pix}) / 2;
            tmp_box_cy = s5a_anchor_y + $signed({1'b0, s5a_b_pix} - {1'b0, s5a_t_pix}) / 2;
            tmp_box_x1 = s5a_anchor_x - $signed({1'b0, s5a_l_pix}) - BBOX_PAD;
            tmp_box_y1 = s5a_anchor_y - $signed({1'b0, s5a_t_pix}) - BBOX_PAD;
            tmp_box_x2 = s5a_anchor_x + $signed({1'b0, s5a_r_pix}) + BBOX_PAD;
            tmp_box_y2 = s5a_anchor_y + $signed({1'b0, s5a_b_pix}) + BBOX_PAD;

            // Boundary filter + clamp + register
            ld_pass_r <= (s5a_conf > CONF_THRESH &&
                          tmp_box_cx >= 8 && tmp_box_cx <= 312 &&
                          tmp_box_cy >= 8 && tmp_box_cy <= 312);
            ld_cx_r   <= (tmp_box_cx < 0) ? 10'd0 : (tmp_box_cx > 415) ? 10'd415 : tmp_box_cx[9:0];
            ld_cy_r   <= (tmp_box_cy < 0) ? 10'd0 : (tmp_box_cy > 415) ? 10'd415 : tmp_box_cy[9:0];
            ld_conf_r <= s5a_conf;
            ld_x1_r   <= (tmp_box_x1 < 0) ? 10'd0 : (tmp_box_x1 > 320) ? 10'd320 : tmp_box_x1[9:0];
            ld_y1_r   <= (tmp_box_y1 < 0) ? 10'd0 : (tmp_box_y1 > 320) ? 10'd320 : tmp_box_y1[9:0];
            ld_x2_r   <= (tmp_box_x2 < 0) ? 10'd0 : (tmp_box_x2 > 320) ? 10'd320 : tmp_box_x2[9:0];
            ld_y2_r   <= (tmp_box_y2 < 0) ? 10'd0 : (tmp_box_y2 > 320) ? 10'd320 : tmp_box_y2[9:0];

            state <= ST_LOAD_WR;
        end

        // ----------------------------------------------------------
        ST_LOAD_WR: begin
            if (ld_pass_r && total_dets < MAX_TOTAL) begin  // Guard against LUTRAM overflow
                cx_buf[total_dets]   <= ld_cx_r;
                cy_buf[total_dets]   <= ld_cy_r;
                conf_buf[total_dets] <= ld_conf_r;
                x1_buf[total_dets]   <= ld_x1_r;
                y1_buf[total_dets]   <= ld_y1_r;
                x2_buf[total_dets]   <= ld_x2_r;
                y2_buf[total_dets]   <= ld_y2_r;
                valid[total_dets]    <= 1'b1;
                total_dets <= total_dets + 1;
            end
            word_idx <= 0;
            state <= ST_LOAD_RD;
        end

        // ==========================================================
        // 4-STAGE PIPELINED IoU NMS
        //   Stage 1 (RD):   Read LUTRAM → register all coords
        //   Stage 2 (CALC): Intersection bounds + box widths/heights
        //   Stage 3 (AREA): Multiply areas (DSP48 inference)
        //   Stage 4 (CMP):  Union + threshold compare + suppress
        // ==========================================================

        // ----------------------------------------------------------
        // Stage 1: Read LUTRAM into pipeline registers
        // ----------------------------------------------------------
        ST_NMS_RD: begin
            if (nms_i < total_dets - 1) begin
                // Latch all coords from LUTRAM
                r_x1_i <= x1_buf[nms_i]; r_y1_i <= y1_buf[nms_i];
                r_x2_i <= x2_buf[nms_i]; r_y2_i <= y2_buf[nms_i];
                r_x1_j <= x1_buf[nms_j]; r_y1_j <= y1_buf[nms_j];
                r_x2_j <= x2_buf[nms_j]; r_y2_j <= y2_buf[nms_j];
                r_cx_i <= cx_buf[nms_i];  r_cy_i <= cy_buf[nms_i];
                r_cx_j <= cx_buf[nms_j];  r_cy_j <= cy_buf[nms_j];
                r_conf_i <= conf_buf[nms_i];
                r_conf_j <= conf_buf[nms_j];
                r_valid_i <= valid[nms_i];
                r_valid_j <= valid[nms_j];
                s1_nms_i <= nms_i;
                s1_nms_j <= nms_j;
                state <= ST_NMS_CALC;
            end else begin
                // NMS complete
                wr_scan <= 0;
                wr_out  <= 0;
                state   <= ST_WRITE;
            end
        end

        // ----------------------------------------------------------
        // Stage 2: Compute intersection box boundaries + widths
        //   All reads from registered values — no LUTRAM access
        //   Critical path: 10-bit comparator + 10-bit subtract ≈ 6 LUT levels
        // ----------------------------------------------------------
        ST_NMS_CALC: begin
            begin : blk_calc
                reg [9:0] ix1, iy1, ix2, iy2;
                // Intersection rectangle
                ix1 = (r_x1_i > r_x1_j) ? r_x1_i : r_x1_j;
                iy1 = (r_y1_i > r_y1_j) ? r_y1_i : r_y1_j;
                ix2 = (r_x2_i < r_x2_j) ? r_x2_i : r_x2_j;
                iy2 = (r_y2_i < r_y2_j) ? r_y2_i : r_y2_j;
                // Intersection width/height (clamp to 0 if no overlap)
                s2_inter_w <= (ix2 > ix1) ? (ix2 - ix1) : 10'd0;
                s2_inter_h <= (iy2 > iy1) ? (iy2 - iy1) : 10'd0;
            end
            // Box widths/heights
            s2_w_i <= r_x2_i - r_x1_i;
            s2_h_i <= r_y2_i - r_y1_i;
            s2_w_j <= r_x2_j - r_x1_j;
            s2_h_j <= r_y2_j - r_y1_j;
            // Pass through
            s2_both_valid <= r_valid_i && r_valid_j;
            s2_conf_i <= r_conf_i;  s2_conf_j <= r_conf_j;
            s2_cx_i <= r_cx_i;  s2_cy_i <= r_cy_i;
            s2_cx_j <= r_cx_j;  s2_cy_j <= r_cy_j;
            s2_x1_i <= r_x1_i; s2_y1_i <= r_y1_i;
            s2_x2_i <= r_x2_i; s2_y2_i <= r_y2_i;
            s2_x1_j <= r_x1_j; s2_y1_j <= r_y1_j;
            s2_x2_j <= r_x2_j; s2_y2_j <= r_y2_j;
            s2_nms_i <= s1_nms_i;
            s2_nms_j <= s1_nms_j;
            state <= ST_NMS_AREA;
        end

        // ----------------------------------------------------------
        // Stage 3: Area multiplies (DSP48E1 inference)
        //   Each 10×10 multiply maps to one DSP48 slice
        //   Critical path: 1 DSP48 multiply ≈ 3 logic levels
        // ----------------------------------------------------------
        ST_NMS_AREA: begin
            s3_inter_area <= s2_inter_w * s2_inter_h;
            s3_area_i     <= s2_w_i * s2_h_i;
            s3_area_j     <= s2_w_j * s2_h_j;
            // Pass through
            s3_both_valid <= s2_both_valid;
            s3_conf_i <= s2_conf_i;  s3_conf_j <= s2_conf_j;
            s3_cx_i <= s2_cx_i;  s3_cy_i <= s2_cy_i;
            s3_cx_j <= s2_cx_j;  s3_cy_j <= s2_cy_j;
            s3_x1_i <= s2_x1_i; s3_y1_i <= s2_y1_i;
            s3_x2_i <= s2_x2_i; s3_y2_i <= s2_y2_i;
            s3_x1_j <= s2_x1_j; s3_y1_j <= s2_y1_j;
            s3_x2_j <= s2_x2_j; s3_y2_j <= s2_y2_j;
            s3_nms_i <= s2_nms_i;
            s3_nms_j <= s2_nms_j;
            state <= ST_NMS_CMP;
        end

        // ----------------------------------------------------------
        // Stage 4: Union + threshold compare + suppress/merge
        //   union = area_i + area_j - inter_area   (21-bit add/sub)
        //   inter*DEN > union*NUM  → suppress      (25-bit multiply + compare)
        //   Critical path: 21-bit add + 25-bit multiply + comparator ≈ 8 LUT levels
        //   (fits comfortably in 10ns)
        // ----------------------------------------------------------
        ST_NMS_CMP: begin
            begin : blk_cmp
                reg [20:0] union_a;
                reg [24:0] inter_scaled, union_scaled;
                union_a = s3_area_i + s3_area_j - s3_inter_area;
                inter_scaled = s3_inter_area * IOU_THRESH_DEN;
                union_scaled = union_a * IOU_THRESH_NUM;

                // Register compare result — don't write to LUTRAM yet
                s4_do_merge <= (s3_both_valid && inter_scaled > union_scaled);
                s4_keep_i   <= ($signed(s3_conf_i) >= $signed(s3_conf_j));
            end

            // Pre-compute averaged coordinates (registered)
            s4_avg_cx <= (s3_cx_i + s3_cx_j) >> 1;
            s4_avg_cy <= (s3_cy_i + s3_cy_j) >> 1;
            s4_avg_x1 <= (s3_x1_i + s3_x1_j) >> 1;
            s4_avg_y1 <= (s3_y1_i + s3_y1_j) >> 1;
            s4_avg_x2 <= (s3_x2_i + s3_x2_j) >> 1;
            s4_avg_y2 <= (s3_y2_i + s3_y2_j) >> 1;
            s4_nms_i  <= s3_nms_i;
            s4_nms_j  <= s3_nms_j;

            state <= ST_NMS_MERGE;
        end

        // ----------------------------------------------------------
        // Stage 5: Write merge results to LUTRAM
        //   Uses registered compare + averaged coords from Stage 4
        //   Critical path: simple MUX + LUTRAM write ≈ 2 LUT levels
        // ----------------------------------------------------------
        ST_NMS_MERGE: begin
            if (s4_do_merge) begin
                // Pure suppression: invalidate weaker detection, keep stronger unchanged
                // (No coordinate averaging — preserves original bbox accuracy)
                if (s4_keep_i) begin
                    valid[s4_nms_j]  <= 0;
                end else begin
                    valid[s4_nms_i]  <= 0;
                end
            end

            // Advance NMS indices
            if (nms_j < total_dets - 1) begin
                nms_j <= nms_j + 1;
                state <= ST_NMS_RD;
            end else begin
                if (nms_i < total_dets - 2) begin
                    nms_i <= nms_i + 1;
                    nms_j <= nms_i + 2;
                    state <= ST_NMS_RD;
                end else begin
                    wr_scan <= 0;
                    wr_out  <= 0;
                    state   <= ST_WRITE;
                end
            end
        end

        // ----------------------------------------------------------
        // Write surviving detections: 3 words per detection
        // ----------------------------------------------------------
        ST_WRITE: begin
            if (wr_scan < total_dets && wr_out < MAX_OUT_DETS) begin
                if (valid[wr_scan]) begin
                    mem_wr_addr <= out_base_addr + {16'b0, wr_out} * 3;
                    mem_wr_data <= {6'b0, cx_buf[wr_scan], 6'b0, cy_buf[wr_scan]};
                    mem_wr_en   <= 1;
                    state <= ST_WRITE_B1;
                end else begin
                    wr_scan <= wr_scan + 1;
                end
            end else begin
                state <= ST_MARK_END;
            end
        end

        ST_WRITE_B1: begin
            mem_wr_addr <= out_base_addr + {16'b0, wr_out} * 3 + 1;
            mem_wr_data <= {6'b0, x1_buf[wr_scan], 6'b0, y1_buf[wr_scan]};
            mem_wr_en   <= 1;
            state <= ST_WRITE_B2;
        end

        ST_WRITE_B2: begin
            mem_wr_addr <= out_base_addr + {16'b0, wr_out} * 3 + 2;
            mem_wr_data <= {6'b0, x2_buf[wr_scan], 6'b0, y2_buf[wr_scan]};
            mem_wr_en   <= 1;
            wr_out  <= wr_out + 1;
            wr_scan <= wr_scan + 1;
            state <= ST_WRITE;
        end

        // ----------------------------------------------------------
        ST_MARK_END: begin
            mem_wr_addr <= out_base_addr + {16'b0, wr_out} * 3;
            mem_wr_data <= 32'hDEADBEEF;
            mem_wr_en   <= 1;
            state <= ST_DONE;
        end

        // ----------------------------------------------------------
        ST_DONE: begin
            done <= 1;
            busy <= 0;
            state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
        endcase
    end
end

endmodule
