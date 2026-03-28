/**
 * DFL (Distribution Focal Loss) Accelerator Module
 * Takes an input of 64 channels (4 bbox coordinates * 16 bins).
 *
 * Implements:
 * 1. Softmax on 16 bins (using look-up table for e^x).
 * 2. Summing to find divisor.
 * 3. Division to get probabilities.
 * 4. Multiplication and accumulation with indices (0 to 15) to get final coordinate.
 *
 * Max-finding is pipelined into 3 stages to avoid long combinational path:
 *   Stage 0a: Latch 16 logits + compute 8 pairwise maximums (2 logic levels)
 *   Stage 0b: Reduce 8 → 4 quarter-maximums (2 logic levels, registered)
 *   Stage 0c: Reduce 4 → 1 final max (4 logic levels, registered)
 */
`timescale 1ns/1ps

module dfl_accelerator #(
    parameter DFL_BINS = 16,
    parameter LUT_FILE = "V:/weights_for_verilog_pe/exp_lut.mem"
)(
    input wire clk,
    input wire rst_n,
    
    // Control Interface
    input wire start,
    output reg done,
    output reg busy,
    
    // Flattened logits port (128 bits = 16 bins * 8 bits)
    input wire [(DFL_BINS*8)-1:0] logits_flat,
    
    // Output Coordinate (Q8.8 format or simple integer based on scale)
    output reg signed [15:0] coord_out
);

// ============================================================================
// LUT for e^x (Approximation)
// ============================================================================
reg [15:0] exp_lut [0:255];

integer lut_i;
initial begin
`ifndef SYNTHESIS
    $readmemh(LUT_FILE, exp_lut);
`else
    for (lut_i = 0; lut_i < 256; lut_i = lut_i + 1) exp_lut[lut_i] = lut_i[15:0] | 16'h0001;
`endif
end

// ============================================================================
// Stage Declarations
// ============================================================================

// Stage 0a: Latch logits + pairwise max (8 comparisons)
reg signed [7:0] logit_vals [0:DFL_BINS-1];
reg signed [7:0] max_pair [0:7];
reg s0a_valid;

// Stage 0b: Quarter reduction 8 → 4 (TIMING FIX: split from old 8→1)
reg signed [7:0] max_quarter [0:3];
reg s0b_valid;

// Stage 0c: Final reduction 4 → 1
reg signed [7:0] max_logit;
reg s0c_valid;

// Stage 1: LUT Read with diff = max - logit
reg [15:0] exp_values [0:DFL_BINS-1];
reg s1_valid;

// Stage 2: Adder tree
reg [23:0] sum_tree_l1 [0:7];
reg [23:0] sum_tree_l2 [0:3];
reg [23:0] sum_tree_l3 [0:1];
reg [23:0] exp_sum;
reg s2_valid_l1, s2_valid_l2, s2_valid_l3;
reg s2_valid;

// Stage 3: Division
localparam DIV_IDLE    = 5'd0;
localparam DIV_INIT    = 5'd1;
localparam DIV_COMPUTE = 5'd2;
localparam DIV_WAIT    = 5'd3;
localparam DIV_STORE   = 5'd4;
localparam DIV_NEXT    = 5'd5;
localparam DIV_DONE    = 5'd31;

reg [4:0] div_state;
reg [4:0] current_bin;
reg [15:0] dividend;
reg [23:0] divisor;  
reg [15:0] probs [0:DFL_BINS-1];
reg s3_valid;
reg sdiv_start;
wire sdiv_done;
wire sdiv_busy;
wire [23:0] sdiv_quotient;

seq_divider #(
    .N_WIDTH(24),
    .D_WIDTH(24),
    .Q_WIDTH(24)
) u_sdiv (
    .clk(clk),
    .rst_n(rst_n),
    .start(sdiv_start),
    .done(sdiv_done),
    .busy(sdiv_busy),
    .numerator({dividend, 8'd0}),
    .denominator(divisor),
    .quotient(sdiv_quotient)
);

// Stage 4: MAC
reg [4:0] mac_counter;
reg [31:0] mac_accumulator;
reg s4_computing;

integer idx_s0, idx_s1, idx_s2_1, idx_s2_2, idx_s2_3;

// ============================================================================
// Probs RAM
// ============================================================================
always @(posedge clk) begin
    if (div_state == DIV_STORE) probs[current_bin] <= sdiv_quotient[15:0];
end

// ============================================================================
// GLOBAL FSM
// ============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 0;
        done <= 0;
        coord_out <= 0;

        s0a_valid <= 0;
        s0b_valid <= 0;
        s0c_valid <= 0;
        max_logit <= -128;
        for (idx_s0 = 0; idx_s0 < DFL_BINS; idx_s0 = idx_s0 + 1) logit_vals[idx_s0] <= 0;

        s1_valid <= 0;
        for (idx_s1 = 0; idx_s1 < DFL_BINS; idx_s1 = idx_s1 + 1) exp_values[idx_s1] <= 0;

        s2_valid_l1 <= 0;
        s2_valid_l2 <= 0;
        s2_valid_l3 <= 0;
        s2_valid <= 0;
        exp_sum <= 0;

        div_state <= DIV_IDLE;
        current_bin <= 0;
        s3_valid <= 0;
        sdiv_start <= 0;
        dividend <= 0;
        divisor <= 0;

        mac_counter <= 0;
        mac_accumulator <= 0;
        s4_computing <= 0;
    end else begin
        done <= 1'b0;
        if (start) busy <= 1'b1;

        // -------------------------------------------------------------
        // Stage 0a: Latch logits & compute 8 pairwise maximums
        //   Only 2 logic levels (compare + mux per pair)
        //   Original was 27 logic levels for serial 16-value compare!
        // -------------------------------------------------------------
        if (start) begin
            for (idx_s0 = 0; idx_s0 < DFL_BINS; idx_s0 = idx_s0 + 1) begin
                logit_vals[idx_s0] <= $signed(logits_flat[idx_s0*8 +: 8]);
            end
            // 8 parallel pairwise max comparisons
            max_pair[0] <= ($signed(logits_flat[ 0*8 +: 8]) > $signed(logits_flat[ 1*8 +: 8])) ?
                            $signed(logits_flat[ 0*8 +: 8]) : $signed(logits_flat[ 1*8 +: 8]);
            max_pair[1] <= ($signed(logits_flat[ 2*8 +: 8]) > $signed(logits_flat[ 3*8 +: 8])) ?
                            $signed(logits_flat[ 2*8 +: 8]) : $signed(logits_flat[ 3*8 +: 8]);
            max_pair[2] <= ($signed(logits_flat[ 4*8 +: 8]) > $signed(logits_flat[ 5*8 +: 8])) ?
                            $signed(logits_flat[ 4*8 +: 8]) : $signed(logits_flat[ 5*8 +: 8]);
            max_pair[3] <= ($signed(logits_flat[ 6*8 +: 8]) > $signed(logits_flat[ 7*8 +: 8])) ?
                            $signed(logits_flat[ 6*8 +: 8]) : $signed(logits_flat[ 7*8 +: 8]);
            max_pair[4] <= ($signed(logits_flat[ 8*8 +: 8]) > $signed(logits_flat[ 9*8 +: 8])) ?
                            $signed(logits_flat[ 8*8 +: 8]) : $signed(logits_flat[ 9*8 +: 8]);
            max_pair[5] <= ($signed(logits_flat[10*8 +: 8]) > $signed(logits_flat[11*8 +: 8])) ?
                            $signed(logits_flat[10*8 +: 8]) : $signed(logits_flat[11*8 +: 8]);
            max_pair[6] <= ($signed(logits_flat[12*8 +: 8]) > $signed(logits_flat[13*8 +: 8])) ?
                            $signed(logits_flat[12*8 +: 8]) : $signed(logits_flat[13*8 +: 8]);
            max_pair[7] <= ($signed(logits_flat[14*8 +: 8]) > $signed(logits_flat[15*8 +: 8])) ?
                            $signed(logits_flat[14*8 +: 8]) : $signed(logits_flat[15*8 +: 8]);
            s0a_valid <= 1'b1;
        end else if (s0a_valid && s0b_valid) begin
            s0a_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 0b: Reduce 8 → 4 quarter-maximums (TIMING FIX)
        //   Only 1 level of compare+mux = ~2 logic levels
        // -------------------------------------------------------------
        if (s0a_valid && !s0b_valid) begin
            max_quarter[0] <= (max_pair[0] > max_pair[1]) ? max_pair[0] : max_pair[1];
            max_quarter[1] <= (max_pair[2] > max_pair[3]) ? max_pair[2] : max_pair[3];
            max_quarter[2] <= (max_pair[4] > max_pair[5]) ? max_pair[4] : max_pair[5];
            max_quarter[3] <= (max_pair[6] > max_pair[7]) ? max_pair[6] : max_pair[7];
            s0b_valid <= 1'b1;
        end else if (s0b_valid && s0c_valid) begin
            s0b_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 0c: Reduce 4 → 1 final max
        //   2 levels of compare+mux = ~4 logic levels
        // -------------------------------------------------------------
        if (s0b_valid && !s0c_valid) begin
            begin : blk_max_final
                reg signed [7:0] h0, h1;
                h0 = (max_quarter[0] > max_quarter[1]) ? max_quarter[0] : max_quarter[1];
                h1 = (max_quarter[2] > max_quarter[3]) ? max_quarter[2] : max_quarter[3];
                max_logit <= (h0 > h1) ? h0 : h1;
            end
            s0c_valid <= 1'b1;
        end else if (s0c_valid && s1_valid) begin
            s0c_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 1: LUT Read using diff = max_logit - logit[i]
        // -------------------------------------------------------------
        if (s0c_valid && !s1_valid) begin
            for (idx_s1 = 0; idx_s1 < DFL_BINS; idx_s1 = idx_s1 + 1) begin
                exp_values[idx_s1] <= exp_lut[max_logit - logit_vals[idx_s1]];
            end
            s1_valid <= 1'b1;
        end else if (s1_valid && s2_valid_l1) begin
            s1_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 2: Adder Tree
        // -------------------------------------------------------------
        if (s1_valid) begin
            for (idx_s2_1 = 0; idx_s2_1 < 8; idx_s2_1 = idx_s2_1 + 1)
                sum_tree_l1[idx_s2_1] <= exp_values[2*idx_s2_1] + exp_values[2*idx_s2_1+1];
            s2_valid_l1 <= 1'b1;
        end else s2_valid_l1 <= 1'b0;

        if (s2_valid_l1) begin
            for (idx_s2_2 = 0; idx_s2_2 < 4; idx_s2_2 = idx_s2_2 + 1)
                sum_tree_l2[idx_s2_2] <= sum_tree_l1[2*idx_s2_2] + sum_tree_l1[2*idx_s2_2+1];
            s2_valid_l2 <= 1'b1;
        end else s2_valid_l2 <= 1'b0;

        if (s2_valid_l2) begin
            for (idx_s2_3 = 0; idx_s2_3 < 2; idx_s2_3 = idx_s2_3 + 1)
                sum_tree_l3[idx_s2_3] <= sum_tree_l2[2*idx_s2_3] + sum_tree_l2[2*idx_s2_3+1];
            s2_valid_l3 <= 1'b1;
        end else s2_valid_l3 <= 1'b0;

        if (s2_valid_l3 && !s2_valid) begin
            exp_sum <= sum_tree_l3[0] + sum_tree_l3[1];
            s2_valid <= 1'b1;
        end

        if (s2_valid && div_state == DIV_DONE) begin
            s2_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 3: Division Array
        // -------------------------------------------------------------
        case (div_state)
            DIV_IDLE: begin
                if (s2_valid) begin
                    div_state <= DIV_INIT;
                    current_bin <= 5'b0;
                    s3_valid <= 1'b0;
                end
            end
            DIV_INIT: begin
                dividend <= exp_values[current_bin];
                divisor <= exp_sum;
                div_state <= DIV_COMPUTE;
            end
            DIV_COMPUTE: begin
                sdiv_start <= 1'b1;
                div_state  <= DIV_WAIT;
            end
            DIV_WAIT: begin
                sdiv_start <= 1'b0;
                if (sdiv_done) div_state <= DIV_STORE;
            end
            DIV_STORE: begin
                div_state <= DIV_NEXT;
            end
            DIV_NEXT: begin
                if (current_bin == DFL_BINS - 1) div_state <= DIV_DONE;
                else begin
                    current_bin <= current_bin + 1'b1;
                    div_state <= DIV_INIT;
                end
            end
            DIV_DONE: begin
                s3_valid <= 1'b1;
                div_state <= DIV_IDLE;
            end
            default: div_state <= DIV_IDLE;
        endcase

        if (s3_valid && s4_computing) begin
            s3_valid <= 1'b0;
        end

        // -------------------------------------------------------------
        // Stage 4: MAC (Weighted Sum)
        // -------------------------------------------------------------
        if (s3_valid && !s4_computing) begin
            s4_computing <= 1'b1;
            mac_counter <= 5'b0;
            mac_accumulator <= 32'b0;
        end
        
        if (s4_computing) begin
            mac_accumulator <= mac_accumulator + mac_mult;
            
            if (mac_counter == DFL_BINS - 1) begin
                s4_computing <= 1'b0;
                coord_out <= (mac_accumulator + mac_mult) << 4;
                done <= 1'b1;
                busy <= 1'b0;
            end else begin
                mac_counter <= mac_counter + 1'b1;
            end
        end
    end
end

(* use_dsp = "no" *) wire [31:0] mac_mult = probs[mac_counter] * mac_counter;

endmodule
