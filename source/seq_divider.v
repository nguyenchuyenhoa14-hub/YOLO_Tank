/**
 * seq_divider.v — Sequential Restoring Divider
 *
 * Computes: quotient = numerator / denominator
 * Algorithm: Shift-and-subtract (restoring division), 1 bit per clock cycle.
 * Latency: Q_WIDTH clock cycles after start.
 *
 * Interface:
 *   start → load operands, begin division
 *   done  → pulses high for 1 cycle when result is ready
 *   quotient → valid when done=1
 *
 * Parameters:
 *   N_WIDTH — bit width of numerator   (default 32)
 *   D_WIDTH — bit width of denominator (default 24)
 *   Q_WIDTH — bit width of quotient    (default 16)
 */
`timescale 1ns/1ps

module seq_divider #(
    parameter N_WIDTH = 32,
    parameter D_WIDTH = 24,
    parameter Q_WIDTH = 16
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    output reg                   done,
    output reg                   busy,
    input  wire [N_WIDTH-1:0]    numerator,
    input  wire [D_WIDTH-1:0]    denominator,
    output reg  [Q_WIDTH-1:0]    quotient
);

// Internal width for the remainder register: must hold D_WIDTH+1 bits
// to detect underflow (sign bit).
localparam R_WIDTH = D_WIDTH + 1;

reg [N_WIDTH-1:0]   num_reg;      // Shift register for numerator bits
reg [R_WIDTH-1:0]   rem_reg;      // Partial remainder
reg [D_WIDTH-1:0]   den_reg;      // Latched denominator
reg [Q_WIDTH-1:0]   q_reg;        // Quotient being built
reg [5:0]           bit_cnt;      // Counts down from Q_WIDTH-1 to 0

localparam S_IDLE    = 2'd0;
localparam S_COMPUTE = 2'd1;
localparam S_DONE    = 2'd2;

reg [1:0] state;

// Trial subtraction
wire [R_WIDTH-1:0] rem_shifted = {rem_reg[R_WIDTH-2:0], num_reg[N_WIDTH-1]};
wire [R_WIDTH-1:0] rem_trial   = rem_shifted - {1'b0, den_reg};

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state    <= S_IDLE;
        done     <= 1'b0;
        busy     <= 1'b0;
        quotient <= {Q_WIDTH{1'b0}};
        num_reg  <= {N_WIDTH{1'b0}};
        rem_reg  <= {R_WIDTH{1'b0}};
        den_reg  <= {D_WIDTH{1'b0}};
        q_reg    <= {Q_WIDTH{1'b0}};
        bit_cnt  <= 6'd0;
    end else begin
        done <= 1'b0;  // default: clear done pulse

        case (state)
        S_IDLE: begin
            if (start) begin
                // Latch operands
                num_reg <= numerator;
                den_reg <= denominator;
                rem_reg <= {R_WIDTH{1'b0}};
                q_reg   <= {Q_WIDTH{1'b0}};
                bit_cnt <= Q_WIDTH[5:0] - 6'd1;
                busy    <= 1'b1;
                state   <= S_COMPUTE;
            end
        end

        S_COMPUTE: begin
            // Shift next bit of numerator into remainder and trial-subtract
            if (!rem_trial[R_WIDTH-1]) begin
                // Subtraction successful (remainder >= 0)
                rem_reg <= rem_trial;
                q_reg   <= {q_reg[Q_WIDTH-2:0], 1'b1};
            end else begin
                // Subtraction failed, restore (just use shifted value)
                rem_reg <= rem_shifted;
                q_reg   <= {q_reg[Q_WIDTH-2:0], 1'b0};
            end

            // Shift numerator left by 1
            num_reg <= {num_reg[N_WIDTH-2:0], 1'b0};

            if (bit_cnt == 6'd0) begin
                state <= S_DONE;
            end else begin
                bit_cnt <= bit_cnt - 6'd1;
            end
        end

        S_DONE: begin
            quotient <= q_reg;
            done     <= 1'b1;
            busy     <= 1'b0;
            state    <= S_IDLE;
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
