/**
 * yolov8_top_core.v — Top-Level Core for YOLOv8 Detection Head
 *
 * Instantiates and orchestrates:
 *   1. detect_head_seq  — CNN pipeline (6-stage conv, shared engine)
 *   2. dfl_accelerator  — DFL bbox coordinate decoder (4 coords × 16 bins)
 *   3. iou_nms_unit      — IoU-based Non-Maximum Suppression
 *
 * Pipeline: Conv → DFL Decode → NMS (fully integrated, no PS intervention)
 *
 * After `done`, the NMS output buffer contains centroid list + 0xDEADBEEF sentinel.
 */
`timescale 1ns/1ps

module yolov8_top_core #(
    // Detect Head parameters
    parameter GRID_W    = 40,
    parameter GRID_H    = 40,
    parameter IN_CH     = 64,
    parameter MID_CH    = 64,
    parameter BBOX_CH   = 64,
    parameter NUM_CLASS  = 1,
    parameter PARALLEL   = 16,
    parameter STRIDE     = 8,
    parameter CONF_THRESH_DET = -40,
    parameter CONF_THRESH_DFL = 67,   // DFL decode threshold (signed INT8)

    // DFL parameters
    parameter DFL_BINS   = 16,

    // NMS parameters
    parameter MAX_P3_DETS    = 48,
    parameter MAX_TOTAL      = 64,
    parameter MAX_OUT_DETS   = 20,
    parameter IOU_THRESH_NUM = 9,
    parameter IOU_THRESH_DEN = 20,
    parameter CONF_THRESH_NMS = -35,
    parameter FRAC_BITS      = 12
)(
    input  wire clk,
    input  wire rst_n,

    // ================================================================
    // PS Control Interface
    // ================================================================
    input  wire start,
    output reg  done,
    output reg  busy,
    output wire need_reload,   // Detect head needs PS to reload input FM

    // ================================================================
    // PS → Detect Head: Write Input Feature Map
    // ================================================================
    input  wire        ext_wr_en,
    input  wire [31:0] ext_wr_addr,
    input  wire [7:0]  ext_wr_data,

    // ================================================================
    // PS → Detect Head: External Weight Interface
    // ================================================================
    input  wire [7:0]         ext_w_rd_data,
    input  wire signed [31:0] ext_b_rd_data,
    input  wire signed [31:0] ext_m_rd_data,
    input  wire [4:0]         ext_s_rd_data,
    input  wire signed [7:0]  ext_z_rd_data,

    // ================================================================
    // Shared Memory Interface (DFL write + NMS read/write)
    // ================================================================
    output wire [21:0] nms_mem_wr_addr,
    output wire        nms_mem_wr_en,
    output wire [31:0] nms_mem_wr_data,

    output wire [21:0] nms_mem_rd_addr,
    output wire        nms_mem_rd_en,
    input  wire [31:0] nms_mem_rd_data,

    // NMS configuration (from PS)
    input  wire [10:0] p3_det_count,   // Unused when DFL integrated (overridden)
    input  wire [8:0]  p4_det_count,
    input  wire [21:0] p3_base_addr,
    input  wire [21:0] p4_base_addr,
    input  wire [21:0] nms_out_base_addr,

    // Performance counter outputs
    output wire [31:0] perf_det_cycles,
    output wire [31:0] perf_nms_cycles
);

localparam PIX = GRID_W * GRID_H;

// ================================================================
// Internal Wires
// ================================================================

// Detect Head ↔ Top
wire       det_done;
wire       det_busy;
reg        det_start;

// Detect Head BRAM read ports (driven by DFL FSM or held at 0)
reg  [31:0] det_rd_addr;
wire [7:0]  det_rd_bbox_data;
wire [7:0]  det_rd_cls_data;

// DFL signals
reg         dfl_start_reg;
wire        dfl_done;
wire        dfl_busy;
reg  [127:0] dfl_logits_reg;
wire signed [15:0] dfl_coord;

// NMS signals
reg         nms_start;
wire        nms_done;
wire        nms_busy;
wire [31:0] nms_perf_count;

// NMS internal memory ports
wire [21:0] nms_int_wr_addr;
wire        nms_int_wr_en;
wire [31:0] nms_int_wr_data;
wire [21:0] nms_int_rd_addr;
wire        nms_int_rd_en;

// DFL phase memory write
reg  [21:0] dfl_mem_wr_addr;
reg         dfl_mem_wr_en;
reg  [31:0] dfl_mem_wr_data;

// ================================================================
// Top-level FSM
// ================================================================
localparam T_IDLE       = 4'd0;
localparam T_RUN_DET    = 4'd1;
localparam T_WAIT_DET   = 4'd2;
localparam T_RUN_DFL    = 4'd3;   // Start DFL decode phase
localparam T_WAIT_DFL   = 4'd4;   // Wait for DFL decode complete
localparam T_RUN_NMS    = 4'd5;
localparam T_WAIT_NMS   = 4'd6;
localparam T_DONE       = 4'd7;

reg [3:0] top_state;

wire [31:0] det_perf_counter;

// Performance counters output
assign perf_det_cycles = det_perf_counter;
assign perf_nms_cycles = nms_perf_count;

assign need_reload = 1'b0;

// ================================================================
// Shared Memory MUX: DFL write vs NMS write
// ================================================================
wire dfl_phase = (top_state == T_RUN_DFL || top_state == T_WAIT_DFL);

assign nms_mem_wr_addr = dfl_phase ? dfl_mem_wr_addr : nms_int_wr_addr;
assign nms_mem_wr_en   = dfl_phase ? dfl_mem_wr_en   : nms_int_wr_en;
assign nms_mem_wr_data = dfl_phase ? dfl_mem_wr_data : nms_int_wr_data;

// NMS drives read port (DFL reads from detect_head BRAM, not shared mem)
assign nms_mem_rd_addr = nms_int_rd_addr;
assign nms_mem_rd_en   = nms_int_rd_en;

// ================================================================
// DFL Decode FSM
// ================================================================
localparam DF_IDLE       = 5'd0;
localparam DF_RD_CLS     = 5'd1;   // Issue BRAM read for cls
localparam DF_WAIT_CLS   = 5'd2;   // Wait 1 cycle for BRAM
localparam DF_CHK_CLS    = 5'd3;   // Check threshold
localparam DF_RD_LOGIT   = 5'd4;   // Issue BRAM read for bbox logit
localparam DF_WAIT_LOGIT = 5'd5;   // Wait 1 cycle
localparam DF_LATCH      = 5'd6;   // Store logit byte
localparam DF_START_DFL  = 5'd7;   // Assert dfl_start
localparam DF_WAIT_DFL   = 5'd8;   // Wait for dfl_done
localparam DF_STORE_COORD= 5'd9;   // Store decoded coord
localparam DF_RD_NBR     = 5'd10;  // Read left neighbor cls
localparam DF_WAIT_NBR   = 5'd11;  // Wait for neighbor read
localparam DF_LATCH_NBR  = 5'd12;  // Latch neighbor conf
localparam DF_WRITE      = 5'd13;  // Write words to shared mem
localparam DF_NEXT       = 5'd14;  // Advance to next cell
localparam DF_DONE       = 5'd15;

reg [4:0]  df_state;
reg [10:0] df_cell_idx;        // 0..PIX-1
reg [1:0]  df_coord_idx;       // 0..3
reg [3:0]  df_logit_idx;       // 0..15
reg [10:0] df_det_count;       // detection counter
reg signed [7:0] df_conf;      // current cell confidence
reg signed [15:0] df_coords [0:3];  // decoded L, T, R, B
reg signed [7:0] df_left_conf; // left neighbor confidence
reg [2:0]  df_wr_word;         // word write index (0..4)
reg        df_done_flag;
reg        df_trigger;         // Top FSM → DFL FSM handshake

// DFL FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        df_state      <= DF_IDLE;
        df_cell_idx   <= 0;
        df_coord_idx  <= 0;
        df_logit_idx  <= 0;
        df_det_count  <= 0;
        df_conf       <= 0;
        df_left_conf  <= -128;
        df_wr_word    <= 0;
        df_done_flag  <= 0;
        dfl_start_reg <= 0;
        dfl_logits_reg <= 0;
        dfl_mem_wr_en <= 0;
        dfl_mem_wr_addr <= 0;
        dfl_mem_wr_data <= 0;
        det_rd_addr   <= 0;
    end else begin
        dfl_start_reg <= 0;
        dfl_mem_wr_en <= 0;

        case (df_state)

        DF_IDLE: begin
            df_done_flag <= 0;
            // Self-initialize when Top FSM asserts df_trigger
            if (df_trigger) begin
                df_state     <= DF_RD_CLS;
                df_cell_idx  <= 0;
                df_det_count <= 0;
                df_conf      <= 0;
                df_left_conf <= -128;
            end
        end

        DF_RD_CLS: begin
            // Issue cls read: addr = cell_idx * NUM_CLASS
            det_rd_addr <= df_cell_idx * NUM_CLASS;
            df_state <= DF_WAIT_CLS;
        end

        DF_WAIT_CLS: begin
            // Wait 1 cycle for BRAM registered output
            df_state <= DF_CHK_CLS;
        end

        DF_CHK_CLS: begin
            df_conf <= $signed(det_rd_cls_data);
            if ($signed(det_rd_cls_data) > CONF_THRESH_DFL) begin
                // Detection passes threshold — decode bbox
                df_coord_idx <= 0;
                df_logit_idx <= 0;
                df_state <= DF_RD_LOGIT;
            end else begin
                // Skip this cell
                df_state <= DF_NEXT;
            end
        end

        DF_RD_LOGIT: begin
            // Read bbox logit: addr = cell_idx * BBOX_CH + coord * 16 + logit_idx
            det_rd_addr <= df_cell_idx * BBOX_CH + {7'd0, df_coord_idx} * DFL_BINS + {7'd0, df_logit_idx};
            df_state <= DF_WAIT_LOGIT;
        end

        DF_WAIT_LOGIT: begin
            df_state <= DF_LATCH;
        end

        DF_LATCH: begin
            // Store logit byte into 128-bit register
            dfl_logits_reg[(df_logit_idx*8) +: 8] <= det_rd_bbox_data;
            if (df_logit_idx == 4'd15) begin
                df_logit_idx <= 0;
                df_state <= DF_START_DFL;
            end else begin
                df_logit_idx <= df_logit_idx + 1;
                df_state <= DF_RD_LOGIT;
            end
        end

        DF_START_DFL: begin
            dfl_start_reg <= 1;
            df_state <= DF_WAIT_DFL;
        end

        DF_WAIT_DFL: begin
            if (dfl_done) begin
                df_state <= DF_STORE_COORD;
            end
        end

        DF_STORE_COORD: begin
            df_coords[df_coord_idx] <= dfl_coord;
            if (df_coord_idx == 2'd3) begin
                // All 4 coords decoded — read left neighbor
                df_state <= DF_RD_NBR;
            end else begin
                df_coord_idx <= df_coord_idx + 1;
                df_logit_idx <= 0;
                df_state <= DF_RD_LOGIT;
            end
        end

        DF_RD_NBR: begin
            df_left_conf <= -128; // default: no neighbor
            if ((df_cell_idx % GRID_W) > 0) begin
                det_rd_addr <= (df_cell_idx - 1) * NUM_CLASS;
                df_state <= DF_WAIT_NBR;
            end else begin
                df_wr_word <= 0;
                df_state <= DF_WRITE;
            end
        end

        DF_WAIT_NBR: begin
            df_state <= DF_LATCH_NBR;
        end

        DF_LATCH_NBR: begin
            df_left_conf <= $signed(det_rd_cls_data);
            df_wr_word <= 0;
            df_state <= DF_WRITE;
        end

        DF_WRITE: begin
            // Write 5-word packed detection to shared memory
            dfl_mem_wr_en <= 1;
            dfl_mem_wr_addr <= p3_base_addr + df_det_count * 5 + {19'd0, df_wr_word};
            case (df_wr_word)
                3'd0: dfl_mem_wr_data <= {16'd0, df_coords[0]};         // L
                3'd1: dfl_mem_wr_data <= {16'd0, df_coords[1]};         // T
                3'd2: dfl_mem_wr_data <= {16'd0, df_coords[2]};         // R
                3'd3: dfl_mem_wr_data <= {16'd0, df_coords[3]};         // B
                3'd4: begin
                    // Word 4: [29:24]=grid_y(6b), [21:16]=grid_x(6b), [15:8]=left_conf, [7:0]=conf
                    dfl_mem_wr_data[31:24] <= {2'b0, (df_cell_idx / GRID_W)};
                    dfl_mem_wr_data[23:16] <= {2'b0, (df_cell_idx % GRID_W)};
                    dfl_mem_wr_data[15:8]  <= df_left_conf;
                    dfl_mem_wr_data[7:0]   <= df_conf;
                end
                default: dfl_mem_wr_data <= 32'd0;
            endcase

            if (df_wr_word == 3'd4) begin
                df_det_count <= df_det_count + 1;
                df_state <= DF_NEXT;
            end else begin
                df_wr_word <= df_wr_word + 1;
            end
        end

        DF_NEXT: begin
            if (df_cell_idx >= PIX - 1) begin
                df_done_flag <= 1;
                df_state <= DF_DONE;
            end else begin
                df_cell_idx <= df_cell_idx + 1;
                df_state <= DF_RD_CLS;
            end
        end

        DF_DONE: begin
            // Stay here until top FSM acknowledges
        end

        default: df_state <= DF_IDLE;
        endcase
    end
end

// ================================================================
// Top FSM
// ================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        top_state  <= T_IDLE;
        done       <= 0;
        busy       <= 0;
        det_start  <= 0;
        nms_start  <= 0;
        df_trigger <= 0;
    end else begin
        det_start  <= 0;
        nms_start  <= 0;
        df_trigger <= 0;

        case (top_state)
        T_IDLE: begin
            done <= 0;
            if (start) begin
                busy <= 1;
                det_start <= 1;
                top_state <= T_RUN_DET;
            end
        end

        T_RUN_DET: begin
            top_state <= T_WAIT_DET;
        end

        T_WAIT_DET: begin
            if (det_done) begin
                top_state <= T_RUN_DFL;
            end
        end

        T_RUN_DFL: begin
            // Signal DFL FSM to self-initialize
            df_trigger <= 1;
            top_state <= T_WAIT_DFL;
        end

        T_WAIT_DFL: begin
            if (df_done_flag) begin
                nms_start <= 1;
                top_state <= T_RUN_NMS;
            end
        end

        T_RUN_NMS: begin
            top_state <= T_WAIT_NMS;
        end

        T_WAIT_NMS: begin
            if (nms_done) begin
                top_state <= T_DONE;
            end
        end

        T_DONE: begin
            done <= 1;
            busy <= 0;
            top_state <= T_IDLE;
        end

        default: top_state <= T_IDLE;
        endcase
    end
end

// ================================================================
// Module 1: Detect Head Sequential Pipeline
// ================================================================
(* dont_touch = "true" *) detect_head_seq #(
    .GRID_W(GRID_W), .GRID_H(GRID_H),
    .IN_CH(IN_CH), .MID_CH(MID_CH),
    .BBOX_CH(BBOX_CH), .NUM_CLASS(NUM_CLASS),
    .PARALLEL(PARALLEL),
    .CONF_THRESH(CONF_THRESH_DET),
    .STRIDE(STRIDE),
    .CV2_0_Z("V:/weights_for_verilog_pe/model_22_cv2_0_0_zero_point.mem"),
    .CV2_1_Z("V:/weights_for_verilog_pe/model_22_cv2_0_1_zero_point.mem"),
    .CV2_2_Z("V:/weights_for_verilog_pe/model_22_cv2_0_2_zero_point.mem"),
    .CV3_0_Z("V:/weights_for_verilog_pe/model_22_cv3_0_0_zero_point.mem"),
    .CV3_1_Z("V:/weights_for_verilog_pe/model_22_cv3_0_1_zero_point.mem"),
    .CV3_2_Z("V:/weights_for_verilog_pe/model_22_cv3_0_2_zero_point.mem")
) u_detect_head (
    .clk(clk),
    .rst_n(rst_n),
    .start(det_start),
    .done(det_done),
    .busy(det_busy),

    // External write port (input FM from PS)
    .ext_wr_en(ext_wr_en),
    .ext_wr_addr(ext_wr_addr),
    .ext_wr_data(ext_wr_data),

    // External read port (bbox/cls results — driven by DFL FSM)
    .ext_rd_addr(det_rd_addr),
    .ext_rd_bbox_data(det_rd_bbox_data),
    .ext_rd_cls_data(det_rd_cls_data),

    // External weight interface
    .ext_w_rd_data(ext_w_rd_data),
    .ext_b_rd_data(ext_b_rd_data),
    .ext_m_rd_data(ext_m_rd_data),
    .ext_s_rd_data(ext_s_rd_data),
    .ext_z_rd_data(ext_z_rd_data),

    .perf_counter(det_perf_counter)
);

// ================================================================
// Module 2: DFL Accelerator (Bbox Coordinate Decoder)
// ================================================================
(* dont_touch = "true" *) dfl_accelerator #(
    .DFL_BINS(DFL_BINS),
    .LUT_FILE("V:/exp_lut_p3.mem")
) u_dfl (
    .clk(clk),
    .rst_n(rst_n),
    .start(dfl_start_reg),
    .done(dfl_done),
    .busy(dfl_busy),
    .logits_flat(dfl_logits_reg),
    .coord_out(dfl_coord)
);

// ================================================================
// Module 3: IoU NMS Unit (division-free, 4-stage pipelined)
// ================================================================
(* dont_touch = "true" *) iou_nms_unit #(
    .MAX_P3_DETS(MAX_P3_DETS),
    .MAX_TOTAL(MAX_TOTAL),
    .MAX_OUT_DETS(MAX_OUT_DETS),
    .P3_GRID(GRID_W),
    .P3_STRIDE(STRIDE),
    .FRAC_BITS(FRAC_BITS),
    .IOU_THRESH_NUM(IOU_THRESH_NUM),
    .IOU_THRESH_DEN(IOU_THRESH_DEN),
    .CONF_THRESH(CONF_THRESH_NMS)
) u_nms (
    .clk(clk),
    .rst_n(rst_n),
    .start(nms_start),
    .done(nms_done),
    .busy(nms_busy),

    .p3_base_addr(p3_base_addr),
    .p3_count(df_det_count),   // ← DFL-counted detections (not PS input)
    .p4_base_addr(p4_base_addr),
    .p4_count(p4_det_count),
    .p5_base_addr(22'd0),
    .p5_count(9'd0),
    .out_base_addr(nms_out_base_addr),

    .mem_rd_addr(nms_int_rd_addr),
    .mem_rd_en(nms_int_rd_en),
    .mem_rd_data(nms_mem_rd_data),
    .mem_wr_addr(nms_int_wr_addr),
    .mem_wr_en(nms_int_wr_en),
    .mem_wr_data(nms_int_wr_data),
    .nms_cycle_count(nms_perf_count)
);

endmodule
