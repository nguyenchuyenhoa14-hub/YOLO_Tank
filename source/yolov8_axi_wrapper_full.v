/**
 * yolov8_axi_wrapper_full.v — AXI4 Full Slave Wrapper for YOLOv8 Top Core
 *
 * Replaces the AXI4-Lite wrapper with burst-capable AXI4 Full interface.
 * Supports burst writes for feature map loading (~24x faster than AXI4-Lite)
 * and burst reads for NMS result retrieval.
 *
 * Address Map (16-bit address space):
 *   Region 0: Control Registers  0x0000–0x007F  (single-beat, backward compatible)
 *   Region 1: Feature Map BRAM   0x4000–0x5FFFF (burst write, byte-addressed)
 *   Region 2: NMS Shared Memory  0x8000–0x8FFF  (burst read/write, word-addressed)
 *
 * Control Register Map (identical to AXI4-Lite version):
 *   0x00  CTRL          [W]  bit[0]=start
 *   0x04  STATUS        [R]  bit[0]=done, bit[1]=busy
 *   0x08  FM_ADDR       [W]  feature map write address (manual mode)
 *   0x0C  FM_DATA       [W]  feature map write data (manual mode, auto ext_wr_en)
 *   0x10  W_DATA        [W]  weight data
 *   0x14  B_DATA        [W]  bias data
 *   0x18  M_DATA        [W]  multiplier data
 *   0x1C  S_DATA        [W]  shift data
 *   0x20  Z_DATA        [W]  zero-point data
 *   0x24  NMS_P3_CNT    [W]  P3 detection count
 *   0x28  NMS_P4_CNT    [W]  P4 detection count
 *   0x2C  NMS_P3_BASE   [W]  P3 base address
 *   0x30  NMS_P4_BASE   [W]  P4 base address
 *   0x34  NMS_OUT_BASE  [W]  NMS output base address
 *   0x38  PERF_DET      [R]  detect head cycle count
 *   0x3C  PERF_NMS      [R]  NMS cycle count
 *
 * Feature Map Burst Write:
 *   PS writes bursts to 0x4000 + byte_offset.
 *   Each 32-bit beat contains 4 packed bytes [B3:B2:B1:B0].
 *   Wrapper unpacks into 4 sequential ext_wr_en pulses with auto-increment.
 *   WREADY is deasserted during the 4-cycle unpack drain.
 *
 * NMS Memory Burst Read/Write:
 *   Direct 32-bit word access. Address = 0x8000 + (word_index * 4).
 *   Burst reads stream NMS results. Burst writes load detection data.
 */
`timescale 1ns/1ps

module yolov8_axi_wrapper_full #(
    // AXI Parameters
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 16,  // 64KB address space
    parameter C_S_AXI_ID_WIDTH   = 12,  // must match interconnect (Zynq GP0 = 12)

    // Core parameters (forwarded to yolov8_top_core)
    parameter GRID_W    = 40,
    parameter GRID_H    = 40,
    parameter IN_CH     = 64,
    parameter MID_CH    = 64,
    parameter BBOX_CH   = 64,
    parameter NUM_CLASS  = 1,
    parameter PARALLEL   = 16,
    parameter STRIDE     = 8,
    parameter CONF_THRESH_DET = -40,
    parameter CONF_THRESH_DFL = 67,
    parameter DFL_BINS   = 16,
    parameter MAX_P3_DETS    = 48,
    parameter MAX_TOTAL      = 64,
    parameter MAX_OUT_DETS   = 20,
    parameter CONF_THRESH_NMS = -35,
    parameter FRAC_BITS      = 12,

    // NMS shared memory size (words)
    parameter NMS_MEM_DEPTH  = 1024
)(
    // ================================================================
    // AXI4 Full Slave Interface
    // ================================================================
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,

    // Write Address Channel
    input  wire [C_S_AXI_ID_WIDTH-1:0]       S_AXI_AWID,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  wire [7:0]                        S_AXI_AWLEN,    // burst length - 1
    input  wire [2:0]                        S_AXI_AWSIZE,   // beat size
    input  wire [1:0]                        S_AXI_AWBURST,  // FIXED=00, INCR=01, WRAP=10
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output reg                               S_AXI_AWREADY,

    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]   S_AXI_WSTRB,
    input  wire                              S_AXI_WLAST,
    input  wire                              S_AXI_WVALID,
    output reg                               S_AXI_WREADY,

    // Write Response Channel
    output wire [C_S_AXI_ID_WIDTH-1:0]       S_AXI_BID,
    output reg  [1:0]                        S_AXI_BRESP,
    output reg                               S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,

    // Read Address Channel
    input  wire [C_S_AXI_ID_WIDTH-1:0]       S_AXI_ARID,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  wire [7:0]                        S_AXI_ARLEN,
    input  wire [2:0]                        S_AXI_ARSIZE,
    input  wire [1:0]                        S_AXI_ARBURST,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output reg                               S_AXI_ARREADY,

    // Read Data Channel
    output wire [C_S_AXI_ID_WIDTH-1:0]       S_AXI_RID,
    output reg  [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_RDATA,
    output reg  [1:0]                        S_AXI_RRESP,
    output reg                               S_AXI_RLAST,
    output reg                               S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);

// ================================================================
// AXI ID pass-through: echo request ID back on response
// ================================================================
reg [C_S_AXI_ID_WIDTH-1:0] wr_id_latch;
reg [C_S_AXI_ID_WIDTH-1:0] rd_id_latch;
assign S_AXI_BID = wr_id_latch;
assign S_AXI_RID = rd_id_latch;

// ================================================================
// Internal clock/reset
// ================================================================
wire clk   = S_AXI_ACLK;
wire rst_n = S_AXI_ARESETN;

// ================================================================
// Address region decode
// ================================================================
localparam REGION_CTRL = 2'd0;  // 0x0000–0x3FFF
localparam REGION_FM   = 2'd1;  // 0x4000–0x7FFF
localparam REGION_NMS  = 2'd2;  // 0x8000–0xBFFF

function [1:0] addr_region;
    input [C_S_AXI_ADDR_WIDTH-1:0] addr;
    begin
        if (addr[15:14] == 2'b00)      addr_region = REGION_CTRL;
        else if (addr[15:14] == 2'b01) addr_region = REGION_FM;
        else                           addr_region = REGION_NMS;
    end
endfunction

// ================================================================
// Control registers (same as AXI4-Lite version)
// ================================================================
reg [31:0] reg_fm_addr;
reg [7:0]  reg_fm_data;
reg        reg_fm_wr_en;
reg [7:0]  reg_w_data;
reg signed [31:0] reg_b_data;
reg signed [31:0] reg_m_data;
reg [4:0]  reg_s_data;
reg signed [7:0]  reg_z_data;
reg [10:0] reg_p3_det_count;
reg [8:0]  reg_p4_det_count;
reg [21:0] reg_p3_base_addr;
reg [21:0] reg_p4_base_addr;
reg [21:0] reg_nms_out_base;

reg        core_start;
reg        done_latch;

// ================================================================
// Core output wires
// ================================================================
wire       core_done;
wire       core_busy;
wire       core_need_reload;
wire [31:0] core_perf_det;
wire [31:0] core_perf_nms;

// NMS memory bus (from core)
wire [21:0] nms_core_rd_addr;
wire        nms_core_rd_en;
wire [31:0] nms_core_rd_data;
wire [21:0] nms_core_wr_addr;
wire        nms_core_wr_en;
wire [31:0] nms_core_wr_data;

// ================================================================
// NMS Shared Memory (Muxed Single-Port BRAM)
// ================================================================
localparam NMS_ADDR_W = $clog2(NMS_MEM_DEPTH);

reg [31:0] nms_mem [0:NMS_MEM_DEPTH-1];
reg [31:0] nms_mem_rdata;

// PS-side NMS memory signals (split to avoid multi-driven net)
reg [NMS_ADDR_W-1:0] ps_nms_wr_addr;  // driven by Write FSM
reg [NMS_ADDR_W-1:0] ps_nms_rd_addr;  // driven by Read FSM
reg [31:0]           ps_nms_wdata;
reg                  ps_nms_wen;

// PS-side mux: write address when writing, read address otherwise
wire [NMS_ADDR_W-1:0] ps_nms_addr = ps_nms_wen ? ps_nms_wr_addr : ps_nms_rd_addr;

// Muxed access
wire [NMS_ADDR_W-1:0] nms_addr = core_busy ?
    (nms_core_wr_en ? nms_core_wr_addr[NMS_ADDR_W-1:0] :
                      nms_core_rd_addr[NMS_ADDR_W-1:0]) :
    ps_nms_addr;

wire [31:0] nms_wdata = core_busy ? nms_core_wr_data : ps_nms_wdata;
wire        nms_wen   = core_busy ? nms_core_wr_en   : ps_nms_wen;

always @(posedge clk) begin
    nms_mem_rdata <= nms_mem[nms_addr];
    if (nms_wen)
        nms_mem[nms_addr] <= nms_wdata;
end

assign nms_core_rd_data = nms_mem_rdata;

// ================================================================
// WRITE FSM — Handles burst writes to all 3 regions
// ================================================================
localparam WR_IDLE       = 3'd0;
localparam WR_CTRL_BEAT  = 3'd1;  // single-beat control register write
localparam WR_FM_BEAT    = 3'd2;  // FM burst: accept beat
localparam WR_FM_DRAIN   = 3'd3;  // FM burst: drain 4 bytes from 32-bit word
localparam WR_NMS_BEAT   = 3'd4;  // NMS burst: word write
localparam WR_RESP       = 3'd5;  // issue BRESP

reg [2:0]  wr_state;
reg [1:0]  wr_region;
reg [7:0]  wr_beat_cnt;     // beats remaining in burst
reg [C_S_AXI_ADDR_WIDTH-1:0] wr_addr;  // current address
reg [31:0] wr_data_latch;   // latched WDATA for FM drain

// FM byte drain
reg [1:0]  fm_drain_idx;    // which byte (0-3) we're writing
reg [31:0] fm_base_byte_addr; // byte address for current FM word

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_state     <= WR_IDLE;
        wr_region    <= 0;
        wr_beat_cnt  <= 0;
        wr_addr      <= 0;
        wr_data_latch <= 0;
        fm_drain_idx <= 0;
        fm_base_byte_addr <= 0;

        S_AXI_AWREADY <= 0;
        S_AXI_WREADY  <= 0;
        S_AXI_BVALID  <= 0;
        S_AXI_BRESP   <= 2'b00;

        core_start     <= 0;
        done_latch     <= 0;
        reg_fm_wr_en   <= 0;
        reg_fm_addr    <= 0;
        reg_fm_data    <= 0;
        reg_w_data     <= 0;
        reg_b_data     <= 0;
        reg_m_data     <= 0;
        reg_s_data     <= 0;
        reg_z_data     <= 0;
        reg_p3_det_count  <= 0;
        reg_p4_det_count  <= 0;
        reg_p3_base_addr  <= 0;
        reg_p4_base_addr  <= 0;
        reg_nms_out_base  <= 0;
        ps_nms_wen     <= 0;
        ps_nms_wr_addr <= 0;
        ps_nms_wdata   <= 0;
    end else begin
        // Default: clear pulses
        core_start   <= 0;
        reg_fm_wr_en <= 0;
        ps_nms_wen   <= 0;

        // Sticky done latch
        if (core_done) done_latch <= 1;

        // B channel handshake
        if (S_AXI_BVALID && S_AXI_BREADY)
            S_AXI_BVALID <= 0;

        case (wr_state)
        // ----------------------------------------------------------
        WR_IDLE: begin
            S_AXI_AWREADY <= 1;
            S_AXI_WREADY  <= 0;
            if (S_AXI_AWVALID && S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 0;
                wr_addr       <= S_AXI_AWADDR;
                wr_beat_cnt   <= S_AXI_AWLEN;  // total beats = AWLEN + 1
                wr_region     <= addr_region(S_AXI_AWADDR);
                wr_id_latch   <= S_AXI_AWID;   // latch ID for BRESP

                case (addr_region(S_AXI_AWADDR))
                    REGION_CTRL: begin
                        S_AXI_WREADY <= 1;
                        wr_state <= WR_CTRL_BEAT;
                    end
                    REGION_FM: begin
                        S_AXI_WREADY <= 1;
                        fm_base_byte_addr <= {18'b0, S_AXI_AWADDR[13:0]}; // offset within FM region
                        wr_state <= WR_FM_BEAT;
                    end
                    REGION_NMS: begin
                        S_AXI_WREADY <= 1;
                        wr_state <= WR_NMS_BEAT;
                    end
                    default: wr_state <= WR_RESP;
                endcase
            end
        end

        // ----------------------------------------------------------
        // Control register: single-beat write
        // ----------------------------------------------------------
        WR_CTRL_BEAT: begin
            if (S_AXI_WVALID && S_AXI_WREADY) begin
                S_AXI_WREADY <= 0;

                case (wr_addr[6:2])
                5'h00: begin
                    core_start <= S_AXI_WDATA[0];
                    if (S_AXI_WDATA[0]) done_latch <= 0;
                end
                5'h02: reg_fm_addr   <= S_AXI_WDATA;
                5'h03: begin
                    reg_fm_data  <= S_AXI_WDATA[7:0];
                    reg_fm_wr_en <= 1;
                end
                5'h04: reg_w_data    <= S_AXI_WDATA[7:0];
                5'h05: reg_b_data    <= S_AXI_WDATA;
                5'h06: reg_m_data    <= S_AXI_WDATA;
                5'h07: reg_s_data    <= S_AXI_WDATA[4:0];
                5'h08: reg_z_data    <= S_AXI_WDATA[7:0];
                5'h09: reg_p3_det_count  <= S_AXI_WDATA[10:0];
                5'h0A: reg_p4_det_count  <= S_AXI_WDATA[8:0];
                5'h0B: reg_p3_base_addr  <= S_AXI_WDATA[21:0];
                5'h0C: reg_p4_base_addr  <= S_AXI_WDATA[21:0];
                5'h0D: reg_nms_out_base  <= S_AXI_WDATA[21:0];
                default: ;
                endcase

                wr_state <= WR_RESP;
            end
        end

        // ----------------------------------------------------------
        // Feature Map: burst write — accept 32-bit beat
        // ----------------------------------------------------------
        WR_FM_BEAT: begin
            if (S_AXI_WVALID && S_AXI_WREADY) begin
                S_AXI_WREADY  <= 0;  // pause while we drain 4 bytes
                wr_data_latch <= S_AXI_WDATA;
                fm_drain_idx  <= 2'd0;
                wr_state      <= WR_FM_DRAIN;

                // Write first byte immediately
                reg_fm_addr  <= fm_base_byte_addr;
                reg_fm_data  <= S_AXI_WDATA[7:0];
                reg_fm_wr_en <= 1;
                fm_drain_idx <= 2'd1;
            end
        end

        // Feature Map: drain remaining 3 bytes from latched word
        WR_FM_DRAIN: begin
            case (fm_drain_idx)
                2'd1: begin
                    reg_fm_addr  <= fm_base_byte_addr + 32'd1;
                    reg_fm_data  <= wr_data_latch[15:8];
                    reg_fm_wr_en <= 1;
                    fm_drain_idx <= 2'd2;
                end
                2'd2: begin
                    reg_fm_addr  <= fm_base_byte_addr + 32'd2;
                    reg_fm_data  <= wr_data_latch[23:16];
                    reg_fm_wr_en <= 1;
                    fm_drain_idx <= 2'd3;
                end
                2'd3: begin
                    reg_fm_addr  <= fm_base_byte_addr + 32'd3;
                    reg_fm_data  <= wr_data_latch[31:24];
                    reg_fm_wr_en <= 1;

                    // Advance to next beat
                    fm_base_byte_addr <= fm_base_byte_addr + 32'd4;

                    if (wr_beat_cnt == 0) begin
                        // Last beat done
                        wr_state <= WR_RESP;
                    end else begin
                        wr_beat_cnt  <= wr_beat_cnt - 8'd1;
                        S_AXI_WREADY <= 1;  // ready for next beat
                        wr_state     <= WR_FM_BEAT;
                    end
                end
                default: fm_drain_idx <= fm_drain_idx; // should not happen
            endcase
        end

        // ----------------------------------------------------------
        // NMS Memory: burst write — direct 32-bit word access
        // ----------------------------------------------------------
        WR_NMS_BEAT: begin
            if (S_AXI_WVALID && S_AXI_WREADY) begin
                // Write to NMS memory
                ps_nms_wr_addr  <= wr_addr[NMS_ADDR_W+1:2]; // word address
                ps_nms_wdata <= S_AXI_WDATA;
                ps_nms_wen   <= 1;

                // Advance address (INCR burst)
                wr_addr <= wr_addr + 16'd4;

                if (wr_beat_cnt == 0) begin
                    S_AXI_WREADY <= 0;
                    wr_state <= WR_RESP;
                end else begin
                    wr_beat_cnt <= wr_beat_cnt - 8'd1;
                end
            end
        end

        // ----------------------------------------------------------
        // Write Response
        // ----------------------------------------------------------
        WR_RESP: begin
            S_AXI_BVALID <= 1;
            S_AXI_BRESP  <= 2'b00; // OKAY
            wr_state     <= WR_IDLE;
        end

        default: wr_state <= WR_IDLE;
        endcase
    end
end

// ================================================================
// READ FSM — Handles burst reads from control regs, NMS memory
// ================================================================
localparam RD_IDLE = 2'd0;
localparam RD_DATA = 2'd1;
localparam RD_NMS_WAIT = 2'd2;  // 1-cycle BRAM read latency

reg [1:0]  rd_state;
reg [1:0]  rd_region;
reg [7:0]  rd_beat_cnt;
reg [C_S_AXI_ADDR_WIDTH-1:0] rd_addr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_state     <= RD_IDLE;
        rd_region    <= 0;
        rd_beat_cnt  <= 0;
        rd_addr      <= 0;
        S_AXI_ARREADY <= 0;
        S_AXI_RVALID  <= 0;
        S_AXI_RRESP   <= 2'b00;
        S_AXI_RDATA   <= 0;
        S_AXI_RLAST   <= 0;
    end else begin
        case (rd_state)
        // ----------------------------------------------------------
        RD_IDLE: begin
            S_AXI_ARREADY <= 1;
            S_AXI_RLAST   <= 0;
            if (S_AXI_ARVALID && S_AXI_ARREADY) begin
                S_AXI_ARREADY <= 0;
                rd_addr       <= S_AXI_ARADDR;
                rd_beat_cnt   <= S_AXI_ARLEN;
                rd_region     <= addr_region(S_AXI_ARADDR);
                rd_id_latch   <= S_AXI_ARID;   // latch ID for RDATA

                if (addr_region(S_AXI_ARADDR) == REGION_NMS) begin
                    // NMS: need 1-cycle BRAM read latency
                    ps_nms_rd_addr <= S_AXI_ARADDR[NMS_ADDR_W+1:2];
                    rd_state <= RD_NMS_WAIT;
                end else begin
                    // Control registers: immediate
                    rd_state <= RD_DATA;
                end
            end
        end

        // ----------------------------------------------------------
        // NMS: wait 1 cycle for BRAM read latency
        // ----------------------------------------------------------
        RD_NMS_WAIT: begin
            // Data available on nms_mem_rdata next cycle
            S_AXI_RDATA  <= nms_mem_rdata;
            S_AXI_RVALID <= 1;
            S_AXI_RRESP  <= 2'b00;
            S_AXI_RLAST  <= (rd_beat_cnt == 0);
            rd_state     <= RD_DATA;
        end

        // ----------------------------------------------------------
        // Data beat output
        // ----------------------------------------------------------
        RD_DATA: begin
            if (!S_AXI_RVALID) begin
                // Prepare data based on region
                if (rd_region == REGION_CTRL) begin
                    case (rd_addr[6:2])
                    5'h01: S_AXI_RDATA <= {30'd0, core_busy, done_latch};
                    5'h0E: S_AXI_RDATA <= core_perf_det;
                    5'h0F: S_AXI_RDATA <= core_perf_nms;
                    default: S_AXI_RDATA <= 32'hDEAD_CAFE;
                    endcase
                    S_AXI_RVALID <= 1;
                    S_AXI_RRESP  <= 2'b00;
                    S_AXI_RLAST  <= (rd_beat_cnt == 0);
                end else begin
                    // NMS: issue next BRAM read
                    ps_nms_rd_addr <= rd_addr[NMS_ADDR_W+1:2];
                    rd_state <= RD_NMS_WAIT;
                end
            end

            if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 0;
                if (rd_beat_cnt == 0) begin
                    S_AXI_RLAST <= 0;
                    rd_state <= RD_IDLE;
                end else begin
                    rd_beat_cnt <= rd_beat_cnt - 8'd1;
                    rd_addr     <= rd_addr + 16'd4;

                    if (rd_region == REGION_NMS) begin
                        ps_nms_rd_addr <= (rd_addr + 16'd4) >> 2;
                        rd_state <= RD_NMS_WAIT;
                    end
                    // CTRL burstz: loop back to prepare next data
                end
            end
        end

        default: rd_state <= RD_IDLE;
        endcase
    end
end

// ================================================================
// YOLOv8 Top Core Instance
// ================================================================
yolov8_top_core #(
    .GRID_W(GRID_W), .GRID_H(GRID_H),
    .IN_CH(IN_CH), .MID_CH(MID_CH),
    .BBOX_CH(BBOX_CH), .NUM_CLASS(NUM_CLASS),
    .PARALLEL(PARALLEL), .STRIDE(STRIDE),
    .CONF_THRESH_DET(CONF_THRESH_DET),
    .CONF_THRESH_DFL(CONF_THRESH_DFL),
    .DFL_BINS(DFL_BINS),
    .MAX_P3_DETS(MAX_P3_DETS), .MAX_TOTAL(MAX_TOTAL),
    .MAX_OUT_DETS(MAX_OUT_DETS),
    .CONF_THRESH_NMS(CONF_THRESH_NMS),
    .FRAC_BITS(FRAC_BITS)
) u_core (
    .clk(clk),
    .rst_n(rst_n),

    // Control
    .start(core_start),
    .done(core_done),
    .busy(core_busy),
    .need_reload(core_need_reload),

    // Feature map write
    .ext_wr_en(reg_fm_wr_en),
    .ext_wr_addr(reg_fm_addr),
    .ext_wr_data(reg_fm_data),

    // Weight interface
    .ext_w_rd_data(reg_w_data),
    .ext_b_rd_data(reg_b_data),
    .ext_m_rd_data(reg_m_data),
    .ext_s_rd_data(reg_s_data),
    .ext_z_rd_data(reg_z_data),

    // NMS memory bus
    .nms_mem_wr_addr(nms_core_wr_addr),
    .nms_mem_wr_en(nms_core_wr_en),
    .nms_mem_wr_data(nms_core_wr_data),
    .nms_mem_rd_addr(nms_core_rd_addr),
    .nms_mem_rd_en(nms_core_rd_en),
    .nms_mem_rd_data(nms_core_rd_data),

    // NMS config
    .p3_det_count(reg_p3_det_count),
    .p4_det_count(reg_p4_det_count),
    .p3_base_addr(reg_p3_base_addr),
    .p4_base_addr(reg_p4_base_addr),
    .nms_out_base_addr(reg_nms_out_base),

    // Performance counters
    .perf_det_cycles(core_perf_det),
    .perf_nms_cycles(core_perf_nms)
);

endmodule
