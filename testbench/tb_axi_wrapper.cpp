#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "Vyolov8_axi_wrapper.h"
#include "verilated.h"

// Grid parameters for P3
#define GRID_W 40
#define GRID_H 40
#define IN_CH 64
#define TOTAL_IN (GRID_W * GRID_H * IN_CH)
#define MAX_OUT_DETS 20

// AXI Register offsets (byte addresses)
#define REG_CTRL          0x00
#define REG_STATUS        0x04
#define REG_FM_ADDR       0x08
#define REG_FM_DATA       0x0C
#define REG_W_DATA        0x10
#define REG_B_DATA        0x14
#define REG_M_DATA        0x18
#define REG_S_DATA        0x1C
#define REG_Z_DATA        0x20
#define REG_NMS_P3_CNT    0x24
#define REG_NMS_P4_CNT    0x28
#define REG_NMS_P3_BASE   0x2C
#define REG_NMS_P4_BASE   0x30
#define REG_NMS_OUT_BASE  0x34
#define REG_PERF_DET      0x38
#define REG_PERF_NMS      0x3C
#define REG_NMS_MEM_ADDR  0x40
#define REG_NMS_MEM_WDATA 0x44
#define REG_NMS_MEM_RDATA 0x48
#define REG_NMS_MEM_WEN   0x4C

uint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

// Clock tick helper
static void tick(Vyolov8_axi_wrapper* top) {
    top->S_AXI_ACLK = 1; top->eval(); main_time += 5;
    top->S_AXI_ACLK = 0; top->eval(); main_time += 5;
}

// AXI4-Lite WRITE transaction
static void axi_write(Vyolov8_axi_wrapper* top, uint32_t addr, uint32_t data) {
    // Drive AW + W channels simultaneously
    top->S_AXI_AWADDR  = addr;
    top->S_AXI_AWVALID = 1;
    top->S_AXI_AWPROT  = 0;
    top->S_AXI_WDATA   = data;
    top->S_AXI_WVALID  = 1;
    top->S_AXI_WSTRB   = 0xF;
    top->S_AXI_BREADY  = 1;

    // Wait for both AWREADY and WREADY
    for (int i = 0; i < 10; i++) {
        tick(top);
        if (top->S_AXI_AWREADY) top->S_AXI_AWVALID = 0;
        if (top->S_AXI_WREADY)  top->S_AXI_WVALID  = 0;
        if (!top->S_AXI_AWVALID && !top->S_AXI_WVALID) break;
    }

    // Wait for BVALID
    for (int i = 0; i < 10; i++) {
        tick(top);
        if (top->S_AXI_BVALID) break;
    }
    tick(top); // Extra cycle to clear response
}

// AXI4-Lite READ transaction
static uint32_t axi_read(Vyolov8_axi_wrapper* top, uint32_t addr) {
    top->S_AXI_ARADDR  = addr;
    top->S_AXI_ARVALID = 1;
    top->S_AXI_ARPROT  = 0;
    top->S_AXI_RREADY  = 1;

    // Wait for ARREADY + RVALID
    for (int i = 0; i < 10; i++) {
        tick(top);
        if (top->S_AXI_ARREADY) top->S_AXI_ARVALID = 0;
        if (top->S_AXI_RVALID) break;
    }

    uint32_t data = top->S_AXI_RDATA;
    tick(top); // Consume response
    return data;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vyolov8_axi_wrapper* top = new Vyolov8_axi_wrapper;

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  YOLOv8 AXI-Lite Wrapper Verilator Testbench    ║\n");
    printf("║  Testing register-based PS→PL communication     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // Initialize AXI signals
    top->S_AXI_ACLK    = 0;
    top->S_AXI_ARESETN = 0;
    top->S_AXI_AWVALID = 0;
    top->S_AXI_WVALID  = 0;
    top->S_AXI_BREADY  = 0;
    top->S_AXI_ARVALID = 0;
    top->S_AXI_RREADY  = 0;

    // Reset
    for (int i = 0; i < 20; i++) tick(top);
    top->S_AXI_ARESETN = 1;
    for (int i = 0; i < 10; i++) tick(top);
    printf("  ✅ Reset complete\n\n");

    // ================================================================
    // Load Input Feature Map via AXI registers
    // ================================================================
    printf("  Loading layer15_output.mem via AXI...\n");
    std::ifstream infile("layer15_output.mem");
    if (!infile.is_open()) {
        printf("ERROR: Cannot open layer15_output.mem\n");
        return -1;
    }

    std::string line;
    uint32_t addr = 0;
    while (std::getline(infile, line) && addr < TOTAL_IN) {
        uint8_t val = (uint8_t)std::stoul(line, nullptr, 16);
        axi_write(top, REG_FM_ADDR, addr);
        axi_write(top, REG_FM_DATA, val);
        addr++;
    }
    infile.close();
    printf("  ✅ Loaded %d input values via AXI\n\n", addr);

    // ================================================================
    // Start pipeline via AXI CTRL register
    // ================================================================
    axi_write(top, REG_CTRL, 1);
    printf("  [AXI] Pipeline started via CTRL register...\n");

    uint64_t cycles = 0;
    bool done_detected = false;

    // Poll STATUS register until done
    while (!Verilated::gotFinish()) {
        tick(top);
        cycles++;

        if (cycles % 100000 == 0) {
            uint32_t status = axi_read(top, REG_STATUS);
            printf("  ... Cycle %lu, STATUS=0x%08x (done=%d, busy=%d)\n",
                   cycles, status, status & 1, (status >> 1) & 1);
        }

        // Direct check (for speed — AXI polling is slow in sim)
        uint32_t status = axi_read(top, REG_STATUS);
        if (status & 0x1) {
            done_detected = true;
            break;
        }

        if (cycles > 50000000) {
            printf("\n  ❌ TIMEOUT at %lu cycles!\n", cycles);
            break;
        }
    }

    if (done_detected) {
        printf("  ✅ Pipeline completed!\n");

        // Read performance counters
        uint32_t perf_det = axi_read(top, REG_PERF_DET);
        uint32_t perf_nms = axi_read(top, REG_PERF_NMS);
        printf("  [Perf] Det Head cycles: %u\n", perf_det);
        printf("  [Perf] NMS cycles:      %u\n\n", perf_nms);

        // Read NMS results from internal BRAM via AXI
        printf("  [NMS Output] Reading via AXI registers:\n");
        int det_count = 0;
        for (int i = 0; i < MAX_OUT_DETS; i++) {
            // Read word 0
            axi_write(top, REG_NMS_MEM_ADDR, i * 3 + 0);
            tick(top); // BRAM read latency
            uint32_t word0 = axi_read(top, REG_NMS_MEM_RDATA);
            if (word0 == 0xDEADBEEF || word0 == 0) break;

            // Read word 1
            axi_write(top, REG_NMS_MEM_ADDR, i * 3 + 1);
            tick(top);
            uint32_t word1 = axi_read(top, REG_NMS_MEM_RDATA);

            // Read word 2
            axi_write(top, REG_NMS_MEM_ADDR, i * 3 + 2);
            tick(top);
            uint32_t word2 = axi_read(top, REG_NMS_MEM_RDATA);

            uint16_t cx = (word0 >> 16) & 0x3FF;
            uint16_t cy = word0 & 0x3FF;
            uint16_t x1 = (word1 >> 16) & 0x3FF;
            uint16_t y1 = word1 & 0x3FF;
            uint16_t x2 = (word2 >> 16) & 0x3FF;
            uint16_t y2 = word2 & 0x3FF;
            printf("  Det#%d: cx=%d, cy=%d, bbox=[%d,%d,%d,%d]\n",
                   det_count, cx, cy, x1, y1, x2, y2);
            det_count++;
        }
        printf("  Total NMS survivors: %d\n", det_count);
    }

    delete top;
    return 0;
}
