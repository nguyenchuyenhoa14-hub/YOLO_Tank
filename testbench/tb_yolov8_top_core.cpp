#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "Vyolov8_top_core.h"
#include "verilated.h"

// Grid parameters for P3
#define GRID_W 40
#define GRID_H 40
#define IN_CH 64
#define BBOX_CH 64
#define TOTAL_IN (GRID_W * GRID_H * IN_CH)
#define MAX_OUT_DETS 20

// Simulated external memory for NMS Output
#define NMS_MEM_SIZE 65536
uint32_t nms_memory[NMS_MEM_SIZE];

uint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vyolov8_top_core* top = new Vyolov8_top_core;

    printf("╔══════════════════════════════════════════╗\n");
    printf("║  YOLOv8 Top Core Verilator Testbench     ║\n");
    printf("║  Grid: %dx%d    IN_CH: %d    PARALLEL: 16 ║\n", GRID_W, GRID_H, IN_CH);
    printf("╚══════════════════════════════════════════╝\n\n");

    // Initialize Memory
    for (int i=0; i<NMS_MEM_SIZE; i++) nms_memory[i] = 0;
    
    // Reset sequence
    top->clk = 0;
    top->rst_n = 0;
    top->start = 0;
    top->ext_wr_en = 0;
    
    // Dummy inputs for un-used ports
    top->ext_w_rd_data = 0;
    top->ext_b_rd_data = 0;
    top->ext_m_rd_data = 0;
    top->ext_s_rd_data = 0;
    top->ext_z_rd_data = 0;
    top->nms_mem_rd_data = 0;
    top->p3_det_count = 0;
    top->p4_det_count = 0;
    top->p3_base_addr = 0;
    top->p4_base_addr = 0;
    top->nms_out_base_addr = 0;

    // Apply Reset
    for (int i=0; i<20; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    top->rst_n = 1;
    for (int i=0; i<20; i++) {
        top->clk = !top->clk;
        top->eval();
    }

    // Load Input Feature Map (layer15_output.mem)
    printf("Loading layer15_output.mem...\n");
    std::ifstream infile("layer15_output.mem");
    if(!infile.is_open()) {
        printf("ERROR: Cannot open layer15_output.mem\n");
        return -1;
    }

    std::string line;
    uint32_t addr = 0;
    while(std::getline(infile, line) && addr < TOTAL_IN) {
        uint8_t val = (uint8_t)std::stoul(line, nullptr, 16);
        top->ext_wr_addr = addr;
        top->ext_wr_data = val;
        top->ext_wr_en = 1;
        
        top->clk = 1; top->eval();
        top->clk = 0; top->eval();
        addr++;
    }
    infile.close();
    top->ext_wr_en = 0;
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
    printf("  ✅ Loaded %d input values\n\n", addr);

    // Start execution
    top->start = 1;
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
    top->start = 0;

    printf("  [Top_Core] Pipeline started...\n");

    uint64_t cycles = 0;
    bool done_detected = false;

    // Run simulation clock loop
    while (!Verilated::gotFinish()) {
        top->clk = 1; top->eval();
        main_time += 5;
        
        // Emulate ASYNC NMS memory Write
        if (top->nms_mem_wr_en) {
            if (top->nms_mem_wr_addr < NMS_MEM_SIZE) {
                nms_memory[top->nms_mem_wr_addr] = top->nms_mem_wr_data;
            }
        }
        
        // Emulate NMS memory Read (Sync, available on next cycle)
        if (top->nms_mem_rd_en) {
            if (top->nms_mem_rd_addr < NMS_MEM_SIZE) {
                top->nms_mem_rd_data = nms_memory[top->nms_mem_rd_addr];
            }
        }

        top->clk = 0; top->eval();
        main_time += 5;
        
        cycles++;

        if (cycles % 100000 == 0) {
            printf("  ... running ... Cycle %lu\n", cycles);
        }

        if (top->done) {
            done_detected = true;
            break;
        }

        if (cycles > 50000000) { // Timeout
            printf("\n  ❌ TIMEOUT at %lu cycles!\n", cycles);
            break;
        }
    }

    if (done_detected) {
        printf("  ✅ Pipeline completed in %lu cycles\n", cycles);
        printf("  [RTL] Est. Latency @ 100 MHz = %.2f ms\n", (float)cycles / 100000.0f);
        printf("  [RTL] Est. FPS @ 100 MHz = %.1f FPS\n\n", 100000000.0f / (float)cycles);

        // Print performance counters
        printf("  [Perf] Det Head cycles: %u\n", top->perf_det_cycles);
        printf("  [Perf] NMS cycles:      %u\n\n", top->perf_nms_cycles);
        
        // Read NMS output memory for final detection results
        // Format: 3 words per detection [cx|cy, x1|y1, x2|y2], terminated by DEADBEEF
        printf("  [NMS Output] Reading detection results from NMS memory:\n");
        int det_count = 0;
        for (int i = 0; i < MAX_OUT_DETS; i++) {
            uint32_t word0 = nms_memory[i * 3 + 0];
            if (word0 == 0xDEADBEEF || word0 == 0) break;
            uint32_t word1 = nms_memory[i * 3 + 1];
            uint32_t word2 = nms_memory[i * 3 + 2];
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
