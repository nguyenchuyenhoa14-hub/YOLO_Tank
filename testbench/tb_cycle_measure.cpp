#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "Vyolov8_top_core.h"
#include "Vyolov8_top_core_yolov8_top_core.h"
#include "Vyolov8_top_core_detect_head_seq__pi1.h"
#include "verilated.h"

// Grid parameters for P3
#define GRID_W 40
#define GRID_H 40
#define IN_CH 64
#define TOTAL_IN (GRID_W * GRID_H * IN_CH)
#define MAX_OUT_DETS 20
#define NMS_MEM_SIZE 65536

uint32_t nms_memory[NMS_MEM_SIZE];
uint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vyolov8_top_core* top = new Vyolov8_top_core;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Per-Layer Cycle Count Measurement           ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    for (int i=0; i<NMS_MEM_SIZE; i++) nms_memory[i] = 0;

    // Reset
    top->clk = 0; top->rst_n = 0; top->start = 0; top->ext_wr_en = 0;
    top->ext_w_rd_data = 0; top->ext_b_rd_data = 0; top->ext_m_rd_data = 0;
    top->ext_s_rd_data = 0; top->ext_z_rd_data = 0; top->nms_mem_rd_data = 0;
    top->p3_det_count = 0; top->p4_det_count = 0;
    top->p3_base_addr = 0; top->p4_base_addr = 0; top->nms_out_base_addr = 0;

    for (int i=0; i<20; i++) { top->clk = !top->clk; top->eval(); }
    top->rst_n = 1;
    for (int i=0; i<20; i++) { top->clk = !top->clk; top->eval(); }

    // Load Input
    printf("Loading layer15_output.mem...\n");
    std::ifstream infile("layer15_output.mem");
    if(!infile.is_open()) { printf("ERROR: Cannot open layer15_output.mem\n"); return -1; }

    std::string line;
    uint32_t addr = 0;
    while(std::getline(infile, line) && addr < TOTAL_IN) {
        uint8_t val = (uint8_t)std::stoul(line, nullptr, 16);
        top->ext_wr_addr = addr; top->ext_wr_data = val; top->ext_wr_en = 1;
        top->clk = 1; top->eval(); top->clk = 0; top->eval();
        addr++;
    }
    infile.close();
    top->ext_wr_en = 0;
    top->clk = 1; top->eval(); top->clk = 0; top->eval();
    printf("  Loaded %d input values\n\n", addr);

    // Start
    top->start = 1;
    top->clk = 1; top->eval(); top->clk = 0; top->eval();
    top->start = 0;

    // Track per-layer cycles
    const char* layer_names[] = {"cv2_0 (3x3, IN->64)", "cv2_1 (3x3, 64->64)", "cv2_2 (1x1, 64->64)",
                                  "cv3_0 (3x3, IN->64)", "cv3_1 (3x3, 64->64)", "cv3_2 (1x1, 64->1)"};
    uint64_t layer_start_cycle[6] = {0};
    uint64_t layer_end_cycle[6] = {0};
    int prev_stage = -1;
    uint64_t det_start = 0;
    uint64_t det_end = 0;
    uint64_t nms_start = 0;
    uint64_t nms_end = 0;
    bool det_started = false;
    bool nms_started = false;

    uint64_t cycles = 0;

    while (!Verilated::gotFinish()) {
        top->clk = 1; top->eval();
        main_time += 5;

        // NMS memory emulation
        if (top->nms_mem_wr_en && top->nms_mem_wr_addr < NMS_MEM_SIZE)
            nms_memory[top->nms_mem_wr_addr] = top->nms_mem_wr_data;
        if (top->nms_mem_rd_en && top->nms_mem_rd_addr < NMS_MEM_SIZE)
            top->nms_mem_rd_data = nms_memory[top->nms_mem_rd_addr];

        top->clk = 0; top->eval();
        main_time += 5;
        cycles++;

        // Track detect head active_stage via hierarchical access
        int cur_stage = top->yolov8_top_core->u_detect_head->active_stage;
        int det_fsm = top->yolov8_top_core->u_detect_head->fsm_state;
        int det_busy = top->yolov8_top_core->u_detect_head->busy;

        // Track detection head start
        if (det_busy && !det_started) {
            det_started = true;
            det_start = cycles;
        }

        // Track layer transitions
        if (det_busy && cur_stage != prev_stage && cur_stage >= 0 && cur_stage <= 5) {
            if (prev_stage >= 0 && prev_stage <= 5) {
                layer_end_cycle[prev_stage] = cycles;
            }
            layer_start_cycle[cur_stage] = cycles;
            printf("  [Cycle %7lu] Stage %d -> %d (%s)\n", cycles, prev_stage, cur_stage,
                   cur_stage <= 5 ? layer_names[cur_stage] : "?");
            prev_stage = cur_stage;
        }

        // Track detection head end
        if (det_started && !det_busy && det_end == 0) {
            det_end = cycles;
            if (prev_stage >= 0 && prev_stage <= 5)
                layer_end_cycle[prev_stage] = cycles;
        }

        // Track NMS
        int nms_busy_sig = top->yolov8_top_core->nms_busy;
        if (nms_busy_sig && !nms_started) {
            nms_started = true;
            nms_start = cycles;
        }
        if (nms_started && !nms_busy_sig && nms_end == 0) {
            nms_end = cycles;
        }

        if (cycles % 500000 == 0)
            printf("  ... running ... Cycle %lu\n", cycles);

        if (top->done) break;
        if (cycles > 50000000) { printf("\n  TIMEOUT!\n"); break; }
    }

    // DFL cycles: between det_end and nms_start
    uint64_t dfl_start = det_end;
    uint64_t dfl_end = nms_start;

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  PER-LAYER CYCLE COUNT RESULTS               ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");
    printf("  %-30s %10s %10s %10s\n", "Layer/Module", "Start", "End", "Cycles");
    printf("  %-30s %10s %10s %10s\n", "------------------------------", "----------", "----------", "----------");

    uint64_t total_conv = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t c = layer_end_cycle[i] - layer_start_cycle[i];
        total_conv += c;
        printf("  %-30s %10lu %10lu %10lu\n", layer_names[i], layer_start_cycle[i], layer_end_cycle[i], c);
    }
    uint64_t dfl_cycles = (dfl_end > dfl_start) ? (dfl_end - dfl_start) : 0;
    uint64_t nms_cycles = (nms_end > nms_start) ? (nms_end - nms_start) : 0;

    printf("  %-30s %10s %10s %10lu\n", "--- Total Conv ---", "", "", total_conv);
    printf("  %-30s %10lu %10lu %10lu\n", "DFL Decode", dfl_start, dfl_end, dfl_cycles);
    printf("  %-30s %10lu %10lu %10lu\n", "NMS", nms_start, nms_end, nms_cycles);
    printf("  %-30s %10s %10s %10lu\n", "=== TOTAL ===", "", "", cycles);
    printf("\n  [Perf Counter] Det Head: %u cycles\n", top->perf_det_cycles);
    printf("  [Perf Counter] NMS:      %u cycles\n", top->perf_nms_cycles);
    printf("  @ 100 MHz: %.2f ms, %.1f FPS\n", cycles / 100000.0, 100000000.0 / cycles);

    delete top;
    return 0;
}
