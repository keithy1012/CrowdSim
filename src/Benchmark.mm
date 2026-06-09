#import "Benchmark.h"
#import "GPUSimulation.h"
#include "Simulation.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// ── Configuration ─────────────────────────────────────────────────────────────

static const uint32_t kAgentCounts[]  = {1'000, 5'000, 10'000, 50'000, 100'000, 250'000};
static const int      kNumCounts      = 6;
static const uint32_t kCPUThreads[]   = {1, 2, 4, 8};
static const int      kNumCPUScenarios = 4;
static const int      kWarmupFrames   = 10;
static const int      kMeasureFrames  = 30;
static const double   kBailMs         = 400.0;  // abort scenario if one frame exceeds this

// ── Timing helpers ────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

static double nowMs() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

// Returns average frame time in ms, or NAN if bailed out.
static double benchmarkCPU(uint32_t agentCount, uint32_t numThreads) {
    Simulation sim(agentCount);

    for (int f = 0; f < kWarmupFrames; f++) {
        if (numThreads > 1) sim.update(1.f / 60.f, numThreads);
        else                sim.update(1.f / 60.f);
    }

    double total = 0.0;
    for (int f = 0; f < kMeasureFrames; f++) {
        double t0 = nowMs();
        if (numThreads > 1) sim.update(1.f / 60.f, numThreads);
        else                sim.update(1.f / 60.f);
        double elapsed = nowMs() - t0;

        if (f == 0 && elapsed > kBailMs) return NAN;
        total += elapsed;
    }
    return total / kMeasureFrames;
}

static double benchmarkGPU(id<MTLDevice> device,
                           id<MTLLibrary> library,
                           uint32_t agentCount,
                           BOOL useGrid) {
    GPUSimulation *sim = [[GPUSimulation alloc] initWithDevice:device
                                                       library:library
                                                    agentCount:agentCount];
    sim.useGridSteering = useGrid;
    id<MTLCommandQueue> queue = [device newCommandQueue];

    for (int f = 0; f < kWarmupFrames; f++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        [sim encodeToCommandBuffer:cmd dt:1.f / 60.f];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    double total = 0.0;
    for (int f = 0; f < kMeasureFrames; f++) {
        double t0 = nowMs();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        [sim encodeToCommandBuffer:cmd dt:1.f / 60.f];
        [cmd commit];
        [cmd waitUntilCompleted];
        double elapsed = nowMs() - t0;

        if (f == 0 && elapsed > kBailMs) return NAN;
        total += elapsed;
    }
    return total / kMeasureFrames;
}

// ── Formatting ────────────────────────────────────────────────────────────────

static std::string fmtMs(double ms) {
    if (std::isnan(ms)) return "  —   ";
    if (ms < 10.0)      return std::to_string((int)std::round(ms * 10)) + "." +
                               std::to_string((int)std::round(ms * 10) % 10) +
                               "  "; // kludge; use snprintf below instead
    char buf[16];
    if (ms < 10.0)   snprintf(buf, sizeof(buf), "%5.2f", ms);
    else if (ms < 100.0) snprintf(buf, sizeof(buf), "%5.1f", ms);
    else             snprintf(buf, sizeof(buf), "%5.0f", ms);
    return std::string(buf);
}

static std::string cell(double ms) {
    if (std::isnan(ms)) return "   —  ";
    char buf[16];
    if      (ms <  10.0) snprintf(buf, sizeof(buf), " %5.2f", ms);
    else if (ms < 100.0) snprintf(buf, sizeof(buf), " %5.1f", ms);
    else                 snprintf(buf, sizeof(buf), " %5.0f", ms);
    return std::string(buf);
}

// ── Entry point ───────────────────────────────────────────────────────────────

int runBenchmark(id<MTLDevice> device, id<MTLLibrary> library) {
    // scenario × count results grid; NAN = bailed
    // Rows 0-3: CPU threads {1,2,4,8}
    // Rows 4-5: GPU {O(N²), Spatial Hash}
    static const int kRows = 6;
    double results[kRows][kNumCounts];
    for (int r = 0; r < kRows; r++)
        for (int c = 0; c < kNumCounts; c++)
            results[r][c] = 0.0;

    printf("\n");
    printf("================================================================\n");
    printf("  CrowdSim — Phase 4 Benchmark\n");
    printf("  %s\n", device.name.UTF8String);
    printf("  %d warmup + %d measurement frames  |  bail threshold: %.0f ms\n",
           kWarmupFrames, kMeasureFrames, kBailMs);
    printf("================================================================\n\n");

    // ── CPU scenarios ─────────────────────────────────────────────────────────
    for (int s = 0; s < kNumCPUScenarios; s++) {
        uint32_t threads = kCPUThreads[s];
        printf("  CPU %2u thread%s ...\n", threads, threads == 1 ? " " : "s");
        fflush(stdout);

        bool bailed = false;
        for (int c = 0; c < kNumCounts; c++) {
            if (bailed) { results[s][c] = NAN; continue; }
            printf("    %6u agents ... ", kAgentCounts[c]); fflush(stdout);
            double ms = benchmarkCPU(kAgentCounts[c], threads);
            results[s][c] = ms;
            if (std::isnan(ms)) { printf("bailed\n"); bailed = true; }
            else                  printf("%.2f ms\n", ms);
        }
    }

    // ── GPU scenarios ─────────────────────────────────────────────────────────
    const char *gpuNames[] = { "GPU  O(N²) [Phase 2]", "GPU  Spatial Hash [Phase 3]" };
    BOOL        gpuGrid[]  = { NO, YES };

    for (int g = 0; g < 2; g++) {
        int row = kNumCPUScenarios + g;
        printf("  %s ...\n", gpuNames[g]);
        fflush(stdout);

        bool bailed = false;
        for (int c = 0; c < kNumCounts; c++) {
            if (bailed) { results[row][c] = NAN; continue; }
            printf("    %6u agents ... ", kAgentCounts[c]); fflush(stdout);
            double ms = benchmarkGPU(device, library, kAgentCounts[c], gpuGrid[g]);
            results[row][c] = ms;
            if (std::isnan(ms)) { printf("bailed\n"); bailed = true; }
            else                  printf("%.2f ms\n", ms);
        }
    }

    // ── Results table ─────────────────────────────────────────────────────────
    printf("\n");
    printf("================================================================\n");
    printf("  Frame time (ms)  ·  lower is better  ·  16.7 ms = 60 FPS\n");
    printf("================================================================\n");

    // Header
    printf("  %-24s", "");
    for (int c = 0; c < kNumCounts; c++) {
        uint32_t n = kAgentCounts[c];
        char buf[10];
        if      (n >= 1'000'000) snprintf(buf, sizeof(buf), " %4uM", n / 1'000'000);
        else if (n >= 1'000)     snprintf(buf, sizeof(buf), " %3uK",  n / 1'000);
        else                     snprintf(buf, sizeof(buf), " %4u",   n);
        printf(" %6s", buf);
    }
    printf("\n");

    // Divider
    printf("  %-24s", "");
    for (int c = 0; c < kNumCounts; c++) printf(" ------");
    printf("\n");

    const char *rowNames[] = {
        "CPU  1 thread",
        "CPU  2 threads",
        "CPU  4 threads",
        "CPU  8 threads",
        "GPU  O(N\xc2\xb2)  [Ph.2]",
        "GPU  Spat.Hash [Ph.3]",
    };

    for (int r = 0; r < kRows; r++) {
        if (r == kNumCPUScenarios)
            printf("  %s\n", std::string(24 + kNumCounts * 7, '-').c_str());
        printf("  %-24s", rowNames[r]);
        for (int c = 0; c < kNumCounts; c++)
            printf("%s", cell(results[r][c]).c_str());
        printf("\n");
    }
    printf("================================================================\n");

    // ── Speedup summary at the largest common agent count ────────────────────
    // Find largest count where CPU 1T and GPU SH both have valid results
    int refCount = -1;
    for (int c = kNumCounts - 1; c >= 0; c--) {
        if (!std::isnan(results[0][c]) && !std::isnan(results[kRows-1][c])) {
            refCount = c; break;
        }
    }
    if (refCount >= 0) {
        double base = results[0][refCount];
        printf("\n  Speedup vs CPU 1-thread at %uK agents:\n",
               kAgentCounts[refCount] / 1000);
        for (int r = 1; r < kRows; r++) {
            double ms = results[r][refCount];
            if (std::isnan(ms)) continue;
            printf("    %-24s %5.1f×\n", rowNames[r], base / ms);
        }
    }

    printf("\n");
    return 0;
}
