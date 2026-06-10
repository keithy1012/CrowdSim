# CrowdSim

GPU-accelerated crowd simulation engine built with C++20 and Metal on Apple Silicon.

Simulates tens to hundreds of thousands of autonomous agents in real time using parallel GPU compute, spatial hashing, and steering behaviors.

---

## Platform

|          |                       |
| -------- | --------------------- |
| Hardware | Apple Silicon (M2+)   |
| OS       | macOS                 |
| GPU API  | Metal                 |
| Language | C++20 / Objective-C++ |
| Build    | CMake 3.20+           |
| Editor   | VS Code               |

---

## Current State

**Phase 0 — Project Setup (complete)**

- CMake build system with Metal, MetalKit, AppKit, Foundation, and QuartzCore frameworks
- Metal render loop via `MTKView` at 60 FPS, clearing to a dark background
- `AgentBuffers` SoA layout defined (`posX/Y`, `velX/Y`, `targetX/Y`, `radius`, `maxSpeed`)
- Metal shader compilation wired into CMake (requires full Xcode; stubs in place)
- VS Code configured with IntelliSense, build task (`Cmd+Shift+B`), and LLDB debug launch (`F5`)

**Phase 1 — CPU Prototype (complete)**

- `Simulation` class with pre-allocated SoA force accumulators (`_fx`, `_fy`)
- Two-pass update: forces accumulated first (reads clean previous-frame state), then velocity and position integrated
- Four steering behaviors: goal seeking, separation (distance-weighted), alignment, cohesion
- O(N²) neighbor scan with early squared-distance reject
- Agents wrap at world edges and reassign random targets on arrival
- Instanced circle rendering via `vs_agent` / `fs_agent` Metal shaders — positions uploaded from CPU each frame
- Verified: 1,000 agents at 60 FPS (Debug); scale `kAgentCount` in `AppDelegate.mm` to benchmark 10K

**Phase 2 — Metal GPU Port (complete)**

- `GPUSimulation` class manages all agent data as `MTLBuffer` SoA arrays (shared storage, CPU writes once at init)
- `SharedTypes.h` — plain-C `SimParams` struct included by both C++ and Metal shaders
- Two compute passes encoded per frame via separate `MTLComputeCommandEncoder` instances (ordering guarantees steer writes are visible to integrate)
- `k_steer` kernel: O(N²) neighbour scan on GPU, goal seeking, separation/alignment/cohesion, GPU-side target reassignment via Wang hash on arrival
- `k_integrate` kernel: velocity clamping, position integration, world-edge wrapping with `floor`-based modulo
- `vs_agent` updated to read SoA `posX`/`posY` directly — no CPU packing step, no readback
- CPU simulation path preserved in `Renderer` for Phase 4 benchmarking
- Verified: 50,000 agents at 60 FPS [GPU]

**Phase 3 — Spatial Hash Grid (complete)**

- World divided into a 26×15 uniform grid (cellSize = neighborRadius = 50 px, 390 total cells)
- 7 compute passes per frame: clear → hash → prefix sum → scatter → reorder → steer → integrate
- `k_clearGrid`: zeros per-cell counters atomically before each frame
- `k_hash`: assigns each agent to a cell, atomically counts agents per cell
- `k_prefixSum`: exclusive prefix sum on cell counts → `cellStart[]` array (single-thread, 390 iterations)
- `k_scatter`: places each agent into its sorted slot using per-cell atomic insert cursors
- `k_reorder`: gathers `posX/Y`, `velX/Y`, `maxSpeed` into cell-sorted SoA buffers
- `k_steerGrid`: iterates only the 3×3 block of cells around each agent (~2,400 candidates vs. 100,000)
- `k_integrate`: unchanged from Phase 2
- `SimParams` extended with `gridWidth`, `gridHeight`, `numCells`, `cellSize`

**Bug encountered — cache-hostile random reads (mitigated):**
Without `k_reorder`, `k_steerGrid` accessed neighbor data as `posX[sortedAgentIndex[k]]` — a random read scattered across the full N-element buffer for every neighbor candidate. At 100K agents this produced ~230M random memory accesses per frame, thrashing the GPU cache and dropping performance to ~10 FPS. The mitigation was adding the `k_reorder` pass, which gathers agent data into cell-sorted order before the steer pass. With sorted SoA, threads in the same SIMD group (which land in the same or adjacent cells after sorting) read sequential memory addresses and share cache lines, restoring full throughput.

- Verified: 100,000 agents at 60 FPS [GPU]

## Phase 4 — Performance Engineering and Benchmarking

**Goal:** Quantify scalability and understand performance bottlenecks across six scenarios: four CPU thread counts and two GPU algorithms.

Run with:

```bash
build/CrowdSim.app/Contents/MacOS/CrowdSim --benchmark
```

**Scenarios benchmarked:**

| #   | Scenario           | Algorithm                                             |
| --- | ------------------ | ----------------------------------------------------- |
| 1   | CPU — 1 thread     | O(N²) steering, single-threaded                       |
| 2   | CPU — 2 threads    | O(N²) steering, force + integrate phases parallelized |
| 3   | CPU — 4 threads    | same                                                  |
| 4   | CPU — 8 threads    | same                                                  |
| 5   | GPU — O(N²)        | Phase 2 `k_steer` kernel, one GPU thread per agent    |
| 6   | GPU — Spatial Hash | Phase 3 `k_steerGrid` + 5-pass grid build             |

**Results — Apple M3, 10 warmup + 30 measurement frames**

Frame time in ms (lower is better). `—` = exceeded 400 ms bail threshold.

```
                           1K     5K    10K    50K   100K   250K
                       ------ ------ ------ ------ ------ ------
CPU  1 thread           0.73  16.49  68.38    —      —      —
CPU  2 threads          0.45   8.50  33.78    —      —      —
CPU  4 threads          0.32   4.45  18.27    —      —      —
CPU  8 threads          0.33   4.10  15.54  373.82   —      —
------------------------------------------------------------------
GPU  O(N²)  [Ph.2]      0.87   1.16   3.45  48.72  189.26   —
GPU  Spat.Hash [Ph.3]   0.23   1.23   0.33   1.94    6.59  37.39
```

**Speedup vs CPU 1-thread at 10K agents:**

| Scenario         | Speedup |
| ---------------- | ------- |
| CPU 2 threads    | 2.0×    |
| CPU 4 threads    | 3.7×    |
| CPU 8 threads    | 4.4×    |
| GPU O(N²)        | 19.8×   |
| GPU Spatial Hash | 208.1×  |

**Analysis:**

_CPU threading — diminishing returns above 4 threads_

Going from 1 → 2 threads gives a near-ideal 2.0× speedup, and 1 → 4 gives 3.7×. The jump from 4 → 8 threads only adds an additional 0.8× (total 4.4×) because the M3 has 4 performance cores and 4 efficiency cores; the efficiency cores run the same O(N²) kernel at lower clock speed and memory bandwidth, contributing less than a full P-core would. The force accumulation pass also reads all N positions for every agent, which is memory-bandwidth bound at high N — adding more threads doesn't hide that cost.

_O(N²) is the hard ceiling for both CPU and GPU_

The CPU 8-thread path bails at 100K and barely survives 50K (374 ms, well outside the 16.7 ms budget). The GPU O(N²) path is faster — 19.8× at 10K — because the GPU has thousands of concurrent threads hiding memory latency, but it still scales quadratically and bails at 250K. At 50K agents the GPU O(N²) frame time is 48.7 ms (≈20 FPS); at 100K it's 189 ms (≈5 FPS). Doubling N quadruples the work, exactly as expected.

_Spatial hashing breaks the O(N²) wall_

The GPU spatial hash path (Phase 3) stays under the 16.7 ms budget through 100K agents (6.59 ms) and only reaches 37.4 ms at 250K — still four times faster than the GPU O(N²) path at 100K. The 208× speedup over CPU 1-thread at 10K comes from two compounding effects: the GPU's parallelism (~20× over single-threaded CPU) combined with the algorithmic improvement from O(N²) to O(N) neighbor search (~10× at this density).

_The 5K anomaly_

The spatial hash path shows 1.23 ms at 5K but only 0.33 ms at 10K. This is not a measurement error — it reflects the overhead of the 5-pass grid build (clear → hash → prefix sum → scatter → reorder) dominating at low N. At 5K the grid build is relatively expensive compared to the steering work; at 10K the steering work grows enough to amortize the build cost. At 50K+ the ratio is favorable and the algorithm clearly wins.

_Real-time thresholds_

The 16.7 ms budget for 60 FPS is only met by:

- CPU: no scenario at 50K+ agents
- GPU O(N²): up to ~10K agents
- GPU Spatial Hash: up to 100K agents comfortably; 250K is reachable (37 ms ≈ 27 FPS) and would hit 60 FPS with further optimization (Phase 5+)

## Phase 5 — Obstacle Avoidance and Crowd Scenarios

**Obstacle representation:** line segments stored in a GPU buffer, uploaded once at scene load via `loadObstacles:count:`.

```c
struct Obstacle {
    float x0, y0;   // segment start
    float x1, y1;   // segment end
};
```

**Avoidance force** — added to `k_steerGrid` (Pass 6) after the flocking forces. For each obstacle segment, the kernel computes the closest point on the segment to the agent, then applies a repulsion force proportional to how deep into the avoidance radius the agent is:

```metal
float2 AB = B - A;
float  t  = clamp(dot(P - A, AB) / dot(AB, AB), 0.f, 1.f);
float2 closest  = A + t * AB;
float2 repulse  = P - closest;
float  dist     = length(repulse);
if (dist < avoidR && dist > 0.001f) {
    float strength = (avoidR - dist) / avoidR;
    force += 3 * maxSpeed * strength * normalize(repulse);
}
```

The weight of 3 makes obstacle avoidance override flocking and goal-seeking near walls. Avoidance radius = 40 px (less than the neighbor radius of 50 px, so it kicks in only on close approach).

**Scenes** — switch with keys `1` / `2` / `3`:

| Key | Scene      | Obstacles                                                                                                                  |
| --- | ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| `1` | Open Field | None                                                                                                                       |
| `2` | Barrier    | Two staggered horizontal walls. Gap on right (wall A), gap on left (wall B) — forces a zigzag path through the 100K crowd. |
| `3` | Pillars    | Six 40×40 px square pillars (24 segments total) in a 3×2 grid — agents navigate around columns.                            |

**Velocity colour coding** — `vs_agent` now reads per-agent velocity and max-speed buffers. Slow agents render blue, fast agents render red. The hue mapping is: `hue = (1 - speed) × 0.667`, giving the full blue→cyan→green→yellow→red spectrum as agents accelerate.

**Obstacle line rendering** — a separate `vs_obstacle` / `fs_obstacle` pipeline renders each segment as a light-grey line using `MTLPrimitiveTypeLine`.

**Freehand drawing** — left-click and drag to draw obstacles directly on the canvas. Each 12 px of mouse movement commits one line segment. Segments are appended to the active scene's preset obstacles and take effect immediately — agents start avoiding them on the next frame. Press `C` to clear all drawn segments and reset to the current scene's preset.

The obstacle buffer starts at 64 segments and doubles automatically whenever it fills up (`GPUSimulation.appendObstacle:` follows the standard doubling strategy, swapping the `id<MTLBuffer>` pointer before the next encode). The practical ceiling is frame time, not memory: each agent checks every obstacle segment inside `k_steerGrid`, so drawing thousands of segments will reduce FPS proportionally.

Agents are not re-spawned when scenes change.

## Phase 6 — Advanced Crowd Navigation (complete)

### Flow Fields

Flow fields replace per-agent goal-seeking with a global vector field precomputed over the grid:

```
Destination (right-click to place)
    ↓
8-directional Dijkstra BFS over 26×15 grid (390 cells, 50 px each)
    ↓
390 normalised float2 vectors uploaded to GPU (3 KB buffer)
    ↓
Each agent samples flowField[cellX + cellY * gridWidth] → O(1) path lookup
```

Benefits: all 100K agents share one pathfinding computation, naturally routes around obstacles, familiar in RTS games.

A white downward-pointing triangle marks the current goal. Right-click anywhere to move it; press `F` to toggle flow-field mode (agents revert to random wandering when off).

Implementation: CPU-side Dijkstra runs on SimParams change via a dirty-flag batched to once per frame. Diagonal moves check both cardinal neighbours to prevent corner-cutting through walls.

### ORCA (Optimal Reciprocal Collision Avoidance)

ORCA replaces the heuristic separation force with mathematically optimal, deadlock-free collision avoidance. Press `O` to toggle.

```
For each agent A and each nearby agent B (spatial-hash neighbour):
    Relative position p = B.pos − A.pos
    Relative velocity v = A.vel − B.vel
    Velocity obstacle VO = set of relative velocities that cause collision within τ seconds
    ORCA half-plane: A's new velocity must lie outside ½ of VO (reciprocal responsibility)
        ↓
Collect up to 20 half-plane constraints
        ↓
2D linear program (RVO2 algorithm): find velocity closest to preferred velocity
        that satisfies all half-plane constraints within the speed disk
        ↓
Write result as a force-equivalent so k_integrate yields the LP solution exactly
```

**Key properties:**

- **Reciprocal**: each agent takes half the avoidance responsibility, eliminating oscillation
- **Optimal**: the LP gives the velocity mathematically closest to the preferred direction
- **Deadlock-free**: the LP always finds a feasible velocity (the partial solution is used on infeasibility)
- **Composable with flow fields**: preferred velocity comes from the flow field (or direct seek); ORCA handles only local collisions

**Implementation details:**

- `k_orca` kernel (Pass 6) has the same buffer layout as `k_steerGrid` — toggled by `SimParams.useORCA`
- LP helper `lp1` / `lp2` are static Metal functions; `lp2` calls `lp1` per violated constraint
- Max 20 ORCA constraints per agent (`kMaxORCA`) — caps LP cost; excess neighbours skipped
- Obstacle avoidance: wall repulsion pre-biases the preferred velocity before the LP, so the LP routes around walls while resolving agent-agent collisions
- Time horizon τ = 1.5 s (configurable via `SimParams.orcaTimeHorizon`); agents begin adjusting velocity when collision is predicted within τ seconds
- Force-equivalent write: `forceX[oi] = (newVel.x − currentVel.x) / dt`; after `k_integrate` adds `force × dt`, the velocity is exactly `newVel` with no other changes needed

### Analysis

_Cost model: O(N × kMaxORCA²) worst case per frame_

`k_steerGrid` scans the same 3×3 spatial-hash neighbourhood but does ~10–20 float ops per neighbour (dot product, clamp, weighted accumulate). `k_orca` does the same scan and then runs the 2D LP. `lp2` iterates at most `kMaxORCA = 20` constraints; for each violated constraint it calls `lp1`, which has an inner loop over all previous constraints — worst-case O(kMaxORCA²) = 400 iterations of cheap arithmetic. In practice the LP terminates in 3–8 calls to `lp1` because most constraints are already satisfied by the preferred velocity. At 100K agents the LP overhead is ~3–4× the `k_steerGrid` steering cost, placing `k_orca` at roughly 20–25 ms on the M3 (vs. 6.6 ms for `k_steerGrid`). This sits just outside the 16.7 ms 60 FPS budget; 30 FPS is comfortable.

_Lane formation — the emergent hallmark of ORCA_

The most visually distinctive behaviour of ORCA is spontaneous lane formation in bidirectional flow. When agents moving left and agents moving right approach each other, the heuristic separation force pushes them apart symmetrically in all directions, causing jitter and repeated course corrections. ORCA prevents this: each agent computes the minimum velocity change to avoid every neighbour, and because the correction is reciprocal, agents converging from opposite directions independently converge on the same side-step direction. After 2–3 frames of adjustment the agents have negotiated a passing lane, and the lane persists because agents behind follow the same preferred velocity and encounter the same half-plane geometry. This emerges from the algorithm rather than being programmed directly.

_Oscillation elimination_

In `k_steerGrid`, two agents directly facing each other at separation-radius range apply equal and opposite forces each frame, causing them to rock back and forth without making progress. ORCA eliminates this because the LP always outputs a velocity that is on the correct side of every half-plane — the correction is a one-shot geometric solve, not an accumulating force. Once the LP solution is applied the constraint is satisfied for that frame, and the next frame's geometry is slightly different (agents have moved), so the oscillation cycle never closes.

_Flow field + ORCA: global path, local avoidance_

Running both together produces the most realistic behaviour. The flow field provides a single globally optimal direction to the goal, computed once on the CPU from a Dijkstra BFS. ORCA ensures agents don't pile up at bottlenecks: as the corridor narrows, agents' ORCA half-planes force them to queue and stagger rather than compress into a single cell. The combination is the canonical architecture used in modern game engines for large crowds: a coarse global planner (flow field, navigation mesh, hierarchical pathfinding) gives the preferred velocity, and a fine local avoidance layer (ORCA, RVO) removes collisions frame by frame.

_Known limitations_

Three simplifications were made for this phase:

1. **No `linearProgram3`**: when more than `kMaxORCA` constraints exist (extreme density), `lp2` returns a partial solution — a velocity that satisfies the first N constraints but not necessarily all of them. The full RVO2 algorithm adds a third pass (`lp3`) that minimises the maximum penetration depth across all unsatisfied constraints. Omitting it means agents in very dense packing can briefly overlap; adding `lp3` is a straightforward O(kMaxORCA) extension.

2. **Force-based obstacle ORCA**: proper ORCA uses dedicated half-plane constraints for static line segments (treating each segment as a zero-velocity agent with infinite mass). Here, wall repulsion is applied as a pre-bias to the preferred velocity before the LP runs. This is simpler and works well at low-to-medium density but can fail when agents are simultaneously pressed against a wall and surrounded by neighbours — the LP may choose a velocity that satisfies all agent constraints but points into the wall.

3. **Fixed combined radius**: all agents have radius 5 px, so `combinedR = 10 px` is hardcoded in the kernel. Supporting heterogeneous radii requires adding a sorted-radius SoA buffer (`sRad`) to the reorder pass and passing it to `k_orca` — a one-pass change with no algorithmic impact.

### Threadgroup Memory Tiling

Both `k_steerGrid` and `k_orca` use the same **cooperative-load tiling** pattern adapted from CUDA shared memory:

**The problem without tiling:**

At 100K agents with 256 agents/cell, the 3×3 neighbourhood contains ~2,300 agents. All 64 threads in a threadgroup process consecutive sorted agents that land in the same cell, so they all scan the same ~2,300 neighbors. Each thread independently reads those 2,300 × 5 floats (posX, posY, velX, velY, maxSpeed) from device (global) memory:

```
64 threads × 2,300 neighbours × 5 floats × 4 bytes = ~2.9 MB of global reads per threadgroup
```

**What tiling does:**

Threads cooperatively load 64 agents at a time into **threadgroup memory** (on-chip SRAM, ~1–2 TB/s bandwidth on M-series vs. ~100–200 GB/s for device memory). Each 64-agent tile requires only 64 loads from global memory — one per thread — and all 64 threads then read from the fast TGSM:

```
2,300 neighbours × 5 floats × 4 bytes = ~46 KB of global reads per threadgroup  (64× fewer)
+ 64 threads × 2,300 fast TGSM reads
```

**Barrier uniformity:**

The standard CUDA pitfall is barrier divergence: if different threads in the same threadgroup enter different numbers of tile iterations, the `threadgroup_barrier` calls become unmatched and execution hangs. This is solved by broadcasting thread-0's cell coordinates to all threads at the start of the kernel. Since all threads use the same reference cell, they visit the same 9 cells and the same tile boundaries, executing every barrier in lockstep.

```metal
// All 64 threads use the same cell reference → same tile loop count → safe barriers
if (lid == 0) { tgRefX[0] = agCellX; tgRefY[0] = agCellY; }
threadgroup_barrier(mem_flags::mem_threadgroup);
int refCellX = tgRefX[0];  // all threads read the same value
```

**When the approximation kicks in:**

At low agent counts (< ~25K), cells have fewer than 64 agents and a threadgroup may span 2 cells. Thread 0's reference cell may differ from a boundary thread's actual cell, so the boundary thread scans thread 0's 3×3 neighborhood instead of its own. The distance check still rejects truly out-of-range agents (correctness is preserved); only a few distant neighbors in the non-reference cells are missed. At 100K agents this never occurs.

**Threadgroup memory footprint:**

| Kernel        | Arrays                   | Size per threadgroup |
| ------------- | ------------------------ | -------------------- |
| `k_steerGrid` | posX/Y + velX/Y + maxSpd | 1,288 B              |
| `k_orca`      | posX/Y + velX/Y          | 1,032 B              |

Both are well under the 32 KB Metal threadgroup memory limit, leaving headroom for the compiler to allocate additional TGSM for other variables.

**Benchmark results (Apple M3):**

```
                               1K     5K    10K    50K   100K   250K   500K
Spat.Hash  No Tile [Ph.3]    0.60  0.61  0.79  1.94   6.47  37.4   147
Spat.Hash  Tiled   [Ph.3+7]  0.23  0.25  0.31  1.93   6.59  38.5   153

Tiling speedup:              2.61× 2.41× 2.56× 1.00×  0.98× 0.97×  0.96×
```

_Analysis:_ The result contradicts the CUDA intuition that tiling always wins, and the reason is architectural.

**Why tiling is 2.5× faster at 1K–10K:** With very few agents the working set is tiny and cells are sparsely occupied. Each threadgroup's 64 threads have different neighbors in different grid cells — the untiled path scatters reads across device memory with no spatial locality. The tiled kernel eliminates this by having each thread issue only one global load per tile iteration (its own lane) while the rest of the neighborhood comes from fast TGSM. The cooperative load pattern also improves SIMD occupancy: every lane does identical work on the same tile, so no thread stalls waiting for a divergent neighbor list to finish.

**Why tiling breaks even or regresses at 50K–500K:** The M3's GPU L2 cache is large relative to the working set at these densities. The `k_reorder` pass (Phase 3) already writes agent data in sorted-by-cell order, so every subsequent global read during the 3×3 neighborhood scan hits a nearly sequential address range — exactly the access pattern L2 prefetchers are designed to exploit. By the time the tiled kernel calls `threadgroup_barrier`, the untiled kernel has already loaded the same data from L2 at comparable bandwidth. The barrier overhead (~few μs per tile boundary) then adds up across the grid, producing the observed ~3% regression at 500K.

**The underlying lesson:** Tiling exists to paper over the gap between register/shared memory bandwidth and off-chip DRAM bandwidth on discrete GPUs. On Apple Silicon the unified memory architecture blurs this boundary — the GPU L2 is physically adjacent to the compute cores and large enough (relative to typical working sets) to capture the reuse that TGSM would otherwise provide. The `k_reorder` sort doubles down on this: it transforms scattered reads into sequential ones, making L2 prefetching nearly perfect. TGSM tiling is still beneficial when the working set genuinely overflows L2 (sub-10K agents in this sim, or arbitrarily large neighborhoods in other workloads), but it is not a universal win on this hardware.

---

## Project Structure

```
CrowdSim/
├── CMakeLists.txt          — Build system
├── src/
│   ├── Agent.h             — SoA agent buffer layout
│   ├── SharedTypes.h       — SimParams struct (shared by C++ and Metal)
│   ├── Simulation.h/.cpp   — CPU steering loop (Phase 1 / Phase 4 benchmarking)
│   ├── GPUSimulation.h/.mm — Metal buffers, k_steer + k_integrate pipelines
│   ├── main.mm             — App entry point
│   ├── AppDelegate.h/.mm   — Window, MTKView, GPUSimulation setup
│   └── Renderer.h/.mm      — CPU + GPU render paths, instanced draw
├── shaders/
│   └── Shaders.metal       — vs_agent / fs_agent, k_steer, k_integrate
└── .vscode/
    ├── tasks.json          — Build / Run tasks
    ├── launch.json         — LLDB debug configuration
    ├── c_cpp_properties.json
    └── settings.json
```

---

## Build

**Prerequisites**

- Xcode (from the App Store) — required for the `metal` shader compiler
- CMake — `brew install cmake`

After installing Xcode, point the toolchain at it:

```bash
sudo xcode-select -s /Applications/Xcode.app
```

**Configure and build**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel
```

Or in VS Code: `Cmd+Shift+B`

**Run**

```bash
# Default — 100 000 agents
open build/CrowdSim.app

# Custom agent count — run the binary directly (open does not forward flags)
build/CrowdSim.app/Contents/MacOS/CrowdSim --agents 250000
build/CrowdSim.app/Contents/MacOS/CrowdSim --agents 10000
```

Or in VS Code: `F5` to build and launch under LLDB (edit `launch.json` `args` to pass `--agents`).

**Benchmark suite**

```bash
build/CrowdSim.app/Contents/MacOS/CrowdSim --benchmark
```

Runs all 6 scenarios (CPU 1/2/4/8 threads, GPU O(N²), GPU Spatial Hash) across 6 agent counts and prints a formatted results table.

**Controls**

| Input           | Action                                                                                  |
| --------------- | --------------------------------------------------------------------------------------- |
| `1` / `2` / `3` | Switch scene (Open Field / Barrier / Pillars) — also clears drawn obstacles             |
| Left-click drag | Draw obstacle segments freehand; agents avoid them immediately                          |
| Right-click     | Place / move the flow-field goal (also enables flow-field mode if not already active)   |
| `F`             | Toggle flow-field navigation (agents seek a shared goal vs. random wandering)           |
| `O`             | Toggle ORCA collision avoidance (replaces heuristic separation with LP-based avoidance) |
| `C`             | Clear drawn segments, restore current scene's preset obstacles                          |

---

## Project Outline

| Phase   | Focus                                    | Agent Target   | Status   |
| ------- | ---------------------------------------- | -------------- | -------- |
| 0       | Project setup, render loop, build system | —              | Complete |
| 1       | CPU prototype, steering behaviors        | 10K @ 60 FPS   | Complete |
| 2       | GPU compute port (Metal)                 | 50K @ 60 FPS   | Complete |
| 3       | Spatial hashing + GPU neighbor search    | 100K @ 60 FPS  | Complete |
| 4       | CPU vs GPU benchmarking suite            | 100K+          | Complete |
| 5       | Obstacle avoidance + crowd scenarios     | 100K @ 60 FPS  | Complete |
| 6       | Flow fields and ORCA navigation          | 250K+ @ 60 FPS | Complete |
| Stretch | 500K+ agents, GPU profiling dashboard    | 500K+ @ 60 FPS | Planned  |

---

## Stretch Goal — Massive Scale and Profiling

**Target: 500K+ agents at 60 FPS.**

Focus areas: memory bandwidth, GPU occupancy, workgroup sizing, cache efficiency.

**GPU profiling overlay:**

```
FPS            | 60
Frame time     | 16.7 ms
Agent count    | 500,000
Compute time   | 11.2 ms
  Grid build   |  1.1 ms
  Neighbor     |  3.4 ms
  Steering     |  4.8 ms
  Integration  |  1.9 ms
Render time    |  4.1 ms
```

Timing captured with `MTLCommandBuffer` completion handlers and a ring buffer of frame samples.

**Long-term goal:** Build a simulation engine combining parallel computing, GPU programming, performance engineering, AI navigation, real-time rendering, and large-scale systems optimization.

---

## Stretch Goal 2 - Stadium / Building Evacuation Safety Simulator

- Allow user to build out 2D maps with obstacle walls, set target entry/exit points, and let the simulation run with X agents. To get 3D representation, they can build multiple 2D laers

---

## Success Criteria

- 100,000+ agents simulated in real time at 60 FPS
- Neighbor search uses spatial hashing (not O(N²) scan)
- Simulation runs entirely on the GPU after initial buffer upload
- At least two crowd scenarios are demonstrable
- Frame time and per-pass compute time are measurable and reproducible
