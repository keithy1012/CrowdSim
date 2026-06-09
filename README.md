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
open build/CrowdSim.app
```

Or in VS Code: `F5` to build and launch under LLDB.

---

## Project Outline

| Phase   | Focus                                    | Agent Target   | Status   |
| ------- | ---------------------------------------- | -------------- | -------- |
| 0       | Project setup, render loop, build system | —              | Complete |
| 1       | CPU prototype, steering behaviors        | 10K @ 60 FPS   | Complete |
| 2       | GPU compute port (Metal)                 | 50K @ 60 FPS   | Complete |
| 3       | Spatial hashing + GPU neighbor search    | 100K @ 60 FPS  | Complete |
| 4       | CPU vs GPU benchmarking suite            | 100K+          | Planned  |
| 5       | Obstacle avoidance + crowd scenarios     | 250K @ 60 FPS  | Planned  |
| 6       | Flow fields and ORCA navigation          | 250K+ @ 60 FPS | Planned  |
| Stretch | 500K+ agents, GPU profiling dashboard    | 500K+ @ 60 FPS | Planned  |

---

## Phase 4 — Performance Engineering and Benchmarking

**Goal:** Quantify scalability and understand performance bottlenecks.

Most simulation projects stop after getting something working. CrowdSim includes a dedicated benchmarking phase to measure how different implementations scale.

**Implementations compared:**

_CPU — Single Thread_

```
Agent Update → O(N²) Neighbor Search → Steering → Integration
```

_CPU — Multi-Threaded_

Workload divided across worker threads:

```
Thread 1 → Agents 0–9,999
Thread 2 → Agents 10,000–19,999
...
```

_GPU — Metal_

One GPU thread per agent:

```
Thread 0 → Agent 0
Thread 1 → Agent 1
...
```

**Metrics collected:**

- FPS and frame time
- Steering pass time
- Neighbor search time
- GPU compute time
- Memory consumption

**Output:** Benchmark reports and scaling curves.

| Agents | CPU    | GPU    |
| ------ | ------ | ------ |
| 10K    | 60 FPS | 60 FPS |
| 50K    | 18 FPS | 60 FPS |
| 100K   | 7 FPS  | 60 FPS |
| 250K   | 2 FPS  | 51 FPS |

**Goal:** Demonstrate measurable performance gains from GPU acceleration and algorithmic optimization.

---

## Phase 5 — Obstacle Avoidance and Crowd Scenarios

**Goal:** Add static obstacles and build three demonstration scenarios.

**Obstacle representation:**

```cpp
struct Obstacle {
    float2 start;
    float2 end;
    float2 normal;
};
```

Static obstacles stored in a GPU buffer, uploaded once at scene load.

**Avoidance in the steering pass:**

```
Detect obstacle
    ↓
Compute closest point on segment to agent
    ↓
Generate avoidance force along normal
    ↓
Adjust velocity
```

**Scenarios:**

- **Stadium Evacuation** — large crowd exits through constrained bottlenecks. Measures throughput, congestion, and crowd density.
- **City Pedestrians** — agents navigate around buildings and intersections. Measures traffic flow, congestion zones, and route efficiency.
- **RTS Army** — large formations move toward objectives. Measures formation integrity, group cohesion, and scalability.

**Visualization modes:**

| Mode     | Color encoding              |
| -------- | --------------------------- |
| Simple   | White circles               |
| Velocity | Hue maps to speed           |
| Density  | Red = high local density    |
| Flow     | Direction vectors per agent |

**Target:** 250,000 agents at 60 FPS.

---

## Phase 6 — Advanced Crowd Navigation

**Goal:** Replace simple steering with techniques used in modern games, robotics, and crowd simulation research.

**Flow fields** replace per-agent goal-seeking with a global vector field precomputed over the grid:

```
Destination
    ↓
Global Vector Field
    ↓
Thousands of Agents sample nearest cell → O(1) path lookup
```

Benefits: extremely scalable, common in RTS games, no per-agent pathfinding cost.

**ORCA (Optimal Reciprocal Collision Avoidance)** replaces heuristic separation with predictive collision avoidance:

```
Predict future collisions
    ↓
Construct velocity constraints (half-planes)
    ↓
Solve local linear program
    ↓
Select closest collision-free velocity
```

Benefits: deadlock reduction, more realistic crowd movement, robotics-grade navigation.

**Goal:** Demonstrate advanced multi-agent navigation beyond traditional boids-style steering.

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

- Allow user to build out 2D or 3D map with obstacle walls, set target entry/exit points, and let the simulation run with X agents.

---

## Success Criteria

- 100,000+ agents simulated in real time at 60 FPS
- Neighbor search uses spatial hashing (not O(N²) scan)
- Simulation runs entirely on the GPU after initial buffer upload
- At least two crowd scenarios are demonstrable
- Frame time and per-pass compute time are measurable and reproducible
