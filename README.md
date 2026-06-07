# CrowdSim

GPU-accelerated crowd simulation engine built with C++20 and Metal on Apple Silicon.

Simulates tens to hundreds of thousands of autonomous agents in real time using parallel GPU compute, spatial hashing, and steering behaviors.

---

## Platform

| | |
|---|---|
| Hardware | Apple Silicon (M2+) |
| OS | macOS |
| GPU API | Metal |
| Language | C++20 / Objective-C++ |
| Build | CMake 3.20+ |
| Editor | VS Code |

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

---

## Project Structure

```
CrowdSim/
├── CMakeLists.txt          — Build system
├── src/
│   ├── Agent.h             — SoA agent buffer layout
│   ├── Simulation.h/.cpp   — CPU steering loop (seek, separate, align, cohere)
│   ├── main.mm             — App entry point
│   ├── AppDelegate.h/.mm   — Window, MTKView, and Simulation setup
│   └── Renderer.h/.mm      — Metal pipeline, instanced draw, per-frame upload
├── shaders/
│   └── Shaders.metal       — vs_agent / fs_agent (circle instancing)
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

The simulation scales through four milestones, each adding a layer of GPU acceleration:

| Phase | Feature | Agent Target | Status |
|---|---|---|---|
| 0 | Project setup, window, render loop | — | Complete |
| 1 | CPU prototype, steering behaviors | 10K @ 60 FPS | Complete |
| 2 | Metal GPU port, 4 compute passes | 50K @ 60 FPS | Next |
| 3 | Spatial hash grid on GPU | 100K @ 60 FPS | |
| 4 | Obstacle avoidance, scenarios | 250K @ 60 FPS | |
| Stretch | Flow fields, ORCA, profiling dashboard | 500K+ @ 60 FPS | |

---

## Phase 2 — Metal GPU Port

**Goal:** Move the entire simulation loop to the GPU. Each Metal thread owns one agent.

**Four compute passes** per frame:

```
Pass 1 — Grid Build
    Each thread: hash agent position → cell ID
    Output: per-agent cell assignments

Pass 2 — Neighbor Search
    Each thread: scan current cell + 8 adjacent cells
    Output: per-agent neighbor list

Pass 3 — Steering Update
    Each thread: compute goal/separation/alignment/cohesion forces
    Output: new velocity per agent

Pass 4 — Physics Integration
    Each thread: position += velocity * deltaTime
    Output: new position per agent
```

**GPU buffer layout** mirrors the CPU SoA arrays. All six float arrays (`posX`, `posY`, `velX`, `velY`, `targetX`, `targetY`) become `MTLBuffer` objects. Updated once on spawn; read/write entirely on the GPU each frame.

**Rendering** switches to a GPU-driven instanced draw — positions are read directly from the position buffers already on the GPU, with no CPU readback.

**Target:** 50,000 agents at 60 FPS.

---

## Phase 3 — Spatial Hash Grid

**Goal:** Replace O(N²) neighbor search with O(N) spatial hashing on the GPU.

**Problem with Phase 2:** Pass 2 still scans all agents to find neighbors. At 50K agents that is 2.5 billion comparisons per frame.

**Solution:** Uniform grid. The world is divided into fixed-size cells. Each agent is mapped to exactly one cell by hashing its position:

```
cellX = floor(posX / cellSize)
cellY = floor(posY / cellSize)
cellID = cellX + cellY * gridWidth
```

Neighbor search then checks only the agent's cell and its 8 adjacent cells — typically a constant number of agents regardless of total crowd size.

**GPU implementation:**

1. Each thread writes its agent's `cellID` into a sort key buffer
2. GPU radix sort orders agents by cell (parallel prefix sum)
3. A second pass builds a `cellStart[]` and `cellEnd[]` lookup table
4. Neighbor search indexes into `cellStart[cellID]` to iterate only the relevant agents

**Memory layout:** The sorted index buffer avoids moving the SoA arrays. Each thread reads `sortedIndex[i]` to access the actual agent data.

**Cell size tuning:** Optimal cell size is roughly 2× the agent interaction radius — small enough to limit neighbors per cell, large enough that most agents have at least one neighbor.

**Target:** 100,000 agents at 60 FPS.

---

## Phase 4 — Obstacle Avoidance and Scenarios

**Goal:** Add static obstacles and build three demonstration scenarios.

**Obstacle representation:** Axis-aligned line segments stored in a GPU buffer. Each segment has a start point, end point, and a normal. The obstacle buffer is uploaded once at scene load and stays on the GPU.

**Avoidance in the steering pass:**

```
For each obstacle within detection range:
    Compute closest point on segment to agent
    If distance < avoidanceRadius:
        Add repulsion force along obstacle normal
```

The detection range is a compile-time constant so the shader loop is bounded.

**Scenarios:**

- **Stadium Evacuation** — agents distributed across a circular arena, multiple exit corridors. Measures exit throughput and congestion at bottlenecks.
- **City Pedestrians** — grid of building obstacles, agents navigating cross-traffic flows. Demonstrates obstacle avoidance at scale.
- **RTS Army** — two large formations moving toward each other. Demonstrates group cohesion and separation under high density.

**Visualization modes** (selectable at runtime):

| Mode | Color encoding |
|---|---|
| Simple | White circles |
| Velocity | Hue maps to speed |
| Density | Red = high local density |
| Flow | Arrow per agent showing direction |

**Target:** 250,000 agents at 60 FPS.

---

## Stretch Goal — Scale, Flow Fields, and Profiling

**500K+ agents at 60 FPS.**

**Flow fields** replace per-agent goal-seeking with a global vector field precomputed over the grid. Each agent samples the nearest field cell to get its desired direction. Cost is O(1) per agent at query time, and the field can be computed once for static goals.

**ORCA (Optimal Reciprocal Collision Avoidance)** replaces the separation/alignment heuristics with a provably deadlock-free velocity obstacle formulation. Each agent solves a small linear program to find the closest velocity that avoids all neighbors within a time horizon.

**GPU profiling overlay** rendered on top of the simulation:

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

Timing is captured with `MTLCommandBuffer` completion handlers and a ring buffer of frame samples.

---

## Success Criteria

- 100,000+ agents simulated in real time at 60 FPS
- Neighbor search uses spatial hashing (not O(N²) scan)
- Simulation runs entirely on the GPU after initial buffer upload
- At least two crowd scenarios are demonstrable
- Frame time and per-pass compute time are measurable and reproducible
