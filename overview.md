# CrowdSim

## GPU-Accelerated Crowd Simulation Engine

### Author

Keith Yao

### Status

Design Phase

### Target Platform

- Apple Silicon (M2)
- macOS
- Metal
- C++20
- VS Code
- CMake

---

# 1. Project Overview

CrowdSim is a real-time GPU-accelerated crowd simulation engine capable of simulating tens to hundreds of thousands of autonomous agents.

The project explores:

- Large-scale agent simulation
- Parallel computing
- Spatial partitioning
- Steering behaviors
- GPU optimization
- Real-time rendering

The system will be implemented using Metal Compute Shaders on Apple Silicon and designed to scale from 10,000 agents to over 500,000 agents while maintaining interactive frame rates.

---

# 2. Objectives

## Functional Objectives

- Simulate large crowds of agents
- Support goal-directed movement
- Implement collision avoidance
- Support static obstacles
- Render all agents in real time
- Allow runtime parameter tuning

## Technical Objectives

- Learn GPU programming using Metal
- Implement spatial hashing
- Optimize memory access patterns
- Explore parallel algorithms
- Measure performance at scale

---

# 3. System Architecture

```text
+---------------------+
|      UI Layer       |
+----------+----------+
           |
           v
+---------------------+
|  Simulation Manager |
+----------+----------+
           |
           v
+---------------------+
|   Crowd Simulator   |
+----------+----------+
           |
           v
+---------------------+
| Spatial Hash Grid   |
+----------+----------+
           |
           v
+---------------------+
| Metal Compute GPU   |
+----------+----------+
           |
           v
+---------------------+
| Rendering Pipeline  |
+---------------------+
```

---

# 4. Core Simulation Model

Each agent represents an autonomous entity moving through the environment.

## Agent Data Structure

```cpp
struct Agent
{
    float2 position;
    float2 velocity;

    float2 target;

    float radius;
    float maxSpeed;
};
```

Each agent attempts to:

1. Move toward its target
2. Avoid nearby agents
3. Avoid obstacles
4. Respect movement constraints

---

# 5. Simulation Loop

Each frame:

```text
Input
  ↓
Build Spatial Grid
  ↓
Neighbor Search
  ↓
Steering Computation
  ↓
Velocity Update
  ↓
Position Integration
  ↓
Rendering
```

Target frame rate:

```text
60 FPS
```

---

# 6. Steering Behaviors

## Goal Seeking

Agents move toward a destination.

```text
desiredDirection =
targetPosition - currentPosition
```

Normalized direction becomes the preferred velocity.

---

## Separation

Agents avoid nearby neighbors.

```text
Move away from nearby agents
```

Prevents overlapping.

---

## Alignment

Agents align movement direction with local neighbors.

Produces more natural group motion.

---

## Cohesion

Agents attempt to stay near nearby group members.

Creates crowd formations.

---

# 7. Spatial Hash Grid

## Problem

Naive neighbor search:

```text
For every agent:
    Compare against every agent
```

Complexity:

```text
O(N²)
```

At 100,000 agents:

```text
10 billion comparisons
```

Not feasible.

---

## Solution

Uniform Spatial Grid

World divided into cells:

```text
+----+----+----+
| A  | B  | C  |
+----+----+----+
| D  | E  | F  |
+----+----+----+
```

Each agent is assigned to a cell.

Neighbor search checks:

- Current cell
- Adjacent cells

instead of the entire simulation.

---

## Expected Complexity

Approximately:

```text
O(N)
```

for practical crowd densities.

---

# 8. GPU Compute Design

## Compute Model

Each GPU thread processes one agent.

```text
Thread 0 -> Agent 0
Thread 1 -> Agent 1
Thread 2 -> Agent 2
...
```

Massively parallel workload.

---

## Compute Passes

### Pass 1

Build Spatial Grid

Responsibilities:

- Compute cell IDs
- Assign agents to cells

---

### Pass 2

Neighbor Search

Responsibilities:

- Find nearby agents
- Build local interaction lists

---

### Pass 3

Steering Update

Responsibilities:

- Goal seeking
- Separation
- Alignment
- Cohesion

---

### Pass 4

Physics Integration

Responsibilities:

```cpp
position += velocity * deltaTime;
```

---

# 9. Memory Layout

## Structure of Arrays (SoA)

Avoid:

```cpp
Agent agents[];
```

Use:

```cpp
float posX[];
float posY[];

float velX[];
float velY[];

float targetX[];
float targetY[];
```

Benefits:

- Better cache behavior
- Improved memory coalescing
- Better GPU utilization

---

# 10. Rendering System

## Rendering Method

Metal Render Pipeline

Agents rendered using:

```text
Instanced Rendering
```

Each agent represented by:

- Circle
- Triangle
- Arrow

depending on visualization mode.

---

## Visualization Modes

### Simple

White circles.

---

### Velocity Mode

Color based on speed.

---

### Density Mode

Color based on local crowd density.

---

### Flow Mode

Display crowd movement patterns.

---

# 11. Obstacle System

Environment contains:

- Walls
- Buildings
- Barriers

Obstacle data stored in GPU buffers.

Agents:

```text
Detect obstacle
      ↓
Compute avoidance force
      ↓
Adjust trajectory
```

---

# 12. Scenarios

## Scenario 1: Stadium Evacuation

Agents attempt to reach exits.

Metrics:

- Exit throughput
- Bottlenecks
- Congestion

---

## Scenario 2: City Pedestrians

Agents navigate around buildings.

Features:

- Obstacle avoidance
- Traffic flow visualization

---

## Scenario 3: RTS Army Simulation

Large unit formations.

Features:

- Group movement
- Formation maintenance

---

# 13. Performance Targets

## Milestone 1

Basic CPU Prototype

```text
10,000 agents
60 FPS
```

---

## Milestone 2

Metal Compute Implementation

```text
50,000 agents
60 FPS
```

---

## Milestone 3

Spatial Hashing

```text
100,000 agents
60 FPS
```

---

## Milestone 4

Obstacle Avoidance

```text
250,000 agents
60 FPS
```

---

## Stretch Goal

```text
500,000+ agents
60 FPS
```

---

# 14. Future Enhancements

## Flow Fields

Replace per-agent pathfinding with global vector fields.

Benefits:

- Better scalability
- RTS-style movement

---

## ORCA

Optimal Reciprocal Collision Avoidance

Features:

- Predictive collision avoidance
- Robotics-grade navigation

---

## A\* Navigation

Global path planning around obstacles.

---

## Multi-Team Simulation

Different groups with competing goals.

---

## GPU Profiling Dashboard

Display:

- FPS
- Frame time
- Agent count
- Compute time
- Neighbor search time

---

# 15. Success Criteria

The project is considered successful when:

- 100,000+ agents are simulated in real time
- Neighbor search uses spatial hashing
- Simulation runs primarily on the GPU
- Multiple crowd scenarios are supported
- Performance metrics are measurable and reproducible

---

# 16. Resume Description

Built CrowdSim, a GPU-accelerated crowd simulation engine using C++ and Metal on Apple Silicon. Implemented large-scale autonomous agent simulation with spatial hashing, parallel neighbor search, steering behaviors, and instanced rendering. Scaled simulation performance from 10,000 to 100,000+ agents while maintaining interactive frame rates through GPU-based optimization and efficient memory layouts.
