#pragma once

#define kMaxEvacExits 8

// Line-segment obstacle — stored in a GPU buffer for avoidance and rendered as lines.
struct Obstacle {
    float x0, y0;   // segment start
    float x1, y1;   // segment end
};

// Evacuation exit zone — agents within radius of (x,y) are removed from simulation.
struct EvacExit {
    float x, y, radius;
    float _pad;     // 16-byte alignment for Metal device buffers
};

// Plain-C struct — included by both C++ and Metal shaders.
// No C++ types, no STL, no Objective-C.
struct SimParams {
    int   agentCount;
    float dt;
    float worldWidth;
    float worldHeight;
    int   frameIndex;
    float neighborRadius2;
    float separationRadius;
    float separationRadius2;
    float arrivalRadius2;
    float weightSeek;
    float weightSep;
    float weightAlign;
    float weightCohere;
    // Phase 3 — spatial hash grid
    int   gridWidth;
    int   gridHeight;
    int   numCells;
    float cellSize;
    // Phase 5 — obstacle avoidance
    int   obstacleCount;
    float obstacleAvoidRadius;
    float obstacleAvoidRadius2;
    // Phase 6 — flow field navigation (0 = direct seek, 1 = sample flow field)
    int   useFlowField;
    // Phase 7 — ORCA velocity-space collision avoidance
    int   useORCA;           // 0 = flocking + repulsion, 1 = ORCA replaces separation
    float orcaTimeHorizon;   // seconds of lookahead for ORCA constraints (default 1.5)
    // Evacuation mode
    int   evacuationMode;    // 0 = normal, 1 = evacuation (no wrap, active flag, density speed)
    int   exitCount;         // number of active EvacExit entries
    float densityJamCount;   // neighbor count within densityRadius at which speed → 10% (default 10)
    float densityRadius2;    // squared radius for local density scan (default 30*30 = 900)
};
