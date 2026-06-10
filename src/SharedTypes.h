#pragma once

// Line-segment obstacle — stored in a GPU buffer for avoidance and rendered as lines.
struct Obstacle {
    float x0, y0;   // segment start
    float x1, y1;   // segment end
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
};
