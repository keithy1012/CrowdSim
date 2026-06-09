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
};
