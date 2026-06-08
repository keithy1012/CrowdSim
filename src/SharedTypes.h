#pragma once

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
};
