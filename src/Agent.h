#pragma once
#include <vector>
#include <cstdint>

// Structure of Arrays layout — better GPU cache coherence than AoS
struct AgentBuffers {
    std::vector<float> posX, posY;
    std::vector<float> velX, velY;
    std::vector<float> targetX, targetY;
    std::vector<float> radius;
    std::vector<float> maxSpeed;

    void resize(uint32_t count) {
        posX.assign(count, 0.f);
        posY.assign(count, 0.f);
        velX.assign(count, 0.f);
        velY.assign(count, 0.f);
        targetX.assign(count, 0.f);
        targetY.assign(count, 0.f);
        radius.assign(count, 5.f);
        maxSpeed.assign(count, 100.f);
    }

    uint32_t count() const { return static_cast<uint32_t>(posX.size()); }
};
