#pragma once
#include "Agent.h"
#include <random>

class Simulation {
public:
    static constexpr float kWorldWidth       = 1280.f;
    static constexpr float kWorldHeight      = 720.f;
    static constexpr float kNeighborRadius   = 50.f;
    static constexpr float kSeparationRadius = 12.f;
    static constexpr float kArrivalRadius    = 20.f;

    explicit Simulation(uint32_t agentCount);
    void update(float dt);
    void update(float dt, uint32_t numThreads);  // parallel variant (Phase 4)

    const AgentBuffers& agents() const { return _agents; }

private:
    AgentBuffers        _agents;
    std::vector<float>  _fx, _fy;
    std::mt19937        _rng;

    void randomizeTarget(uint32_t i);
    void accumulateForces(uint32_t lo, uint32_t hi);
    void integrateSlice(uint32_t lo, uint32_t hi, float dt);
};
