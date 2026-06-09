#include "Simulation.h"
#include <cmath>
#include <algorithm>
#include <thread>

static constexpr float kWeightSeek   = 1.0f;
static constexpr float kWeightSep    = 2.0f;
static constexpr float kWeightAlign  = 0.4f;
static constexpr float kWeightCohere = 0.2f;

// Each thread has its own RNG so randomizeTarget is race-free in the MT path.
static thread_local std::mt19937 s_tlRng(std::random_device{}());

Simulation::Simulation(uint32_t agentCount) : _rng(42) {
    _agents.resize(agentCount);
    _fx.resize(agentCount);
    _fy.resize(agentCount);

    std::uniform_real_distribution<float> rx(0.f, kWorldWidth);
    std::uniform_real_distribution<float> ry(0.f, kWorldHeight);
    std::uniform_real_distribution<float> spd(80.f, 120.f);

    for (uint32_t i = 0; i < agentCount; i++) {
        _agents.posX[i]     = rx(_rng);
        _agents.posY[i]     = ry(_rng);
        _agents.maxSpeed[i] = spd(_rng);
        _agents.radius[i]   = 5.f;
        randomizeTarget(i);
    }
}

void Simulation::randomizeTarget(uint32_t i) {
    std::uniform_real_distribution<float> rx(0.f, kWorldWidth);
    std::uniform_real_distribution<float> ry(0.f, kWorldHeight);
    _agents.targetX[i] = rx(s_tlRng);
    _agents.targetY[i] = ry(s_tlRng);
}

// ── Private helpers ───────────────────────────────────────────────────────────

void Simulation::accumulateForces(uint32_t lo, uint32_t hi) {
    const uint32_t n   = _agents.count();
    const float    nr2 = kNeighborRadius   * kNeighborRadius;
    const float    sr2 = kSeparationRadius * kSeparationRadius;
    const float    ar2 = kArrivalRadius    * kArrivalRadius;

    for (uint32_t i = lo; i < hi; i++) {
        const float px = _agents.posX[i];
        const float py = _agents.posY[i];
        const float ms = _agents.maxSpeed[i];

        float forceX = 0.f, forceY = 0.f;

        float gdx = _agents.targetX[i] - px;
        float gdy = _agents.targetY[i] - py;
        float gd2 = gdx * gdx + gdy * gdy;

        if (gd2 < ar2) {
            randomizeTarget(i);
        } else {
            float inv = ms / std::sqrt(gd2);
            forceX += kWeightSeek * gdx * inv;
            forceY += kWeightSeek * gdy * inv;
        }

        float sepX = 0.f, sepY = 0.f;
        float aliVX = 0.f, aliVY = 0.f;
        float cohX = 0.f, cohY = 0.f;
        int   count = 0;

        for (uint32_t j = 0; j < n; j++) {
            if (j == i) continue;
            float dx  = _agents.posX[j] - px;
            float dy  = _agents.posY[j] - py;
            float nd2 = dx * dx + dy * dy;
            if (nd2 > nr2) continue;

            float nd = std::sqrt(nd2);

            if (nd2 < sr2 && nd > 0.001f) {
                float strength = (kSeparationRadius - nd) / kSeparationRadius;
                sepX -= (dx / nd) * strength;
                sepY -= (dy / nd) * strength;
            }

            aliVX += _agents.velX[j];
            aliVY += _agents.velY[j];
            cohX  += _agents.posX[j];
            cohY  += _agents.posY[j];
            count++;
        }

        forceX += kWeightSep * sepX;
        forceY += kWeightSep * sepY;

        if (count > 0) {
            float inv = 1.f / count;
            forceX += kWeightAlign * aliVX * inv;
            forceY += kWeightAlign * aliVY * inv;

            float cdx = cohX * inv - px;
            float cdy = cohY * inv - py;
            float cd  = std::sqrt(cdx * cdx + cdy * cdy);
            if (cd > 0.001f) {
                forceX += kWeightCohere * (cdx / cd) * ms;
                forceY += kWeightCohere * (cdy / cd) * ms;
            }
        }

        _fx[i] = forceX;
        _fy[i] = forceY;
    }
}

void Simulation::integrateSlice(uint32_t lo, uint32_t hi, float dt) {
    for (uint32_t i = lo; i < hi; i++) {
        _agents.velX[i] += _fx[i] * dt;
        _agents.velY[i] += _fy[i] * dt;

        float spd = std::sqrt(_agents.velX[i] * _agents.velX[i] +
                               _agents.velY[i] * _agents.velY[i]);
        if (spd > _agents.maxSpeed[i]) {
            float s = _agents.maxSpeed[i] / spd;
            _agents.velX[i] *= s;
            _agents.velY[i] *= s;
        }

        _agents.posX[i] += _agents.velX[i] * dt;
        _agents.posY[i] += _agents.velY[i] * dt;

        _agents.posX[i] = std::fmod(_agents.posX[i] + kWorldWidth,  kWorldWidth);
        _agents.posY[i] = std::fmod(_agents.posY[i] + kWorldHeight, kWorldHeight);
    }
}

// ── Public update ─────────────────────────────────────────────────────────────

void Simulation::update(float dt) {
    const uint32_t n = _agents.count();
    accumulateForces(0, n);
    integrateSlice(0, n, dt);
}

void Simulation::update(float dt, uint32_t numThreads) {
    if (numThreads <= 1) { update(dt); return; }

    const uint32_t n         = _agents.count();
    const uint32_t slice     = (n + numThreads - 1) / numThreads;

    // Force pass — threads write to disjoint _fx/_fy ranges, all read shared posX/Y
    {
        std::vector<std::thread> workers;
        for (uint32_t t = 1; t < numThreads; t++) {
            uint32_t lo = t * slice;
            uint32_t hi = std::min(lo + slice, n);
            workers.emplace_back([this, lo, hi]{ accumulateForces(lo, hi); });
        }
        accumulateForces(0, std::min(slice, n));
        for (auto& w : workers) w.join();
    }

    // Integrate pass — each thread owns its slice exclusively
    {
        std::vector<std::thread> workers;
        for (uint32_t t = 1; t < numThreads; t++) {
            uint32_t lo = t * slice;
            uint32_t hi = std::min(lo + slice, n);
            workers.emplace_back([this, lo, hi, dt]{ integrateSlice(lo, hi, dt); });
        }
        integrateSlice(0, std::min(slice, n), dt);
        for (auto& w : workers) w.join();
    }
}
