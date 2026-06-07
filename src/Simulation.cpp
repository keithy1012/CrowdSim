#include "Simulation.h"
#include <cmath>
#include <algorithm>

static constexpr float kWeightSeek   = 1.0f;
static constexpr float kWeightSep    = 2.0f;
static constexpr float kWeightAlign  = 0.4f;
static constexpr float kWeightCohere = 0.2f;

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
    _agents.targetX[i] = rx(_rng);
    _agents.targetY[i] = ry(_rng);
}

void Simulation::update(float dt) {
    const uint32_t n   = _agents.count();
    const float    nr2 = kNeighborRadius  * kNeighborRadius;
    const float    sr2 = kSeparationRadius * kSeparationRadius;

    // --- Pass 1: accumulate forces (reads previous-frame state, no writes to positions/velocities) ---
    for (uint32_t i = 0; i < n; i++) {
        const float px = _agents.posX[i];
        const float py = _agents.posY[i];
        const float ms = _agents.maxSpeed[i];

        float forceX = 0.f, forceY = 0.f;

        // Goal seeking
        float gdx = _agents.targetX[i] - px;
        float gdy = _agents.targetY[i] - py;
        float gd2 = gdx * gdx + gdy * gdy;

        if (gd2 < kArrivalRadius * kArrivalRadius) {
            randomizeTarget(i);
        } else {
            float inv = ms / std::sqrt(gd2);
            forceX += kWeightSeek * gdx * inv;
            forceY += kWeightSeek * gdy * inv;
        }

        // Neighborhood scan (O(N²))
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

            // Separation — inversely proportional to distance
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

            // Alignment: match average neighbor velocity
            forceX += kWeightAlign * aliVX * inv;
            forceY += kWeightAlign * aliVY * inv;

            // Cohesion: steer toward centroid
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

    // --- Pass 2: integrate velocity then position ---
    for (uint32_t i = 0; i < n; i++) {
        _agents.velX[i] += _fx[i] * dt;
        _agents.velY[i] += _fy[i] * dt;

        // Clamp to max speed
        float spd = std::sqrt(_agents.velX[i] * _agents.velX[i] +
                               _agents.velY[i] * _agents.velY[i]);
        if (spd > _agents.maxSpeed[i]) {
            float s = _agents.maxSpeed[i] / spd;
            _agents.velX[i] *= s;
            _agents.velY[i] *= s;
        }

        _agents.posX[i] += _agents.velX[i] * dt;
        _agents.posY[i] += _agents.velY[i] * dt;

        // Wrap at world edges
        _agents.posX[i] = std::fmod(_agents.posX[i] + kWorldWidth,  kWorldWidth);
        _agents.posY[i] = std::fmod(_agents.posY[i] + kWorldHeight, kWorldHeight);
    }
}
