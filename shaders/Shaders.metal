#include <metal_stdlib>
#include "SharedTypes.h"
using namespace metal;

// ── Shared helpers ────────────────────────────────────────────────────────────

// Wang hash — fast GPU random number from a seed
static uint wang_hash(uint s) {
    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s ^= s >> 4u;
    s *= 0x27d4eb2du;
    s ^= s >> 15u;
    return s;
}

// ── Phase 1 / 2: instanced circle rendering ───────────────────────────────────

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Buffer layout (matches Renderer.mm):
//   0 — float *posX   (per-instance X)
//   1 — float *posY   (per-instance Y)
//   2 — float *rad    (per-instance radius)
//   3 — float2 vp     (viewport size, same for all vertices)
vertex VertexOut vs_agent(
    uint               vid  [[vertex_id]],
    uint               iid  [[instance_id]],
    device const float *posX [[buffer(0)]],
    device const float *posY [[buffer(1)]],
    device const float *rad  [[buffer(2)]],
    constant float2   &vp   [[buffer(3)]]
) {
    float2 offsets[4] = {
        float2(-1.f, -1.f),
        float2(-1.f,  1.f),
        float2( 1.f, -1.f),
        float2( 1.f,  1.f)
    };

    float2 offset   = offsets[vid];
    float2 worldPos = float2(posX[iid] + offset.x * rad[iid],
                             posY[iid] + offset.y * rad[iid]);

    // World [0, vp] → NDC [-1, 1]; flip Y (world Y down, NDC Y up)
    float2 ndc = (worldPos / vp) * 2.f - 1.f;
    ndc.y      = -ndc.y;

    VertexOut out;
    out.position = float4(ndc, 0.f, 1.f);
    out.uv       = offset;
    return out;
}

fragment float4 fs_agent(VertexOut in [[stage_in]]) {
    if (length(in.uv) > 1.f) discard_fragment();
    return float4(1.f, 1.f, 1.f, 1.f);
}

// ── Phase 2: GPU compute kernels ─────────────────────────────────────────────

// Pass 1 — Steering
// Reads positions/velocities from the previous frame (no writes to pos/vel).
// Writes accumulated forces to forceX/forceY.
// Reassigns targets on arrival using a deterministic per-agent hash.
//
// Buffer layout:
//   0 posX (r)   1 posY (r)   2 velX (r)   3 velY (r)
//   4 targetX (rw)  5 targetY (rw)
//   6 maxSpeed (r)  7 forceX (w)  8 forceY (w)  9 SimParams
kernel void k_steer(
    device const float  *posX     [[buffer(0)]],
    device const float  *posY     [[buffer(1)]],
    device const float  *velX     [[buffer(2)]],
    device const float  *velY     [[buffer(3)]],
    device float        *targetX  [[buffer(4)]],
    device float        *targetY  [[buffer(5)]],
    device const float  *maxSpeed [[buffer(6)]],
    device float        *forceX   [[buffer(7)]],
    device float        *forceY   [[buffer(8)]],
    constant SimParams  &p        [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;

    const float px = posX[gid], py = posY[gid];
    const float ms = maxSpeed[gid];

    float fx = 0.f, fy = 0.f;

    // Goal seeking
    float gdx = targetX[gid] - px;
    float gdy = targetY[gid] - py;
    float gd2 = gdx * gdx + gdy * gdy;

    if (gd2 < p.arrivalRadius2) {
        // Reassign target — deterministic hash seeded by agent id + frame
        uint seed  = wang_hash(gid ^ (uint)(p.frameIndex) * 2654435761u);
        targetX[gid] = (float)(seed & 0xFFFFu) / 65535.f * p.worldWidth;
        seed         = wang_hash(seed);
        targetY[gid] = (float)(seed & 0xFFFFu) / 65535.f * p.worldHeight;
    } else {
        float inv = ms / sqrt(gd2);
        fx += p.weightSeek * gdx * inv;
        fy += p.weightSeek * gdy * inv;
    }

    // O(N²) neighbourhood scan — replaced by spatial hash in Phase 3
    float sepX = 0.f, sepY = 0.f;
    float aliVX = 0.f, aliVY = 0.f;
    float cohX = 0.f, cohY = 0.f;
    int   count = 0;

    for (int j = 0; j < p.agentCount; j++) {
        if (j == (int)gid) continue;
        float dx  = posX[j] - px;
        float dy  = posY[j] - py;
        float nd2 = dx * dx + dy * dy;
        if (nd2 > p.neighborRadius2) continue;

        float nd = sqrt(nd2);

        if (nd2 < p.separationRadius2 && nd > 0.001f) {
            float strength = (p.separationRadius - nd) / p.separationRadius;
            sepX -= (dx / nd) * strength;
            sepY -= (dy / nd) * strength;
        }

        aliVX += velX[j];
        aliVY += velY[j];
        cohX  += posX[j];
        cohY  += posY[j];
        count++;
    }

    fx += p.weightSep * sepX;
    fy += p.weightSep * sepY;

    if (count > 0) {
        float inv = 1.f / (float)count;
        fx += p.weightAlign * aliVX * inv;
        fy += p.weightAlign * aliVY * inv;

        float cdx = cohX * inv - px;
        float cdy = cohY * inv - py;
        float cd  = length(float2(cdx, cdy));
        if (cd > 0.001f) {
            fx += p.weightCohere * (cdx / cd) * ms;
            fy += p.weightCohere * (cdy / cd) * ms;
        }
    }

    forceX[gid] = fx;
    forceY[gid] = fy;
}

// Pass 2 — Physics integration
// Reads forceX/Y, updates velX/Y then posX/Y in place.
//
// Buffer layout:
//   0 posX (rw)  1 posY (rw)  2 velX (rw)  3 velY (rw)
//   4 forceX (r)  5 forceY (r)  6 maxSpeed (r)  7 SimParams
kernel void k_integrate(
    device float        *posX     [[buffer(0)]],
    device float        *posY     [[buffer(1)]],
    device float        *velX     [[buffer(2)]],
    device float        *velY     [[buffer(3)]],
    device const float  *forceX   [[buffer(4)]],
    device const float  *forceY   [[buffer(5)]],
    device const float  *maxSpeed [[buffer(6)]],
    constant SimParams  &p        [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;

    float vx = velX[gid] + forceX[gid] * p.dt;
    float vy = velY[gid] + forceY[gid] * p.dt;

    // Clamp to max speed
    float spd = length(float2(vx, vy));
    if (spd > maxSpeed[gid]) {
        float s = maxSpeed[gid] / spd;
        vx *= s;
        vy *= s;
    }

    velX[gid] = vx;
    velY[gid] = vy;

    // Integrate position with world wrapping
    float nx = posX[gid] + vx * p.dt;
    float ny = posY[gid] + vy * p.dt;
    posX[gid] = nx - p.worldWidth  * floor(nx / p.worldWidth);
    posY[gid] = ny - p.worldHeight * floor(ny / p.worldHeight);
}
