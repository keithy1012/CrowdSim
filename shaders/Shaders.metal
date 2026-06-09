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

// Compact HSV → RGB; h in [0,1], s/v in [0,1]
static float3 hsv2rgb(float h, float s, float v) {
    float4 K = float4(1.f, 2.f / 3.f, 1.f / 3.f, 3.f);
    float3 p = abs(fract(float3(h) + K.xyz) * 6.f - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.f, 1.f), s);
}

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float  speed [[flat]];  // normalised [0,1]; [[flat]] = not interpolated across quad
};

// Buffer layout (matches Renderer.mm):
//   0 — float *posX   (per-instance X)
//   1 — float *posY   (per-instance Y)
//   2 — float *rad    (per-instance radius)
//   3 — float2 vp     (viewport size)
//   4 — float *velX   (per-instance vel X, for speed colouring)
//   5 — float *velY   (per-instance vel Y)
//   6 — float *maxSpd (per-instance max speed)
vertex VertexOut vs_agent(
    uint               vid    [[vertex_id]],
    uint               iid    [[instance_id]],
    device const float *posX   [[buffer(0)]],
    device const float *posY   [[buffer(1)]],
    device const float *rad    [[buffer(2)]],
    constant float2   &vp     [[buffer(3)]],
    device const float *velX   [[buffer(4)]],
    device const float *velY   [[buffer(5)]],
    device const float *maxSpd [[buffer(6)]]
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
    out.speed    = saturate(length(float2(velX[iid], velY[iid])) /
                            max(maxSpd[iid], 0.001f));
    return out;
}

// Velocity colouring: slow → blue (hue 0.667), fast → red (hue 0)
fragment float4 fs_agent(VertexOut in [[stage_in]]) {
    if (length(in.uv) > 1.f) discard_fragment();
    float3 color = hsv2rgb((1.f - in.speed) * 0.667f, 0.85f, 1.f);
    return float4(color, 1.f);
}

// ── Phase 5: obstacle line rendering ─────────────────────────────────────────

struct ObstacleVert {
    float4 position [[position]];
};

// Buffer layout:
//   0 — Obstacle *obs  (x0,y0,x1,y1 per segment)
//   1 — float2 vp
vertex ObstacleVert vs_obstacle(
    uint                   vid [[vertex_id]],
    device const Obstacle *obs [[buffer(0)]],
    constant float2       &vp  [[buffer(1)]]
) {
    uint seg   = vid / 2;
    uint pt    = vid % 2;  // 0 = start, 1 = end
    float x    = pt == 0 ? obs[seg].x0 : obs[seg].x1;
    float y    = pt == 0 ? obs[seg].y0 : obs[seg].y1;
    float2 ndc = float2(x, y) / vp * 2.f - 1.f;
    ndc.y      = -ndc.y;
    ObstacleVert out;
    out.position = float4(ndc, 0.f, 1.f);
    return out;
}

fragment float4 fs_obstacle(ObstacleVert in [[stage_in]]) {
    return float4(0.85f, 0.85f, 0.85f, 1.f);
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

// ── Phase 3: spatial hash grid build + O(9-cell) neighbour search ─────────────

// Pass 1 — zero per-cell counters before use each frame
kernel void k_clearGrid(
    device atomic_uint *cellCount [[buffer(0)]],
    device atomic_uint *insertPos [[buffer(1)]],
    constant SimParams &p         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.numCells) return;
    atomic_store_explicit(&cellCount[gid], 0u, memory_order_relaxed);
    atomic_store_explicit(&insertPos[gid], 0u, memory_order_relaxed);
}

// Pass 2 — map each agent to its cell, atomically count agents per cell
kernel void k_hash(
    device const float *posX      [[buffer(0)]],
    device const float *posY      [[buffer(1)]],
    device uint        *cellID    [[buffer(2)]],
    device atomic_uint *cellCount [[buffer(3)]],
    constant SimParams &p         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    int cx = (int)(posX[gid] / p.cellSize);
    int cy = (int)(posY[gid] / p.cellSize);
    cx = clamp(cx, 0, p.gridWidth  - 1);
    cy = clamp(cy, 0, p.gridHeight - 1);
    uint cid    = (uint)(cx + cy * p.gridWidth);
    cellID[gid] = cid;
    atomic_fetch_add_explicit(&cellCount[cid], 1u, memory_order_relaxed);
}

// Pass 3 — exclusive prefix sum on cellCount → cellStart; seed insertPos for scatter
// Dispatched with exactly 1 thread; 390 iterations is negligible on GPU.
kernel void k_prefixSum(
    device const uint  *cellCount [[buffer(0)]],
    device uint        *cellStart [[buffer(1)]],
    device atomic_uint *insertPos [[buffer(2)]],
    constant SimParams &p         [[buffer(3)]]
) {
    uint acc = 0;
    for (int c = 0; c < p.numCells; c++) {
        cellStart[c] = acc;
        atomic_store_explicit(&insertPos[c], acc, memory_order_relaxed);
        acc += cellCount[c];
    }
}

// Pass 4 — scatter each agent into its sorted slot using atomic insert offsets
kernel void k_scatter(
    device const uint  *cellID           [[buffer(0)]],
    device atomic_uint *insertPos        [[buffer(1)]],
    device uint        *sortedAgentIndex [[buffer(2)]],
    constant SimParams &p                [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    uint cid = cellID[gid];
    uint pos = atomic_fetch_add_explicit(&insertPos[cid], 1u, memory_order_relaxed);
    sortedAgentIndex[pos] = gid;
}

// Pass 5 — gather agent data into sorted order so k_steerGrid reads sequentially
//
// Without this, k_steerGrid does posX[sortedAgentIndex[k]] — random reads across N
// elements per neighbour candidate — causing cache misses.  With sorted SoA, threads
// in the same SIMD group (same or adjacent cells after sort) share cache lines.
kernel void k_reorder(
    device const uint  *sortedAgentIndex [[buffer(0)]],
    device const float *posX             [[buffer(1)]],
    device const float *posY             [[buffer(2)]],
    device const float *velX             [[buffer(3)]],
    device const float *velY             [[buffer(4)]],
    device const float *maxSpeed         [[buffer(5)]],
    device float       *sPosX            [[buffer(6)]],
    device float       *sPosY            [[buffer(7)]],
    device float       *sVelX            [[buffer(8)]],
    device float       *sVelY            [[buffer(9)]],
    device float       *sMaxSpeed        [[buffer(10)]],
    constant SimParams &p                [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    uint j         = sortedAgentIndex[gid];
    sPosX[gid]     = posX[j];
    sPosY[gid]     = posY[j];
    sVelX[gid]     = velX[j];
    sVelY[gid]     = velY[j];
    sMaxSpeed[gid] = maxSpeed[j];
}

// Pass 6 — steering with 3×3 grid cell neighbour lookup + obstacle avoidance
//
// gid = sorted position k.  Agents sorted by cellID, so threads in the same SIMD
// group land in the same or adjacent cells and access the same sorted data → shared
// cache lines, no scatter reads.
//
// Buffer layout:
//   0 sPosX (r)  1 sPosY (r)  2 sVelX (r)  3 sVelY (r)  4 sMaxSpeed (r)
//   5 targetX (rw)  6 targetY (rw)
//   7 forceX (w)    8 forceY (w)
//   9 cellStart (r)  10 cellCount (r)  11 sortedAgentIndex (r)  12 SimParams
//   13 obstacles (r)
kernel void k_steerGrid(
    device const float *sPosX            [[buffer(0)]],
    device const float *sPosY            [[buffer(1)]],
    device const float *sVelX            [[buffer(2)]],
    device const float *sVelY            [[buffer(3)]],
    device const float *sMaxSpeed        [[buffer(4)]],
    device float       *targetX          [[buffer(5)]],
    device float       *targetY          [[buffer(6)]],
    device float       *forceX           [[buffer(7)]],
    device float       *forceY           [[buffer(8)]],
    device const uint  *cellStart        [[buffer(9)]],
    device const uint  *cellCount        [[buffer(10)]],
    device const uint     *sortedAgentIndex [[buffer(11)]],
    constant SimParams    &p               [[buffer(12)]],
    device const Obstacle *obstacles       [[buffer(13)]],
    uint gid [[thread_position_in_grid]]   // gid = sorted index k
) {
    if ((int)gid >= p.agentCount) return;

    float px = sPosX[gid];
    float py = sPosY[gid];
    float ms = sMaxSpeed[gid];
    uint  oi = sortedAgentIndex[gid];  // original agent index — used for target/force writes

    float fx = 0.f, fy = 0.f;

    // Goal seeking (target stored by original index)
    float gdx = targetX[oi] - px;
    float gdy = targetY[oi] - py;
    float gd2 = gdx * gdx + gdy * gdy;

    if (gd2 < p.arrivalRadius2) {
        uint seed   = wang_hash(oi ^ (uint)(p.frameIndex) * 2654435761u);
        targetX[oi] = (float)(seed & 0xFFFFu) / 65535.f * p.worldWidth;
        seed        = wang_hash(seed);
        targetY[oi] = (float)(seed & 0xFFFFu) / 65535.f * p.worldHeight;
    } else {
        float inv = ms / sqrt(gd2);
        fx += p.weightSeek * gdx * inv;
        fy += p.weightSeek * gdy * inv;
    }

    // 3×3 grid neighbourhood — reads sPosX/Y/sVelX/Y sequentially within each cell
    int agCellX = (int)(px / p.cellSize);
    int agCellY = (int)(py / p.cellSize);

    float sepX = 0.f, sepY = 0.f;
    float aliVX = 0.f, aliVY = 0.f;
    float cohX = 0.f, cohY = 0.f;
    int   count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        int ny = agCellY + dy;
        if (ny < 0 || ny >= p.gridHeight) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = agCellX + dx;
            if (nx < 0 || nx >= p.gridWidth) continue;

            uint cid   = (uint)(nx + ny * p.gridWidth);
            uint start = cellStart[cid];
            uint end   = start + cellCount[cid];

            for (uint k = start; k < end; k++) {
                if (k == gid) continue;  // self-skip by sorted index

                float ndx = sPosX[k] - px;  // sequential read
                float ndy = sPosY[k] - py;  // sequential read
                float nd2 = ndx * ndx + ndy * ndy;
                if (nd2 > p.neighborRadius2) continue;

                float nd = sqrt(nd2);

                if (nd2 < p.separationRadius2 && nd > 0.001f) {
                    float strength = (p.separationRadius - nd) / p.separationRadius;
                    sepX -= (ndx / nd) * strength;
                    sepY -= (ndy / nd) * strength;
                }

                aliVX += sVelX[k];  // sequential read
                aliVY += sVelY[k];  // sequential read
                cohX  += sPosX[k];  // cached
                cohY  += sPosY[k];  // cached
                count++;
            }
        }
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

    // Obstacle avoidance — closest point on each line segment, repulsion if within radius
    float avoidR = p.obstacleAvoidRadius;
    for (int oi2 = 0; oi2 < p.obstacleCount; oi2++) {
        float2 A  = float2(obstacles[oi2].x0, obstacles[oi2].y0);
        float2 B  = float2(obstacles[oi2].x1, obstacles[oi2].y1);
        float2 AB = B - A;
        float2 AP = float2(px, py) - A;
        float  t  = clamp(dot(AP, AB) / dot(AB, AB), 0.f, 1.f);
        float2 closest  = A + t * AB;
        float2 repulse  = float2(px, py) - closest;
        float  dist     = length(repulse);
        if (dist < avoidR && dist > 0.001f) {
            float strength = (avoidR - dist) / avoidR;
            fx += 3.f * ms * strength * (repulse.x / dist);
            fy += 3.f * ms * strength * (repulse.y / dist);
        }
    }

    forceX[oi] = fx;
    forceY[oi] = fy;
}
