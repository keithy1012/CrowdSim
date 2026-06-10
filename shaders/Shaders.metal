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
//   7 — float *active (per-instance; 0 = exited, move off-screen)
vertex VertexOut vs_agent(
    uint               vid    [[vertex_id]],
    uint               iid    [[instance_id]],
    device const float *posX   [[buffer(0)]],
    device const float *posY   [[buffer(1)]],
    device const float *rad    [[buffer(2)]],
    constant float2   &vp     [[buffer(3)]],
    device const float *velX   [[buffer(4)]],
    device const float *velY   [[buffer(5)]],
    device const float *maxSpd [[buffer(6)]],
    device const float *active [[buffer(7)]]
) {
    float2 offsets[4] = {
        float2(-1.f, -1.f),
        float2(-1.f,  1.f),
        float2( 1.f, -1.f),
        float2( 1.f,  1.f)
    };

    // Move inactive (exited) agents outside the clip volume — no fragment work
    if (active[iid] < 0.5f) {
        VertexOut out;
        out.position = float4(3.f, 3.f, 0.f, 1.f);
        out.uv       = float2(0.f);
        out.speed    = 0.f;
        return out;
    }

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

// ── Phase 6: flow field goal marker (downward-pointing triangle) ──────────────

// Buffer layout:
//   0 — float2 goal  (world-space position)
//   1 — float2 vp    (viewport size in logical points)
vertex ObstacleVert vs_goal(
    uint             vid  [[vertex_id]],
    constant float2 &goal [[buffer(0)]],
    constant float2 &vp   [[buffer(1)]]
) {
    // Downward-pointing triangle: tip marks the goal, base sits above it
    float2 offsets[3] = {
        float2(-11.f, -13.f),  // top-left
        float2( 11.f, -13.f),  // top-right
        float2(  0.f,  13.f),  // tip  (world Y down = visually below)
    };
    float2 worldPos = goal + offsets[vid];
    float2 ndc      = worldPos / vp * 2.f - 1.f;
    ndc.y           = -ndc.y;
    ObstacleVert out;
    out.position = float4(ndc, 0.f, 1.f);
    return out;
}

fragment float4 fs_goal(ObstacleVert in [[stage_in]]) {
    return float4(1.f, 1.f, 1.f, 1.f);
}

// ── Evacuation: exit zone rendering (green ring, one quad per exit) ───────────

struct ExitVert {
    float4 position [[position]];
    float2 uv;
};

// Buffer layout:
//   0 — EvacExit *exits  (x, y, radius, _pad per exit)
//   1 — float2 vp
vertex ExitVert vs_exit(
    uint                    vid   [[vertex_id]],
    uint                    iid   [[instance_id]],
    device const EvacExit  *exits [[buffer(0)]],
    constant float2        &vp    [[buffer(1)]]
) {
    float2 offsets[4] = {
        float2(-1.f, -1.f), float2(-1.f,  1.f),
        float2( 1.f, -1.f), float2( 1.f,  1.f)
    };
    float2 off = offsets[vid];
    float  r   = exits[iid].radius;
    float2 world = float2(exits[iid].x + off.x * r, exits[iid].y + off.y * r);
    float2 ndc   = world / vp * 2.f - 1.f;
    ndc.y        = -ndc.y;
    ExitVert out;
    out.position = float4(ndc, 0.f, 1.f);
    out.uv       = off;
    return out;
}

fragment float4 fs_exit(ExitVert in [[stage_in]]) {
    float d = length(in.uv);
    if (d > 1.f) discard_fragment();
    float alpha = d > 0.72f ? 0.85f : 0.20f;   // bright outer ring, dim fill
    return float4(0.05f, 0.95f, 0.25f, alpha);
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
//   4 forceX (r)  5 forceY (r)  6 maxSpeed (r)  7 SimParams  8 active (r)
kernel void k_integrate(
    device float        *posX     [[buffer(0)]],
    device float        *posY     [[buffer(1)]],
    device float        *velX     [[buffer(2)]],
    device float        *velY     [[buffer(3)]],
    device const float  *forceX   [[buffer(4)]],
    device const float  *forceY   [[buffer(5)]],
    device const float  *maxSpeed [[buffer(6)]],
    constant SimParams  &p        [[buffer(7)]],
    device const float  *active   [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    if (p.evacuationMode && active[gid] < 0.5f) return;  // skip exited agents

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

    float nx = posX[gid] + vx * p.dt;
    float ny = posY[gid] + vy * p.dt;

    // Evacuation: clamp to world bounds (arena has walls; agents must not wrap)
    // Normal mode: wrap so agents cycle continuously
    if (p.evacuationMode) {
        posX[gid] = clamp(nx, 0.f, p.worldWidth);
        posY[gid] = clamp(ny, 0.f, p.worldHeight);
    } else {
        posX[gid] = nx - p.worldWidth  * floor(nx / p.worldWidth);
        posY[gid] = ny - p.worldHeight * floor(ny / p.worldHeight);
    }
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
    device const float *active           [[buffer(12)]],
    device float       *sActive          [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    uint j         = sortedAgentIndex[gid];
    sPosX[gid]     = posX[j];
    sPosY[gid]     = posY[j];
    sVelX[gid]     = velX[j];
    sVelY[gid]     = velY[j];
    sMaxSpeed[gid] = maxSpeed[j];
    sActive[gid]   = active[j];
}

// Pass 6 — steering with tiled 3×3 grid cell neighbour lookup + obstacle avoidance
//
// Tiling (CUDA shared-memory pattern adapted for Metal threadgroup memory):
//   At 100K agents / 390 cells ≈ 256 agents/cell.  All 64 threads in a threadgroup
//   process consecutive sorted agents that are almost always in the SAME cell, so they
//   all scan the SAME 3×3 neighbourhood (~2300 agents).  Without tiling, 64 threads each
//   read those 2300 × 5 floats from device memory → ~736K reads.  With tiling, threads
//   cooperatively load 64 agents at a time into threadgroup (on-chip) memory and all 64
//   threads read from there → ~2300 × 5 global reads + fast TGSM reads.
//
// Barrier uniformity: all threads use thread-0's cell as the reference so every thread
//   in the group iterates the same set of 9 cells and the same number of 64-agent tiles,
//   guaranteeing that every threadgroup_barrier is executed by all threads simultaneously.
//   At lower agent counts a thread may be in a different cell than thread 0; the distance
//   check below still rejects out-of-range agents so results remain correct (some distant
//   neighbours in the non-reference cells may be missed, which is acceptable at low density).
//
// Buffer layout: identical to before (0–14).
kernel void k_steerGrid(
    device const float    *sPosX            [[buffer(0)]],
    device const float    *sPosY            [[buffer(1)]],
    device const float    *sVelX            [[buffer(2)]],
    device const float    *sVelY            [[buffer(3)]],
    device const float    *sMaxSpeed        [[buffer(4)]],
    device float          *targetX          [[buffer(5)]],
    device float          *targetY          [[buffer(6)]],
    device float          *forceX           [[buffer(7)]],
    device float          *forceY           [[buffer(8)]],
    device const uint     *cellStart        [[buffer(9)]],
    device const uint     *cellCount        [[buffer(10)]],
    device const uint     *sortedAgentIndex [[buffer(11)]],
    constant SimParams    &p               [[buffer(12)]],
    device const Obstacle *obstacles       [[buffer(13)]],
    device const float2   *flowField       [[buffer(14)]],
    device const float    *sActive         [[buffer(15)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // ── Threadgroup (on-chip) memory — 6 × 64 × 4 B = 1536 B per threadgroup ──
    threadgroup float tgPosX   [64];
    threadgroup float tgPosY   [64];
    threadgroup float tgVelX   [64];
    threadgroup float tgVelY   [64];
    threadgroup float tgMaxSpd [64];
    threadgroup float tgActive [64];
    threadgroup int   tgRefX   [1];   // thread-0's cell X, broadcast to all
    threadgroup int   tgRefY   [1];

    bool  active = ((int)gid < p.agentCount);
    float px     = active ? sPosX   [gid] : 0.f;
    float py     = active ? sPosY   [gid] : 0.f;
    float ms     = active ? sMaxSpeed[gid]: 1.f;
    uint  oi     = active ? sortedAgentIndex[gid] : 0u;
    // alive = valid gid AND not yet exited (evacuation active flag)
    bool  alive  = active && (sActive[gid] > 0.5f);

    int agCellX = clamp((int)(px / p.cellSize), 0, p.gridWidth  - 1);
    int agCellY = clamp((int)(py / p.cellSize), 0, p.gridHeight - 1);

    // Broadcast thread-0's cell to all threads in the group
    if (lid == 0) { tgRefX[0] = agCellX; tgRefY[0] = agCellY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int refCellX = tgRefX[0];
    int refCellY = tgRefY[0];

    float fx = 0.f, fy = 0.f;

    if (alive) {
        if (p.useFlowField) {
            float2 dir = flowField[agCellX + agCellY * p.gridWidth];
            fx += p.weightSeek * dir.x * ms;
            fy += p.weightSeek * dir.y * ms;
        } else {
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
        }
    }

    float sepX = 0.f, sepY = 0.f;
    float aliVX = 0.f, aliVY = 0.f;
    float cohX = 0.f, cohY = 0.f;
    int   count = 0;
    int   densCount = 0;

    // ── Tiled 3×3 neighbourhood scan ──────────────────────────────────────────
    for (int dy = -1; dy <= 1; dy++) {
        int ny = refCellY + dy;              // uniform across threadgroup
        if (ny < 0 || ny >= p.gridHeight) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = refCellX + dx;          // uniform across threadgroup
            if (nx < 0 || nx >= p.gridWidth) continue;

            uint cid   = (uint)(nx + ny * p.gridWidth);
            uint start = cellStart[cid];     // uniform
            uint end   = start + cellCount[cid]; // uniform

            for (uint tileBase = start; tileBase < end; tileBase += 64) {
                // ── Cooperative load ────────────────────────────────────────
                uint k = tileBase + lid;
                if (k < (uint)p.agentCount) {
                    tgPosX  [lid] = sPosX   [k];
                    tgPosY  [lid] = sPosY   [k];
                    tgVelX  [lid] = sVelX   [k];
                    tgVelY  [lid] = sVelY   [k];
                    tgMaxSpd[lid] = sMaxSpeed[k];
                    tgActive[lid] = sActive [k];
                } else {
                    tgPosX  [lid] = 1e30f;
                    tgPosY  [lid] = 1e30f;
                    tgVelX  [lid] = 0.f; tgVelY[lid] = 0.f; tgMaxSpd[lid] = 1.f;
                    tgActive[lid] = 0.f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ── Scan loaded tile (alive threads only) ────────────────────
                if (alive) {
                    uint tileSize = min(tileBase + 64u, end) - tileBase;
                    for (uint ti = 0; ti < tileSize; ti++) {
                        if (tileBase + ti == gid) continue;    // self-skip
                        if (tgActive[ti] < 0.5f) continue;    // skip exited agents

                        float ndx = tgPosX[ti] - px;
                        float ndy = tgPosY[ti] - py;
                        float nd2 = ndx * ndx + ndy * ndy;
                        if (nd2 > p.neighborRadius2) continue;

                        if (nd2 < p.densityRadius2) densCount++;

                        float nd = sqrt(nd2);

                        if (nd2 < p.separationRadius2 && nd > 0.001f) {
                            float strength = (p.separationRadius - nd) / p.separationRadius;
                            sepX -= (ndx / nd) * strength;
                            sepY -= (ndy / nd) * strength;
                        }

                        aliVX += tgVelX  [ti];
                        aliVY += tgVelY  [ti];
                        cohX  += tgPosX  [ti];
                        cohY  += tgPosY  [ti];
                        count++;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (!active || !alive) return;

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

    // Density-dependent speed: scale all forces down in dense crowds (evacuation only)
    float speedFactor = (p.evacuationMode && p.densityJamCount > 0.f)
        ? clamp(1.f - (float)densCount / p.densityJamCount, 0.1f, 1.f)
        : 1.f;

    forceX[oi] = fx * speedFactor;
    forceY[oi] = fy * speedFactor;
}

// Untiled reference kernel — original k_steerGrid without threadgroup memory.
// Used only by the tiling benchmark to measure the baseline.
kernel void k_steerGrid_notiled(
    device const float    *sPosX            [[buffer(0)]],
    device const float    *sPosY            [[buffer(1)]],
    device const float    *sVelX            [[buffer(2)]],
    device const float    *sVelY            [[buffer(3)]],
    device const float    *sMaxSpeed        [[buffer(4)]],
    device float          *targetX          [[buffer(5)]],
    device float          *targetY          [[buffer(6)]],
    device float          *forceX           [[buffer(7)]],
    device float          *forceY           [[buffer(8)]],
    device const uint     *cellStart        [[buffer(9)]],
    device const uint     *cellCount        [[buffer(10)]],
    device const uint     *sortedAgentIndex [[buffer(11)]],
    constant SimParams    &p               [[buffer(12)]],
    device const Obstacle *obstacles       [[buffer(13)]],
    device const float2   *flowField       [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;

    float px = sPosX[gid];
    float py = sPosY[gid];
    float ms = sMaxSpeed[gid];
    uint  oi = sortedAgentIndex[gid];

    float fx = 0.f, fy = 0.f;

    int agCellX = clamp((int)(px / p.cellSize), 0, p.gridWidth  - 1);
    int agCellY = clamp((int)(py / p.cellSize), 0, p.gridHeight - 1);

    if (p.useFlowField) {
        float2 dir = flowField[agCellX + agCellY * p.gridWidth];
        fx += p.weightSeek * dir.x * ms;
        fy += p.weightSeek * dir.y * ms;
    } else {
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
    }

    float sepX = 0.f, sepY = 0.f, aliVX = 0.f, aliVY = 0.f;
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
                if (k == gid) continue;
                float ndx = sPosX[k] - px;
                float ndy = sPosY[k] - py;
                float nd2 = ndx * ndx + ndy * ndy;
                if (nd2 > p.neighborRadius2) continue;
                float nd = sqrt(nd2);
                if (nd2 < p.separationRadius2 && nd > 0.001f) {
                    float strength = (p.separationRadius - nd) / p.separationRadius;
                    sepX -= (ndx / nd) * strength;
                    sepY -= (ndy / nd) * strength;
                }
                aliVX += sVelX[k]; aliVY += sVelY[k];
                cohX  += sPosX[k]; cohY  += sPosY[k];
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
        float cdx = cohX * inv - px, cdy = cohY * inv - py;
        float cd  = length(float2(cdx, cdy));
        if (cd > 0.001f) {
            fx += p.weightCohere * (cdx / cd) * ms;
            fy += p.weightCohere * (cdy / cd) * ms;
        }
    }

    float avoidR = p.obstacleAvoidRadius;
    for (int oi2 = 0; oi2 < p.obstacleCount; oi2++) {
        float2 A = float2(obstacles[oi2].x0, obstacles[oi2].y0);
        float2 B = float2(obstacles[oi2].x1, obstacles[oi2].y1);
        float2 AB = B - A, AP = float2(px, py) - A;
        float  t  = clamp(dot(AP, AB) / dot(AB, AB), 0.f, 1.f);
        float2 repulse = float2(px, py) - (A + t * AB);
        float  dist    = length(repulse);
        if (dist < avoidR && dist > 0.001f) {
            float strength = (avoidR - dist) / avoidR;
            fx += 3.f * ms * strength * (repulse.x / dist);
            fy += 3.f * ms * strength * (repulse.y / dist);
        }
    }

    forceX[oi] = fx;
    forceY[oi] = fy;
}

// ── Phase 7: ORCA (Optimal Reciprocal Collision Avoidance) ────────────────────
//
// Implements the RVO2 algorithm: for each pair of nearby agents, one ORCA half-plane
// constraint is built in velocity space.  A 2D linear program then finds the velocity
// closest to the preferred (goal-seeking) velocity that satisfies every constraint.
//
// Buffer layout: identical to k_steerGrid (buffers 0–14) — no extra bindings needed.

// 2D cross product (signed area of parallelogram spanned by a and b).
static float det2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

// RVO2 linearProgram1: find the point on constraint line `lineNo` that is
// (a) inside the speed disk of radius `radius`, (b) satisfies constraints [0, lineNo),
// and (c) is closest to `optVel` projected onto the line.
// Returns false when the problem is infeasible.
static bool lp1(
    thread const float2 *lp,   // constraint line points (velocity-space)
    thread const float2 *ld,   // constraint line directions (unit vectors)
    int    lineNo,
    float  radius,
    float2 optVel,
    thread float2 &result
) {
    // Speed-disk intersection: |lp + t*ld|² = radius²
    float dot_  = dot(lp[lineNo], ld[lineNo]);
    float disc  = dot_ * dot_ + radius * radius - dot(lp[lineNo], lp[lineNo]);
    if (disc < 0.f) return false;

    float sqrtD = sqrt(disc);
    float tL    = -dot_ - sqrtD;
    float tR    = -dot_ + sqrtD;

    // Clip range to previous half-planes
    for (int j = 0; j < lineNo; j++) {
        float denom = det2(ld[lineNo], ld[j]);
        float numer = det2(ld[j], lp[lineNo] - lp[j]);
        if (abs(denom) < 1e-5f) {
            if (numer < 0.f) return false;
            continue;
        }
        float t = numer / denom;
        if (denom >= 0.f) tR = min(tR, t);
        else              tL = max(tL, t);
        if (tL > tR) return false;
    }

    // Project optVel onto the constraint line and clamp to the feasible segment
    float t = dot(ld[lineNo], optVel - lp[lineNo]);
    result  = lp[lineNo] + clamp(t, tL, tR) * ld[lineNo];
    return true;
}

// RVO2 linearProgram2: minimum-norm feasible velocity.
// Iterates constraints in order; on violation re-solves on the violated line boundary.
// On full infeasibility returns the partial solution (satisfies the prefix of constraints).
static float2 lp2(
    thread const float2 *lp,
    thread const float2 *ld,
    int    numLines,
    float  radius,
    float2 optVel
) {
    float  spd    = length(optVel);
    float2 result = spd > radius ? (optVel / spd) * radius : optVel;

    for (int i = 0; i < numLines; i++) {
        if (det2(ld[i], lp[i] - result) > 0.f) {
            float2 prev = result;
            if (!lp1(lp, ld, i, radius, optVel, result))
                result = prev;  // partial solution — accept and continue
        }
    }
    return result;
}

#define kMaxORCA 20  // max ORCA constraints per agent (caps LP size)

// Pass 6 (ORCA mode): replace flocking with velocity-space collision avoidance.
//
// Per agent:
//   1. Preferred velocity = flow-field direction × maxSpeed (or direct seek).
//   2. Obstacle repulsion pre-biases the preferred velocity.
//   3. One ORCA half-plane per nearby agent → 2D LP → collision-free velocity.
//   4. Force written as (newVel - currentVel)/dt so k_integrate yields newVel exactly.
// k_orca also uses the same tiled cooperative-load pattern.
// Only posX/Y and velX/Y are needed from neighbours (maxSpeed is not used for constraints).
// Threadgroup memory: 4 × 64 × 4 B = 1024 B per threadgroup.
kernel void k_orca(
    device const float    *sPosX            [[buffer(0)]],
    device const float    *sPosY            [[buffer(1)]],
    device const float    *sVelX            [[buffer(2)]],
    device const float    *sVelY            [[buffer(3)]],
    device const float    *sMaxSpeed        [[buffer(4)]],
    device float          *targetX          [[buffer(5)]],
    device float          *targetY          [[buffer(6)]],
    device float          *forceX           [[buffer(7)]],
    device float          *forceY           [[buffer(8)]],
    device const uint     *cellStart        [[buffer(9)]],
    device const uint     *cellCount        [[buffer(10)]],
    device const uint     *sortedAgentIndex [[buffer(11)]],
    constant SimParams    &p               [[buffer(12)]],
    device const Obstacle *obstacles       [[buffer(13)]],
    device const float2   *flowField       [[buffer(14)]],
    device const float    *sActive         [[buffer(15)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // ── Threadgroup memory ───────────────────────────────────────────────────
    threadgroup float tgPosX   [64];
    threadgroup float tgPosY   [64];
    threadgroup float tgVelX   [64];
    threadgroup float tgVelY   [64];
    threadgroup float tgActive [64];
    threadgroup int   tgRefX   [1];
    threadgroup int   tgRefY   [1];

    bool  active = ((int)gid < p.agentCount);
    float px     = active ? sPosX   [gid] : 0.f;
    float py     = active ? sPosY   [gid] : 0.f;
    float ms     = active ? sMaxSpeed[gid]: 1.f;
    uint  oi     = active ? sortedAgentIndex[gid] : 0u;
    bool  alive  = active && (sActive[gid] > 0.5f);

    int agCellX = clamp((int)(px / p.cellSize), 0, p.gridWidth  - 1);
    int agCellY = clamp((int)(py / p.cellSize), 0, p.gridHeight - 1);

    if (lid == 0) { tgRefX[0] = agCellX; tgRefY[0] = agCellY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int refCellX = tgRefX[0];
    int refCellY = tgRefY[0];

    // ── Preferred velocity ───────────────────────────────────────────────────
    float2 prefVel = float2(0.f);
    if (alive) {
        if (p.useFlowField) {
            float2 dir = flowField[agCellX + agCellY * p.gridWidth];
            prefVel = dir * ms;
        } else {
            float gdx = targetX[oi] - px;
            float gdy = targetY[oi] - py;
            float gd2 = gdx * gdx + gdy * gdy;
            if (gd2 < p.arrivalRadius2) {
                uint seed   = wang_hash(oi ^ (uint)(p.frameIndex) * 2654435761u);
                targetX[oi] = (float)(seed & 0xFFFFu) / 65535.f * p.worldWidth;
                seed        = wang_hash(seed);
                targetY[oi] = (float)(seed & 0xFFFFu) / 65535.f * p.worldHeight;
            } else {
                prefVel = float2(gdx, gdy) * (ms / sqrt(gd2));
            }
        }

        float avoidR = p.obstacleAvoidRadius;
        for (int oi2 = 0; oi2 < p.obstacleCount; oi2++) {
            float2 A  = float2(obstacles[oi2].x0, obstacles[oi2].y0);
            float2 B  = float2(obstacles[oi2].x1, obstacles[oi2].y1);
            float2 AB = B - A;
            float2 AP = float2(px, py) - A;
            float  abLen2 = dot(AB, AB);
            if (abLen2 < 1e-8f) continue;
            float  t       = clamp(dot(AP, AB) / abLen2, 0.f, 1.f);
            float2 closest = A + t * AB;
            float2 repulse = float2(px, py) - closest;
            float  dist    = length(repulse);
            if (dist < avoidR && dist > 0.001f) {
                float strength = (avoidR - dist) / avoidR;
                prefVel += ms * strength * (repulse / dist);
            }
        }
        float prefSpd = length(prefVel);
        if (prefSpd > ms) prefVel *= ms / prefSpd;
    }

    // ── Build ORCA half-planes from nearby agents (tiled) ────────────────────
    float2 lp_arr[kMaxORCA];
    float2 ld_arr[kMaxORCA];
    int numORCA   = 0;
    int densCount = 0;

    float2 myVel = alive ? float2(sVelX[gid], sVelY[gid]) : float2(0.f);
    float  tau   = p.orcaTimeHorizon;
    float  combR = 10.f;
    float  combR2 = combR * combR;

    for (int dy = -1; dy <= 1; dy++) {
        int ny = refCellY + dy;              // uniform across threadgroup
        if (ny < 0 || ny >= p.gridHeight) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = refCellX + dx;          // uniform
            if (nx < 0 || nx >= p.gridWidth) continue;

            uint cid   = (uint)(nx + ny * p.gridWidth);
            uint start = cellStart[cid];     // uniform
            uint end   = start + cellCount[cid];

            for (uint tileBase = start; tileBase < end; tileBase += 64) {
                // ── Cooperative load ────────────────────────────────────────
                uint k = tileBase + lid;
                if (k < (uint)p.agentCount) {
                    tgPosX  [lid] = sPosX  [k];
                    tgPosY  [lid] = sPosY  [k];
                    tgVelX  [lid] = sVelX  [k];
                    tgVelY  [lid] = sVelY  [k];
                    tgActive[lid] = sActive[k];
                } else {
                    tgPosX  [lid] = 1e30f;
                    tgPosY  [lid] = 1e30f;
                    tgVelX  [lid] = 0.f;
                    tgVelY  [lid] = 0.f;
                    tgActive[lid] = 0.f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ── Scan loaded tile ─────────────────────────────────────────
                if (alive) {
                    uint tileSize = min(tileBase + 64u, end) - tileBase;
                    for (uint ti = 0; ti < tileSize; ti++) {
                        if (tileBase + ti == gid) continue;
                        if (tgActive[ti] < 0.5f) continue;     // skip exited agents

                        float2 relPos = float2(tgPosX[ti] - px, tgPosY[ti] - py);
                        float  dist2  = dot(relPos, relPos);
                        if (dist2 > p.neighborRadius2) continue;

                        if (dist2 < p.densityRadius2) densCount++;

                        if (numORCA >= kMaxORCA) continue; // cap reached; skip write

                        float2 relVel = myVel - float2(tgVelX[ti], tgVelY[ti]);
                        float2 lineDir, u;

                        if (dist2 > combR2) {
                            float2 w    = relVel - relPos / tau;
                            float wLen2 = dot(w, w);
                            float dotWP = dot(w, relPos);

                            if (dotWP < 0.f && dotWP * dotWP > combR2 * wLen2) {
                                float  wLen  = sqrt(wLen2 + 1e-10f);
                                float2 unitW = w / wLen;
                                lineDir = float2(unitW.y, -unitW.x);
                                u = (combR / tau - wLen) * unitW;
                            } else {
                                float leg = sqrt(max(dist2 - combR2, 0.f));
                                if (det2(relPos, w) > 0.f) {
                                    lineDir = float2(
                                        relPos.x * leg - relPos.y * combR,
                                        relPos.x * combR + relPos.y * leg) / dist2;
                                } else {
                                    lineDir = -float2(
                                        relPos.x * leg + relPos.y * combR,
                                       -relPos.x * combR + relPos.y * leg) / dist2;
                                }
                                u = dot(relVel, lineDir) * lineDir - relVel;
                            }
                        } else {
                            float  invDt = 1.f / p.dt;
                            float2 w     = relVel - relPos * invDt;
                            float  wLen  = length(w);
                            float2 unitW = wLen > 1e-6f ? w / wLen : float2(0.f, 1.f);
                            lineDir = float2(unitW.y, -unitW.x);
                            u = (combR * invDt - wLen) * unitW;
                        }

                        lp_arr[numORCA] = myVel + 0.5f * u;
                        ld_arr[numORCA] = lineDir;
                        numORCA++;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (!active || !alive) return;

    // Density-dependent speed: scale preferred velocity in dense crowds (evacuation only)
    if (p.evacuationMode && p.densityJamCount > 0.f) {
        float sf = clamp(1.f - (float)densCount / p.densityJamCount, 0.1f, 1.f);
        prefVel *= sf;
    }

    float2 newVel = lp2(lp_arr, ld_arr, numORCA, ms, prefVel);

    forceX[oi] = (newVel.x - myVel.x) / p.dt;
    forceY[oi] = (newVel.y - myVel.y) / p.dt;
}

// ── Evacuation: exit detection ────────────────────────────────────────────────
//
// Runs after k_integrate each frame (evacuation mode only).
// Agents within any exit zone radius are marked inactive and atomically removed
// from liveCount.  Once active[i] = 0 the agent is permanently inert.
//
// Buffer layout:
//   0 posX (r)   1 posY (r)   2 active (rw)   3 exitTime (w)
//   4 liveCount (atomic rw)   5 exits (r)      6 SimParams
kernel void k_checkExit(
    device const float     *posX      [[buffer(0)]],
    device const float     *posY      [[buffer(1)]],
    device float           *active    [[buffer(2)]],
    device float           *exitTime  [[buffer(3)]],
    device atomic_uint     *liveCount [[buffer(4)]],
    device const EvacExit  *exits     [[buffer(5)]],
    constant SimParams     &p         [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.agentCount) return;
    if (active[gid] < 0.5f) return;   // already exited

    float px = posX[gid], py = posY[gid];
    for (int e = 0; e < p.exitCount; e++) {
        float dx = px - exits[e].x;
        float dy = py - exits[e].y;
        if (dx * dx + dy * dy < exits[e].radius * exits[e].radius) {
            active[gid]   = 0.f;
            exitTime[gid] = (float)p.frameIndex;
            atomic_fetch_sub_explicit(liveCount, 1u, memory_order_relaxed);
            return;
        }
    }
}
