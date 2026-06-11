// AgentSim.cu — CUDA port of the Metal crowd-simulation (AgentTraverse)
//
// Compile & run in Google Colab:
//   !nvcc -O2 -arch=sm_75 -o sim AgentSim.cu && ./sim
//   (sm_75 = T4; use sm_86 for A100, sm_89 for L4)
//
// Demo: 500 agents evacuate through a 100-px gap in a vertical wall.
// Outputs a live-count line every simulated second; prints total wall-clock
// time and per-frame GPU timing when all agents have exited.
// ─────────────────────────────────────────────────────────────────────────

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <random>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>

// ── Simulation parameters (all in pixels unless noted) ──────────────────────

static constexpr int   N_AGENTS      = 500;
static constexpr float WORLD_W       = 1280.f;
static constexpr float WORLD_H       = 720.f;
static constexpr float DT            = 1.f / 60.f;
static constexpr float MAX_SPEED     = 80.f;          // px/s
static constexpr float AGENT_RADIUS  = 5.f;
static constexpr float COMB_R        = 10.f;          // combined radius (2×agent)
static constexpr float NEIGHBOR_R    = 50.f;
static constexpr float DENSITY_R     = 30.f;
static constexpr float DENSITY_JAM   = 8.f;
static constexpr float ORLA_HORIZON  = 1.5f;          // ORCA time horizon (s)
static constexpr float OBS_AVOID_R   = 40.f;

// Spatial-hash grid — cellSize ≥ NEIGHBOR_R so 3×3 covers all neighbors
static constexpr float CELL_SIZE     = 50.f;
static constexpr int   GRID_W        = 26;             // ceil(1280/50)
static constexpr int   GRID_H        = 15;
static constexpr int   NUM_CELLS     = GRID_W * GRID_H;

// Flow-field grid — double resolution for obstacle navigation
static constexpr float FLOW_CELL     = 25.f;
static constexpr int   FLOW_W        = 52;             // ceil(1280/25)
static constexpr int   FLOW_H        = 29;             // ceil(720/25)
static constexpr int   FLOW_CELLS    = FLOW_W * FLOW_H;
static constexpr float BLOCK_R       = 14.f;           // obstacle bloat radius
static constexpr float CLEAR_RADIUS  = 2.0f;           // clearance penalty (cells)
static constexpr float CLEAR_PENALTY = 3.0f;

static constexpr int   MAX_EXITS     = 8;
static constexpr int   MAX_OBS       = 64;
static constexpr int   MAX_ORCA      = 20;
static constexpr int   TILE          = 64;             // threadblock size for k_orca

// ── Plain-C structs (shared between host and device) ────────────────────────

struct SimParams {
    int   agentCount;
    float dt;
    float worldWidth, worldHeight;
    int   frameIndex;
    float neighborRadius2;
    float separationRadius, separationRadius2;
    float arrivalRadius2;
    float weightSeek, weightSep, weightAlign, weightCohere;
    int   gridWidth, gridHeight, numCells;
    float cellSize;
    int   flowGridWidth, flowGridHeight;
    float flowCellSize;
    int   obstacleCount;
    float obstacleAvoidRadius, obstacleAvoidRadius2;
    int   useFlowField;
    int   useORCA;
    float orcaTimeHorizon;
    int   evacuationMode;
    int   exitCount;
    float densityJamCount;
    float densityRadius2;
};

struct Obstacle { float x0, y0, x1, y1; };
struct EvacExit  { float x, y, radius, _pad; };

// ── CUDA error-check macro ───────────────────────────────────────────────────

#define CK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// ── Device-side float2 operators (CUDA doesn't provide these by default) ────

// __host__ __device__ so these work in both CPU (buildFlowField) and GPU kernels
__host__ __device__ __forceinline__ float2 operator+(float2 a, float2 b)
    { return make_float2(a.x+b.x, a.y+b.y); }
__host__ __device__ __forceinline__ float2 operator-(float2 a, float2 b)
    { return make_float2(a.x-b.x, a.y-b.y); }
__host__ __device__ __forceinline__ float2 operator*(float2 a, float s)
    { return make_float2(a.x*s, a.y*s); }
__host__ __device__ __forceinline__ float2 operator*(float s, float2 a)
    { return make_float2(a.x*s, a.y*s); }
__host__ __device__ __forceinline__ float2& operator+=(float2 &a, float2 b)
    { a.x += b.x; a.y += b.y; return a; }
__host__ __device__ __forceinline__ float2& operator*=(float2 &a, float s)
    { a.x *= s; a.y *= s; return a; }
__host__ __device__ __forceinline__ float dot2(float2 a, float2 b)
    { return a.x*b.x + a.y*b.y; }
__host__ __device__ __forceinline__ float len2(float2 a)
    { return sqrtf(a.x*a.x + a.y*a.y); }
__host__ __device__ __forceinline__ float2 norm2(float2 a)
    { float l = len2(a); return l > 1e-6f ? a*(1.f/l) : make_float2(0.f,0.f); }
__host__ __device__ __forceinline__ float det2(float2 a, float2 b)
    { return a.x*b.y - a.y*b.x; }

// ── Deterministic per-agent randomization (same hash as Metal) ──────────────

__host__ __device__ __forceinline__ unsigned wang_hash(unsigned s) {
    s ^= 2747636419u; s *= 2654435769u; s ^= s >> 16;
    s *= 2654435769u; s ^= s >> 16;     s *= 2654435769u;
    return s;
}

// ── RVO2 linear programming (ported from lp1/lp2 in Shaders.metal) ──────────
// lp1: find the point on constraint line `lineNo` that:
//   (a) lies inside the speed disk of radius `radius`,
//   (b) satisfies all previous constraints [0, lineNo), and
//   (c) is closest to `optVel` projected onto the line.

__host__ __device__ bool lp1(
    const float2 *lp, const float2 *ld,
    int lineNo, float radius, float2 optVel,
    float2 &result
) {
    float dot_  = dot2(lp[lineNo], ld[lineNo]);
    float disc  = dot_*dot_ + radius*radius - dot2(lp[lineNo], lp[lineNo]);
    if (disc < 0.f) return false;

    float sqrtD = sqrtf(disc);
    float tL    = -dot_ - sqrtD;
    float tR    = -dot_ + sqrtD;

    for (int j = 0; j < lineNo; j++) {
        float denom = det2(ld[lineNo], ld[j]);
        float numer = det2(ld[j], lp[lineNo] - lp[j]);
        if (fabsf(denom) < 1e-5f) { if (numer < 0.f) return false; continue; }
        float t = numer / denom;
        if (denom >= 0.f) tR = fminf(tR, t);
        else              tL = fmaxf(tL, t);
        if (tL > tR) return false;
    }
    float t = dot2(ld[lineNo], optVel - lp[lineNo]);
    result   = lp[lineNo] + ld[lineNo] * fmaxf(tL, fminf(tR, t));
    return true;
}

// lp2: iterate constraints; on violation call lp1 to project onto boundary.
__host__ __device__ float2 lp2(
    const float2 *lp, const float2 *ld,
    int numLines, float radius, float2 optVel
) {
    float  spd    = len2(optVel);
    float2 result = spd > radius ? optVel*(radius/spd) : optVel;

    for (int i = 0; i < numLines; i++) {
        if (det2(ld[i], lp[i] - result) > 0.f) {
            float2 prev = result;
            if (!lp1(lp, ld, i, radius, optVel, result)) result = prev;
        }
    }
    return result;
}

// ── Kernel 1: clear per-cell counters ───────────────────────────────────────

__global__ void k_clearGrid(unsigned *cellCount, unsigned *insertPos,
                             const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.numCells) return;
    cellCount[gid] = 0u;
    insertPos[gid] = 0u;
}

// ── Kernel 2: hash each agent to a cell, count agents per cell ───────────────

__global__ void k_hash(const float *posX, const float *posY,
                        unsigned *cellID, unsigned *cellCount,
                        const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.agentCount) return;
    int cx = max(0, min(p.gridWidth  - 1, (int)(posX[gid] / p.cellSize)));
    int cy = max(0, min(p.gridHeight - 1, (int)(posY[gid] / p.cellSize)));
    unsigned cid = (unsigned)(cx + cy * p.gridWidth);
    cellID[gid]  = cid;
    atomicAdd(&cellCount[cid], 1u);
}

// ── Kernel 3: exclusive prefix sum (single-thread; 390 cells is trivial) ────

__global__ void k_prefixSum(const unsigned *cellCount, unsigned *cellStart,
                              unsigned *insertPos, const SimParams p)
{
    unsigned acc = 0;
    for (int c = 0; c < p.numCells; c++) {
        cellStart[c]  = acc;
        insertPos[c]  = acc;          // seed scatter positions
        acc          += cellCount[c];
    }
}

// ── Kernel 4: scatter each agent into its sorted slot ───────────────────────

__global__ void k_scatter(const unsigned *cellID, unsigned *insertPos,
                           unsigned *sortedIdx, const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.agentCount) return;
    unsigned cid = cellID[gid];
    unsigned pos = atomicAdd(&insertPos[cid], 1u);
    sortedIdx[pos] = (unsigned)gid;
}

// ── Kernel 5: gather agent data in sorted order (improves cache locality) ───

__global__ void k_reorder(const unsigned *sortedIdx,
                           const float *posX,  const float *posY,
                           const float *velX,  const float *velY,
                           const float *maxSpd, const float *active,
                           float *sPosX, float *sPosY,
                           float *sVelX, float *sVelY,
                           float *sMaxSpd, float *sActive,
                           const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.agentCount) return;
    unsigned j  = sortedIdx[gid];
    sPosX  [gid] = posX  [j];
    sPosY  [gid] = posY  [j];
    sVelX  [gid] = velX  [j];
    sVelY  [gid] = velY  [j];
    sMaxSpd[gid] = maxSpd[j];
    sActive[gid] = active[j];
}

// ── Kernel 6: ORCA steering (tiled 3×3 neighbor scan + ORCA half-planes) ────
// Thread-block size MUST equal TILE (64) — shared-memory tile holds 64 agents.

__global__ void k_orca(
    const float *sPosX, const float *sPosY,
    const float *sVelX, const float *sVelY, const float *sMaxSpd,
    float *targetX, float *targetY,
    float *forceX,  float *forceY,
    const unsigned *cellStart, const unsigned *cellCount,
    const unsigned *sortedIdx,
    const SimParams p,
    const Obstacle *obstacles,
    const float2   *flowField,
    const float    *sActive
) {
    __shared__ float tgPosX   [TILE];
    __shared__ float tgPosY   [TILE];
    __shared__ float tgVelX   [TILE];
    __shared__ float tgVelY   [TILE];
    __shared__ float tgActive [TILE];
    __shared__ int   tgRefX;
    __shared__ int   tgRefY;

    int  gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int  lid  = threadIdx.x;
    bool inBounds = (gid < p.agentCount);

    float px  = inBounds ? sPosX   [gid] : 0.f;
    float py  = inBounds ? sPosY   [gid] : 0.f;
    float ms  = inBounds ? sMaxSpd [gid] : 1.f;
    unsigned oi = inBounds ? sortedIdx[gid] : 0u;
    bool alive   = inBounds && (sActive[gid] > 0.5f);

    int agCX = max(0, min(p.gridWidth  - 1, (int)(px / p.cellSize)));
    int agCY = max(0, min(p.gridHeight - 1, (int)(py / p.cellSize)));
    int fCX  = max(0, min(p.flowGridWidth  - 1, (int)(px / p.flowCellSize)));
    int fCY  = max(0, min(p.flowGridHeight - 1, (int)(py / p.flowCellSize)));

    // Thread 0 broadcasts its reference cell to the whole block so every thread
    // scans the same 3×3 neighborhood — they are all near the same sorted cell.
    if (lid == 0) { tgRefX = agCX; tgRefY = agCY; }
    __syncthreads();
    int refCX = tgRefX, refCY = tgRefY;

    // ── Preferred velocity: flow field + obstacle repulsion ──────────────────
    float2 prefVel = make_float2(0.f, 0.f);
    if (alive) {
        if (p.useFlowField) {
            prefVel = flowField[fCX + fCY * p.flowGridWidth] * ms;
        } else {
            float gdx = targetX[oi] - px, gdy = targetY[oi] - py;
            float gd2 = gdx*gdx + gdy*gdy;
            if (gd2 < p.arrivalRadius2) {
                unsigned seed = wang_hash((unsigned)oi ^ (unsigned)p.frameIndex * 2654435761u);
                targetX[oi]   = (float)(seed & 0xFFFFu) / 65535.f * p.worldWidth;
                seed          = wang_hash(seed);
                targetY[oi]   = (float)(seed & 0xFFFFu) / 65535.f * p.worldHeight;
            } else {
                prefVel = make_float2(gdx, gdy) * (ms / sqrtf(gd2));
            }
        }
        // Obstacle repulsion
        for (int oi2 = 0; oi2 < p.obstacleCount; oi2++) {
            float2 A  = make_float2(obstacles[oi2].x0, obstacles[oi2].y0);
            float2 B  = make_float2(obstacles[oi2].x1, obstacles[oi2].y1);
            float2 AB = B - A;
            float2 AP = make_float2(px, py) - A;
            float  ab2 = dot2(AB, AB);
            if (ab2 < 1e-8f) continue;
            float  t   = fmaxf(0.f, fminf(1.f, dot2(AP, AB) / ab2));
            float2 cl  = A + AB*t;
            float2 rep = make_float2(px, py) - cl;
            float  d   = len2(rep);
            if (d < p.obstacleAvoidRadius && d > 0.001f) {
                float str = (p.obstacleAvoidRadius - d) / p.obstacleAvoidRadius;
                prefVel  += (ms * str / d) * rep;
            }
        }
        float sp = len2(prefVel);
        if (sp > ms) prefVel *= ms / sp;
    }

    // ── Build ORCA constraints (tiled cooperative load) ─────────────────────
    float2 lp_arr[MAX_ORCA];
    float2 ld_arr[MAX_ORCA];
    int numORCA   = 0;
    int densCount = 0;

    float prefSpd = len2(prefVel);
    float2 fwd    = prefSpd > 0.001f ? prefVel*(1.f/prefSpd) : make_float2(0.f,0.f);
    float2 myVel  = alive ? make_float2(sVelX[gid], sVelY[gid]) : make_float2(0.f,0.f);
    float  tau    = p.orcaTimeHorizon;

    for (int dy = -1; dy <= 1; dy++) {
        int ny = refCY + dy;
        if (ny < 0 || ny >= p.gridHeight) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = refCX + dx;
            if (nx < 0 || nx >= p.gridWidth) continue;

            unsigned cid   = (unsigned)(nx + ny * p.gridWidth);
            unsigned start = cellStart[cid];
            unsigned end   = start + cellCount[cid];

            for (unsigned tileBase = start; tileBase < end; tileBase += TILE) {
                // Cooperative load: each thread loads one neighbor into shared memory
                unsigned k = tileBase + (unsigned)lid;
                if (k < (unsigned)p.agentCount) {
                    tgPosX  [lid] = sPosX  [k];
                    tgPosY  [lid] = sPosY  [k];
                    tgVelX  [lid] = sVelX  [k];
                    tgVelY  [lid] = sVelY  [k];
                    tgActive[lid] = sActive[k];
                } else {
                    tgPosX  [lid] = 1e30f; tgPosY  [lid] = 1e30f;
                    tgVelX  [lid] = 0.f;   tgVelY  [lid] = 0.f;
                    tgActive[lid] = 0.f;
                }
                __syncthreads();

                if (alive) {
                    unsigned tileSize = min(tileBase + (unsigned)TILE, end) - tileBase;
                    for (unsigned ti = 0; ti < tileSize; ti++) {
                        if (tileBase + ti == (unsigned)gid) continue;
                        if (tgActive[ti] < 0.5f) continue;

                        float2 relPos = make_float2(tgPosX[ti]-px, tgPosY[ti]-py);
                        float  dist2  = dot2(relPos, relPos);
                        if (dist2 > p.neighborRadius2) continue;

                        // Count only agents ahead (density crowding model)
                        if (dist2 < p.densityRadius2 && dot2(relPos, fwd) > 0.f)
                            densCount++;

                        if (numORCA >= MAX_ORCA) continue;

                        float2 relVel  = myVel - make_float2(tgVelX[ti], tgVelY[ti]);
                        float2 lineDir, u;
                        float  combR2  = COMB_R * COMB_R;

                        if (dist2 > combR2) {
                            float2 w    = relVel - relPos*(1.f/tau);
                            float  wL2  = dot2(w, w);
                            float  dwp  = dot2(w, relPos);
                            if (dwp < 0.f && dwp*dwp > combR2*wL2) {
                                float  wLen  = sqrtf(wL2 + 1e-10f);
                                float2 unitW = w*(1.f/wLen);
                                lineDir = make_float2(unitW.y, -unitW.x);
                                u = (COMB_R/tau - wLen) * unitW;
                            } else {
                                float leg = sqrtf(fmaxf(dist2 - combR2, 0.f));
                                if (det2(relPos, w) > 0.f) {
                                    lineDir = make_float2(
                                        relPos.x*leg - relPos.y*COMB_R,
                                        relPos.x*COMB_R + relPos.y*leg) * (1.f/dist2);
                                } else {
                                    lineDir = make_float2(
                                        -(relPos.x*leg + relPos.y*COMB_R),
                                         relPos.x*COMB_R - relPos.y*leg) * (1.f/dist2);
                                }
                                u = dot2(relVel, lineDir)*lineDir - relVel;
                            }
                        } else {
                            float  invDt = 1.f / p.dt;
                            float2 w     = relVel - relPos*invDt;
                            float  wLen  = len2(w);
                            float2 unitW = wLen > 1e-6f ? w*(1.f/wLen) : make_float2(0.f,1.f);
                            lineDir = make_float2(unitW.y, -unitW.x);
                            u = (COMB_R*invDt - wLen) * unitW;
                        }
                        lp_arr[numORCA] = myVel + 0.5f*u;
                        ld_arr[numORCA] = lineDir;
                        numORCA++;
                    }
                }
                __syncthreads();
            }
        }
    }

    if (!inBounds || !alive) return;

    // Density-dependent speed (forward crowding only)
    if (p.evacuationMode && p.densityJamCount > 0.f) {
        float sf = fmaxf(0.1f, fminf(1.f, 1.f - (float)densCount / p.densityJamCount));
        prefVel *= sf;
    }

    float2 newVel = lp2(lp_arr, ld_arr, numORCA, ms, prefVel);
    forceX[oi] = (newVel.x - myVel.x) / p.dt;
    forceY[oi] = (newVel.y - myVel.y) / p.dt;
}

// ── Kernel 7: integrate forces → velocity → position ────────────────────────

__global__ void k_integrate(float *posX, float *posY,
                              float *velX, float *velY,
                              const float *forceX, const float *forceY,
                              const float *maxSpd, const float *active,
                              const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.agentCount) return;
    if (p.evacuationMode && active[gid] < 0.5f) return;

    float vx = velX[gid] + forceX[gid] * p.dt;
    float vy = velY[gid] + forceY[gid] * p.dt;
    float sp = sqrtf(vx*vx + vy*vy);
    if (sp > maxSpd[gid]) { float s = maxSpd[gid]/sp; vx *= s; vy *= s; }

    float nx = posX[gid] + vx * p.dt;
    float ny = posY[gid] + vy * p.dt;

    if (p.evacuationMode) {
        if      (nx < 0.f)          { nx = 0.f;          vx = 0.f; }
        else if (nx > p.worldWidth)  { nx = p.worldWidth;  vx = 0.f; }
        if      (ny < 0.f)          { ny = 0.f;           vy = 0.f; }
        else if (ny > p.worldHeight) { ny = p.worldHeight; vy = 0.f; }
        posX[gid] = nx; posY[gid] = ny;
    } else {
        posX[gid] = nx - p.worldWidth  * floorf(nx / p.worldWidth);
        posY[gid] = ny - p.worldHeight * floorf(ny / p.worldHeight);
    }
    velX[gid] = vx; velY[gid] = vy;
}

// ── Kernel 8: detect agents reaching an exit zone ───────────────────────────

__global__ void k_checkExit(const float *posX, const float *posY,
                              float *active, float *exitTime,
                              unsigned *liveCount,
                              const EvacExit *exits, const SimParams p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.agentCount) return;
    if (active[gid] < 0.5f) return;

    float px = posX[gid], py = posY[gid];
    for (int e = 0; e < p.exitCount; e++) {
        float dx = px - exits[e].x, dy = py - exits[e].y;
        if (dx*dx + dy*dy < exits[e].radius * exits[e].radius) {
            active  [gid] = 0.f;
            exitTime[gid] = (float)p.frameIndex;
            atomicSub(liveCount, 1u);
            return;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU: clearance-weighted flow-field BFS
// Mirrors recomputeFlowField() in GPUSimulation.mm exactly.
// ─────────────────────────────────────────────────────────────────────────────

struct FlowVec { float x, y; };

static void buildFlowField(
    const std::vector<Obstacle> &obs,
    const std::vector<EvacExit> &exits,
    std::vector<FlowVec> &ff          // output: FLOW_CELLS elements
) {
    const float cellSz = FLOW_CELL;
    const int   W = FLOW_W, H = FLOW_H, NC = FLOW_CELLS;

    // ── Pass 1: wall-distance BFS ─────────────────────────────────────────
    // Flood outward from every cell that overlaps a wall or world edge.
    std::vector<float> wallDist(NC, 1e30f);
    std::vector<bool>  blocked (NC, false);
    std::queue<int>    q;

    auto markBlocked = [&](int c) {
        if (blocked[c]) return;
        blocked[c]   = true;
        wallDist[c]  = 0.f;
        q.push(c);
    };

    for (int c = 0; c < NC; c++) {
        float cx = (c % W + 0.5f) * cellSz;
        float cy = (c / W + 0.5f) * cellSz;

        // World edges
        if (cx < BLOCK_R || cx > WORLD_W - BLOCK_R ||
            cy < BLOCK_R || cy > WORLD_H - BLOCK_R)
        {
            markBlocked(c);
            continue;
        }
        // Obstacle segments
        for (const auto &ob : obs) {
            float2 A = make_float2(ob.x0, ob.y0);
            float2 B = make_float2(ob.x1, ob.y1);
            float  ax = B.x - A.x, ay = B.y - A.y;
            float  len2ab = ax*ax + ay*ay;
            float  t = 0.f;
            if (len2ab > 1e-8f) {
                t = ((cx-A.x)*ax + (cy-A.y)*ay) / len2ab;
                t = fmaxf(0.f, fminf(1.f, t));
            }
            float closestX = A.x + t*ax, closestY = A.y + t*ay;
            float dx = cx - closestX, dy = cy - closestY;
            if (dx*dx + dy*dy < BLOCK_R*BLOCK_R) { markBlocked(c); break; }
        }
    }

    // BFS outward from blocked cells — Chebyshev neighbours
    while (!q.empty()) {
        int c = q.front(); q.pop();
        int cx2 = c % W, cy2 = c / W;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy2 + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx2 + dx;
                if (nx < 0 || nx >= W) continue;
                int n = nx + ny * W;
                float d = wallDist[c] + cellSz * sqrtf((float)(dx*dx + dy*dy));
                if (d < wallDist[n]) { wallDist[n] = d; q.push(n); }
            }
        }
    }

    // ── Pass 2: Dijkstra with clearance penalty ───────────────────────────
    static const int   dirs8[8][2] = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
    };
    static const float dists8[8]   = {1,1,1,1,1.4142f,1.4142f,1.4142f,1.4142f};

    std::vector<float> cost(NC, 1e30f);

    // Seed: every cell within any exit radius gets cost=0
    for (const auto &ex : exits) {
        for (int c = 0; c < NC; c++) {
            if (blocked[c]) continue;
            float pcx = (c % W + 0.5f) * cellSz;
            float pcy = (c / W + 0.5f) * cellSz;
            float dx  = pcx - ex.x, dy = pcy - ex.y;
            if (dx*dx + dy*dy < ex.radius*ex.radius) cost[c] = 0.f;
        }
    }

    // Priority queue: (cost, cellIndex)
    using PQEntry = std::pair<float, int>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    for (int c = 0; c < NC; c++) if (cost[c] == 0.f) pq.push({0.f, c});

    while (!pq.empty()) {
        PQEntry top = pq.top(); pq.pop();
        float g = top.first; int c = top.second;
        if (g > cost[c] + 1e-4f) continue;
        int cx2 = c % W, cy2 = c / W;
        for (int d = 0; d < 8; d++) {
            int nx = cx2 + dirs8[d][0], ny = cy2 + dirs8[d][1];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int n = nx + ny * W;
            if (blocked[n]) continue;

            // Clearance penalty: cells near walls cost more → agents prefer corridors
            float clr    = fminf(wallDist[n] / cellSz, CLEAR_RADIUS) / CLEAR_RADIUS;
            float pen    = (1.f - clr) * CLEAR_PENALTY;
            float newCost = cost[c] + dists8[d] * cellSz * (1.f + pen);
            if (newCost < cost[n]) {
                cost[n] = newCost;
                pq.push({newCost, n});
            }
        }
    }

    // ── Derive flow vectors from cost gradient ────────────────────────────
    ff.resize(NC);
    for (int c = 0; c < NC; c++) {
        if (blocked[c]) { ff[c] = {0.f, 0.f}; continue; }

        // Cost=0 cells: point toward nearest exit center
        if (cost[c] == 0.f) {
            float pcx = (c % W + 0.5f) * cellSz, pcy = (c / W + 0.5f) * cellSz;
            float bestD2 = 1e30f, dirX = 0.f, dirY = 0.f;
            for (const auto &ex : exits) {
                float ddx = ex.x - pcx, ddy = ex.y - pcy;
                float d2  = ddx*ddx + ddy*ddy;
                if (d2 < bestD2) { bestD2 = d2; dirX = ddx; dirY = ddy; }
            }
            float len = sqrtf(dirX*dirX + dirY*dirY);
            ff[c]     = len > 0.001f ? FlowVec{dirX/len, dirY/len} : FlowVec{0.f,0.f};
            continue;
        }

        // Weighted blend of cost-improving neighbors
        int cx2 = c % W, cy2 = c / W;
        float sumX = 0.f, sumY = 0.f, sumW = 0.f;
        for (int d = 0; d < 8; d++) {
            int nx = cx2 + dirs8[d][0], ny = cy2 + dirs8[d][1];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int n = nx + ny * W;
            if (cost[n] >= cost[c]) continue;   // must improve
            float improvement = cost[c] - cost[n];
            sumX += dirs8[d][0] * improvement;
            sumY += dirs8[d][1] * improvement;
            sumW += improvement;
        }
        if (sumW < 1e-6f) { ff[c] = {0.f, 0.f}; continue; }
        float len = sqrtf(sumX*sumX + sumY*sumY);
        ff[c] = len > 1e-6f ? FlowVec{sumX/len, sumY/len} : FlowVec{0.f,0.f};
    }
}

// ── Simulation state (GPU + CPU mirrors for stats) ──────────────────────────

struct Sim {
    // Agent SoA — original and sorted copies
    float *posX,  *posY;
    float *velX,  *velY;
    float *maxSpd;
    float *active, *exitTime;
    float *sPosX, *sPosY, *sVelX, *sVelY, *sMaxSpd, *sActive;
    float *forceX, *forceY;
    float *targetX, *targetY;

    // Spatial hash
    unsigned *cellID, *cellCount, *cellStart, *insertPos, *sortedIdx;

    // Flow field and exits on GPU
    float2   *d_flowField;
    EvacExit *d_exits;
    Obstacle *d_obstacles;
    unsigned *d_liveCount;

    SimParams params;
    int       N;
};

static Sim* simCreate(int N,
                       const std::vector<float> &hPosX, const std::vector<float> &hPosY,
                       const std::vector<EvacExit> &exits,
                       const std::vector<Obstacle> &obs,
                       const std::vector<FlowVec>  &ff)
{
    Sim *s = new Sim;
    s->N   = N;

    auto alloc = [&](void **p, size_t bytes) { CK(cudaMalloc(p, bytes)); };
    size_t fN  = N * sizeof(float);
    size_t uN  = N * sizeof(unsigned);

    alloc((void**)&s->posX,    fN);  alloc((void**)&s->posY,    fN);
    alloc((void**)&s->velX,    fN);  alloc((void**)&s->velY,    fN);
    alloc((void**)&s->maxSpd,  fN);
    alloc((void**)&s->active,  fN);  alloc((void**)&s->exitTime,fN);
    alloc((void**)&s->sPosX,   fN);  alloc((void**)&s->sPosY,   fN);
    alloc((void**)&s->sVelX,   fN);  alloc((void**)&s->sVelY,   fN);
    alloc((void**)&s->sMaxSpd, fN);  alloc((void**)&s->sActive, fN);
    alloc((void**)&s->forceX,  fN);  alloc((void**)&s->forceY,  fN);
    alloc((void**)&s->targetX, fN);  alloc((void**)&s->targetY, fN);

    alloc((void**)&s->cellID,    uN);
    alloc((void**)&s->cellCount, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->cellStart, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->insertPos, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->sortedIdx, uN);

    alloc((void**)&s->d_flowField, FLOW_CELLS * sizeof(float2));
    alloc((void**)&s->d_exits,     exits.size() * sizeof(EvacExit));
    alloc((void**)&s->d_obstacles, (obs.empty() ? 1 : obs.size()) * sizeof(Obstacle));
    alloc((void**)&s->d_liveCount, sizeof(unsigned));

    // Upload initial data
    CK(cudaMemcpy(s->posX,   hPosX.data(), fN, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->posY,   hPosY.data(), fN, cudaMemcpyHostToDevice));
    CK(cudaMemset(s->velX,   0, fN)); CK(cudaMemset(s->velY,   0, fN));
    CK(cudaMemset(s->forceX, 0, fN)); CK(cudaMemset(s->forceY, 0, fN));

    std::vector<float> ones(N, 1.f), speeds(N, MAX_SPEED);
    CK(cudaMemcpy(s->active,  ones.data(),   fN, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->maxSpd,  speeds.data(), fN, cudaMemcpyHostToDevice));
    CK(cudaMemset(s->exitTime, 0, fN));

    // Random targets (for non-flow-field mode; not used in evacuation)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(0, WORLD_W), ry(0, WORLD_H);
    std::vector<float> tx(N), ty(N);
    for (int i = 0; i < N; i++) { tx[i] = rx(rng); ty[i] = ry(rng); }
    CK(cudaMemcpy(s->targetX, tx.data(), fN, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->targetY, ty.data(), fN, cudaMemcpyHostToDevice));

    // Flow field (CPU float {x,y} → GPU float2)
    std::vector<float2> gpuFF(FLOW_CELLS);
    for (int i = 0; i < FLOW_CELLS; i++)
        gpuFF[i] = make_float2(ff[i].x, ff[i].y);
    CK(cudaMemcpy(s->d_flowField, gpuFF.data(), FLOW_CELLS*sizeof(float2), cudaMemcpyHostToDevice));

    CK(cudaMemcpy(s->d_exits, exits.data(), exits.size()*sizeof(EvacExit), cudaMemcpyHostToDevice));
    if (!obs.empty())
        CK(cudaMemcpy(s->d_obstacles, obs.data(), obs.size()*sizeof(Obstacle), cudaMemcpyHostToDevice));

    unsigned lc = (unsigned)N;
    CK(cudaMemcpy(s->d_liveCount, &lc, sizeof(unsigned), cudaMemcpyHostToDevice));

    // SimParams
    SimParams &p = s->params;
    memset(&p, 0, sizeof(p));
    p.agentCount        = N;
    p.dt                = DT;
    p.worldWidth        = WORLD_W;
    p.worldHeight       = WORLD_H;
    p.neighborRadius2   = NEIGHBOR_R * NEIGHBOR_R;
    p.separationRadius  = COMB_R;
    p.separationRadius2 = COMB_R * COMB_R;
    p.arrivalRadius2    = 50.f * 50.f;
    p.weightSeek        = 1.f;
    p.weightSep         = 0.f;
    p.weightAlign       = 0.f;
    p.weightCohere      = 0.f;
    p.gridWidth         = GRID_W;
    p.gridHeight        = GRID_H;
    p.numCells          = NUM_CELLS;
    p.cellSize          = CELL_SIZE;
    p.flowGridWidth     = FLOW_W;
    p.flowGridHeight    = FLOW_H;
    p.flowCellSize      = FLOW_CELL;
    p.obstacleCount     = (int)obs.size();
    p.obstacleAvoidRadius  = OBS_AVOID_R;
    p.obstacleAvoidRadius2 = OBS_AVOID_R * OBS_AVOID_R;
    p.useFlowField      = 1;
    p.useORCA           = 1;
    p.orcaTimeHorizon   = ORLA_HORIZON;
    p.evacuationMode    = 1;
    p.exitCount         = (int)exits.size();
    p.densityJamCount   = DENSITY_JAM;
    p.densityRadius2    = DENSITY_R * DENSITY_R;

    return s;
}

// ── One simulation frame ─────────────────────────────────────────────────────

static void simStep(Sim *s) {
    SimParams p = s->params;
    int N = s->N;

    dim3 bk128(128), bk64(64);
    dim3 gN  ((N + 127) / 128);
    dim3 gNC ((NUM_CELLS + 127) / 128);
    dim3 gN64((N + 63) / 64);     // for k_orca (block size must be 64)

    // 1 — Clear hash
    k_clearGrid<<<gNC, bk128>>>(s->cellCount, s->insertPos, p);

    // 2 — Hash agents to cells
    k_hash<<<gN, bk128>>>(s->posX, s->posY, s->cellID, s->cellCount, p);

    // 3 — Prefix sum (single thread; 390 iterations)
    k_prefixSum<<<1, 1>>>(s->cellCount, s->cellStart, s->insertPos, p);

    // 4 — Scatter into sorted slots
    k_scatter<<<gN, bk128>>>(s->cellID, s->insertPos, s->sortedIdx, p);

    // 5 — Reorder SoA for cache-friendly neighbor reads
    k_reorder<<<gN, bk128>>>(s->sortedIdx,
        s->posX,  s->posY,  s->velX,  s->velY,  s->maxSpd,  s->active,
        s->sPosX, s->sPosY, s->sVelX, s->sVelY, s->sMaxSpd, s->sActive, p);

    // 6 — ORCA steering (block size = TILE = 64)
    k_orca<<<gN64, bk64>>>(
        s->sPosX, s->sPosY, s->sVelX, s->sVelY, s->sMaxSpd,
        s->targetX, s->targetY, s->forceX, s->forceY,
        s->cellStart, s->cellCount, s->sortedIdx, p,
        s->d_obstacles, s->d_flowField, s->sActive);

    // 7 — Integrate
    k_integrate<<<gN, bk128>>>(
        s->posX, s->posY, s->velX, s->velY,
        s->forceX, s->forceY, s->maxSpd, s->active, p);

    // 8 — Exit detection
    k_checkExit<<<gN, bk128>>>(
        s->posX, s->posY, s->active, s->exitTime,
        s->d_liveCount, s->d_exits, p);

    s->params.frameIndex++;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-path simulation — same algorithm, plain C++ loops, no CUDA.
// Run with: ./sim --cpu
// Lets you measure single-threaded CPU vs GPU on identical input.
// ─────────────────────────────────────────────────────────────────────────────

struct SimCPU {
    float *posX,  *posY;
    float *velX,  *velY;
    float *maxSpd;
    float *active, *exitTime;
    float *sPosX, *sPosY, *sVelX, *sVelY, *sMaxSpd, *sActive;
    float *forceX, *forceY;
    float *targetX, *targetY;
    unsigned *cellID, *cellCount, *cellStart, *insertPos, *sortedIdx;
    float2   *flowField;
    EvacExit  exits[MAX_EXITS];
    Obstacle  obstacles[MAX_OBS];
    unsigned  liveCount;
    SimParams params;
    int       N;
};

static SimCPU* simCPUCreate(int N,
                              const std::vector<float> &hPosX,
                              const std::vector<float> &hPosY,
                              const std::vector<EvacExit> &exits,
                              const std::vector<Obstacle> &obs,
                              const std::vector<FlowVec>  &ff)
{
    SimCPU *s = new SimCPU;
    s->N = N;

    auto alloc = [&](void **p, size_t bytes) { *p = malloc(bytes); };
    size_t fN = N * sizeof(float), uN = N * sizeof(unsigned);

    alloc((void**)&s->posX,    fN);  alloc((void**)&s->posY,    fN);
    alloc((void**)&s->velX,    fN);  alloc((void**)&s->velY,    fN);
    alloc((void**)&s->maxSpd,  fN);
    alloc((void**)&s->active,  fN);  alloc((void**)&s->exitTime,fN);
    alloc((void**)&s->sPosX,   fN);  alloc((void**)&s->sPosY,   fN);
    alloc((void**)&s->sVelX,   fN);  alloc((void**)&s->sVelY,   fN);
    alloc((void**)&s->sMaxSpd, fN);  alloc((void**)&s->sActive, fN);
    alloc((void**)&s->forceX,  fN);  alloc((void**)&s->forceY,  fN);
    alloc((void**)&s->targetX, fN);  alloc((void**)&s->targetY, fN);

    alloc((void**)&s->cellID,    uN);
    alloc((void**)&s->cellCount, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->cellStart, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->insertPos, NUM_CELLS * sizeof(unsigned));
    alloc((void**)&s->sortedIdx, uN);
    alloc((void**)&s->flowField, FLOW_CELLS * sizeof(float2));

    memcpy(s->posX, hPosX.data(), fN);
    memcpy(s->posY, hPosY.data(), fN);
    memset(s->velX,   0, fN); memset(s->velY,   0, fN);
    memset(s->forceX, 0, fN); memset(s->forceY, 0, fN);
    memset(s->exitTime, 0, fN);
    for (int i = 0; i < N; i++) { s->active[i] = 1.f; s->maxSpd[i] = MAX_SPEED; }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(0, WORLD_W), ry(0, WORLD_H);
    for (int i = 0; i < N; i++) { s->targetX[i] = rx(rng); s->targetY[i] = ry(rng); }

    for (int i = 0; i < FLOW_CELLS; i++) s->flowField[i] = make_float2(ff[i].x, ff[i].y);

    int numExits = (int)exits.size(), numObs = (int)obs.size();
    memcpy(s->exits,     exits.data(), numExits * sizeof(EvacExit));
    memcpy(s->obstacles, obs.data(),   numObs   * sizeof(Obstacle));

    s->liveCount = (unsigned)N;

    SimParams &p = s->params;
    memset(&p, 0, sizeof(p));
    p.agentCount        = N;
    p.dt                = DT;
    p.worldWidth        = WORLD_W;
    p.worldHeight       = WORLD_H;
    p.neighborRadius2   = NEIGHBOR_R * NEIGHBOR_R;
    p.separationRadius  = COMB_R;
    p.separationRadius2 = COMB_R * COMB_R;
    p.arrivalRadius2    = 50.f * 50.f;
    p.weightSeek        = 1.f;
    p.gridWidth         = GRID_W;
    p.gridHeight        = GRID_H;
    p.numCells          = NUM_CELLS;
    p.cellSize          = CELL_SIZE;
    p.flowGridWidth     = FLOW_W;
    p.flowGridHeight    = FLOW_H;
    p.flowCellSize      = FLOW_CELL;
    p.obstacleCount     = numObs;
    p.obstacleAvoidRadius  = OBS_AVOID_R;
    p.obstacleAvoidRadius2 = OBS_AVOID_R * OBS_AVOID_R;
    p.useFlowField      = 1;
    p.useORCA           = 1;
    p.orcaTimeHorizon   = ORLA_HORIZON;
    p.evacuationMode    = 1;
    p.exitCount         = numExits;
    p.densityJamCount   = DENSITY_JAM;
    p.densityRadius2    = DENSITY_R * DENSITY_R;

    return s;
}

// CPU equivalents of each kernel: just for-loops over agents / cells.

static void cpu_clearGrid(unsigned *cellCount, unsigned *insertPos, int numCells) {
    for (int c = 0; c < numCells; c++) { cellCount[c] = 0; insertPos[c] = 0; }
}

static void cpu_hash(const float *posX, const float *posY,
                      unsigned *cellID, unsigned *cellCount, const SimParams &p)
{
    for (int i = 0; i < p.agentCount; i++) {
        int cx = std::max(0, std::min(p.gridWidth  - 1, (int)(posX[i] / p.cellSize)));
        int cy = std::max(0, std::min(p.gridHeight - 1, (int)(posY[i] / p.cellSize)));
        unsigned cid = (unsigned)(cx + cy * p.gridWidth);
        cellID[i] = cid;
        cellCount[cid]++;
    }
}

static void cpu_prefixSum(const unsigned *cellCount, unsigned *cellStart,
                           unsigned *insertPos, const SimParams &p)
{
    unsigned acc = 0;
    for (int c = 0; c < p.numCells; c++) {
        cellStart[c] = insertPos[c] = acc;
        acc += cellCount[c];
    }
}

static void cpu_scatter(const unsigned *cellID, unsigned *insertPos,
                         unsigned *sortedIdx, const SimParams &p)
{
    for (int i = 0; i < p.agentCount; i++) {
        unsigned cid = cellID[i];
        sortedIdx[insertPos[cid]++] = (unsigned)i;
    }
}

static void cpu_reorder(const unsigned *sortedIdx,
                         const float *posX, const float *posY,
                         const float *velX, const float *velY,
                         const float *maxSpd, const float *active,
                         float *sPosX, float *sPosY,
                         float *sVelX, float *sVelY,
                         float *sMaxSpd, float *sActive,
                         const SimParams &p)
{
    for (int i = 0; i < p.agentCount; i++) {
        unsigned j  = sortedIdx[i];
        sPosX  [i] = posX  [j]; sPosY  [i] = posY  [j];
        sVelX  [i] = velX  [j]; sVelY  [i] = velY  [j];
        sMaxSpd[i] = maxSpd[j]; sActive[i] = active[j];
    }
}

// CPU ORCA: same algorithm as k_orca but no shared memory — direct array reads.
static void cpu_orca(
    const float *sPosX, const float *sPosY,
    const float *sVelX, const float *sVelY, const float *sMaxSpd,
    float *targetX, float *targetY,
    float *forceX,  float *forceY,
    const unsigned *cellStart, const unsigned *cellCount,
    const unsigned *sortedIdx,
    const SimParams &p,
    const Obstacle *obstacles,
    const float2   *flowField,
    const float    *sActive)
{
    for (int gid = 0; gid < p.agentCount; gid++) {
        bool alive = (sActive[gid] > 0.5f);
        float px = sPosX[gid], py = sPosY[gid];
        float ms = sMaxSpd[gid];
        unsigned oi = sortedIdx[gid];

        int agCX = std::max(0, std::min(p.gridWidth  - 1, (int)(px / p.cellSize)));
        int agCY = std::max(0, std::min(p.gridHeight - 1, (int)(py / p.cellSize)));
        int fCX  = std::max(0, std::min(p.flowGridWidth  - 1, (int)(px / p.flowCellSize)));
        int fCY  = std::max(0, std::min(p.flowGridHeight - 1, (int)(py / p.flowCellSize)));

        float2 prefVel = make_float2(0.f, 0.f);
        if (alive) {
            prefVel = flowField[fCX + fCY * p.flowGridWidth] * ms;
            for (int oi2 = 0; oi2 < p.obstacleCount; oi2++) {
                float2 A  = make_float2(obstacles[oi2].x0, obstacles[oi2].y0);
                float2 B  = make_float2(obstacles[oi2].x1, obstacles[oi2].y1);
                float2 AB = B - A;
                float2 AP = make_float2(px, py) - A;
                float  ab2 = dot2(AB, AB);
                if (ab2 < 1e-8f) continue;
                float  t   = std::max(0.f, std::min(1.f, dot2(AP, AB) / ab2));
                float2 cl  = A + AB * t;
                float2 rep = make_float2(px, py) - cl;
                float  d   = len2(rep);
                if (d < p.obstacleAvoidRadius && d > 0.001f) {
                    float str = (p.obstacleAvoidRadius - d) / p.obstacleAvoidRadius;
                    prefVel += (ms * str / d) * rep;
                }
            }
            float sp = len2(prefVel);
            if (sp > ms) prefVel *= ms / sp;
        }

        float2 lp_arr[MAX_ORCA], ld_arr[MAX_ORCA];
        int numORCA = 0, densCount = 0;
        float prefSpd = len2(prefVel);
        float2 fwd    = prefSpd > 0.001f ? prefVel * (1.f/prefSpd) : make_float2(0.f, 0.f);
        float2 myVel  = alive ? make_float2(sVelX[gid], sVelY[gid]) : make_float2(0.f, 0.f);
        float  tau    = p.orcaTimeHorizon;

        for (int dy = -1; dy <= 1; dy++) {
            int ny = agCY + dy;
            if (ny < 0 || ny >= p.gridHeight) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = agCX + dx;
                if (nx < 0 || nx >= p.gridWidth) continue;
                unsigned cid   = (unsigned)(nx + ny * p.gridWidth);
                unsigned start = cellStart[cid];
                unsigned end   = start + cellCount[cid];

                for (unsigned k = start; k < end; k++) {
                    if (k == (unsigned)gid) continue;
                    if (sActive[k] < 0.5f) continue;

                    float2 relPos = make_float2(sPosX[k]-px, sPosY[k]-py);
                    float  dist2  = dot2(relPos, relPos);
                    if (dist2 > p.neighborRadius2) continue;

                    if (dist2 < p.densityRadius2 && dot2(relPos, fwd) > 0.f)
                        densCount++;

                    if (numORCA >= MAX_ORCA) continue;

                    float2 relVel  = myVel - make_float2(sVelX[k], sVelY[k]);
                    float2 lineDir, u;
                    float  combR2  = COMB_R * COMB_R;

                    if (dist2 > combR2) {
                        float2 w    = relVel - relPos * (1.f / tau);
                        float  wL2  = dot2(w, w);
                        float  dwp  = dot2(w, relPos);
                        if (dwp < 0.f && dwp*dwp > combR2*wL2) {
                            float  wLen  = sqrtf(wL2 + 1e-10f);
                            float2 unitW = w * (1.f / wLen);
                            lineDir = make_float2(unitW.y, -unitW.x);
                            u = (COMB_R/tau - wLen) * unitW;
                        } else {
                            float leg = sqrtf(std::max(dist2 - combR2, 0.f));
                            if (det2(relPos, w) > 0.f) {
                                lineDir = make_float2(relPos.x*leg - relPos.y*COMB_R,
                                                      relPos.x*COMB_R + relPos.y*leg) * (1.f/dist2);
                            } else {
                                lineDir = make_float2(-(relPos.x*leg + relPos.y*COMB_R),
                                                       relPos.x*COMB_R - relPos.y*leg) * (1.f/dist2);
                            }
                            u = dot2(relVel, lineDir) * lineDir - relVel;
                        }
                    } else {
                        float  invDt = 1.f / p.dt;
                        float2 w     = relVel - relPos * invDt;
                        float  wLen  = len2(w);
                        float2 unitW = wLen > 1e-6f ? w * (1.f/wLen) : make_float2(0.f, 1.f);
                        lineDir = make_float2(unitW.y, -unitW.x);
                        u = (COMB_R*invDt - wLen) * unitW;
                    }
                    lp_arr[numORCA] = myVel + 0.5f * u;
                    ld_arr[numORCA] = lineDir;
                    numORCA++;
                }
            }
        }

        if (!alive) continue;

        if (p.evacuationMode && p.densityJamCount > 0.f) {
            float sf = std::max(0.1f, std::min(1.f, 1.f - (float)densCount / p.densityJamCount));
            prefVel *= sf;
        }

        float2 newVel   = lp2(lp_arr, ld_arr, numORCA, ms, prefVel);
        forceX[oi] = (newVel.x - myVel.x) / p.dt;
        forceY[oi] = (newVel.y - myVel.y) / p.dt;
    }
}

static void cpu_integrate(float *posX, float *posY,
                           float *velX, float *velY,
                           const float *forceX, const float *forceY,
                           const float *maxSpd, const float *active,
                           const SimParams &p)
{
    for (int i = 0; i < p.agentCount; i++) {
        if (p.evacuationMode && active[i] < 0.5f) continue;
        float vx = velX[i] + forceX[i] * p.dt;
        float vy = velY[i] + forceY[i] * p.dt;
        float sp = sqrtf(vx*vx + vy*vy);
        if (sp > maxSpd[i]) { float s = maxSpd[i]/sp; vx *= s; vy *= s; }
        float nx = posX[i] + vx * p.dt, ny = posY[i] + vy * p.dt;
        if (p.evacuationMode) {
            if      (nx < 0.f)          { nx = 0.f;          vx = 0.f; }
            else if (nx > p.worldWidth)  { nx = p.worldWidth;  vx = 0.f; }
            if      (ny < 0.f)          { ny = 0.f;           vy = 0.f; }
            else if (ny > p.worldHeight) { ny = p.worldHeight; vy = 0.f; }
            posX[i] = nx; posY[i] = ny;
        } else {
            posX[i] = nx - p.worldWidth  * floorf(nx / p.worldWidth);
            posY[i] = ny - p.worldHeight * floorf(ny / p.worldHeight);
        }
        velX[i] = vx; velY[i] = vy;
    }
}

static void cpu_checkExit(const float *posX, const float *posY,
                           float *active, float *exitTime,
                           unsigned &liveCount,
                           const EvacExit *exits, const SimParams &p)
{
    for (int i = 0; i < p.agentCount; i++) {
        if (active[i] < 0.5f) continue;
        for (int e = 0; e < p.exitCount; e++) {
            float dx = posX[i] - exits[e].x, dy = posY[i] - exits[e].y;
            if (dx*dx + dy*dy < exits[e].radius * exits[e].radius) {
                active  [i] = 0.f;
                exitTime[i] = (float)p.frameIndex;
                liveCount--;
                break;
            }
        }
    }
}

static void simCPUStep(SimCPU *s) {
    SimParams &p = s->params;
    cpu_clearGrid(s->cellCount, s->insertPos, p.numCells);
    cpu_hash     (s->posX, s->posY, s->cellID, s->cellCount, p);
    cpu_prefixSum(s->cellCount, s->cellStart, s->insertPos, p);
    cpu_scatter  (s->cellID, s->insertPos, s->sortedIdx, p);
    cpu_reorder  (s->sortedIdx,
                  s->posX, s->posY, s->velX, s->velY, s->maxSpd, s->active,
                  s->sPosX, s->sPosY, s->sVelX, s->sVelY, s->sMaxSpd, s->sActive, p);
    cpu_orca     (s->sPosX, s->sPosY, s->sVelX, s->sVelY, s->sMaxSpd,
                  s->targetX, s->targetY, s->forceX, s->forceY,
                  s->cellStart, s->cellCount, s->sortedIdx, p,
                  s->obstacles, s->flowField, s->sActive);
    cpu_integrate(s->posX, s->posY, s->velX, s->velY,
                  s->forceX, s->forceY, s->maxSpd, s->active, p);
    cpu_checkExit(s->posX, s->posY, s->active, s->exitTime,
                  s->liveCount, s->exits, p);
    p.frameIndex++;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    bool cpuMode = (argc > 1 && strcmp(argv[1], "--cpu") == 0);
    printf("AgentSim %s — %d agents, world %.0f×%.0f\n",
           cpuMode ? "CPU" : "CUDA", N_AGENTS, WORLD_W, WORLD_H);

    // ── Scenario: vertical wall at x=640, gap 310–410 (100 px) ─────────────
    std::vector<Obstacle> obstacles = {
        {640.f, 0.f,   640.f, 310.f},   // wall above gap
        {640.f, 410.f, 640.f, 720.f},   // wall below gap
    };

    std::vector<EvacExit> exits = {
        {1200.f, 360.f, 60.f, 0.f}     // exit on right side
    };

    // ── Build flow field (CPU) ───────────────────────────────────────────────
    printf("Building flow field... ");
    fflush(stdout);
    std::vector<FlowVec> ff;
    buildFlowField(obstacles, exits, ff);
    printf("done (%d cells, %.0f px/cell)\n", FLOW_CELLS, FLOW_CELL);

    // ── Spawn agents: random positions in left half ─────────────────────────
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rx(60.f, 580.f), ry(60.f, 660.f);
    std::vector<float> hPosX(N_AGENTS), hPosY(N_AGENTS);
    for (int i = 0; i < N_AGENTS; i++) { hPosX[i] = rx(rng); hPosY[i] = ry(rng); }

    const int MAX_FRAMES  = 6000;    // 100 simulated seconds at 60fps
    const int PRINT_EVERY = 60;

    // ── CPU path ─────────────────────────────────────────────────────────────
    if (cpuMode) {
        SimCPU *sim = simCPUCreate(N_AGENTS, hPosX, hPosY, exits, obstacles, ff);

        printf("\nFrame   SimTime  Alive   ms/frame\n");
        printf("------  -------  ------  --------\n");

        auto wallStart = std::chrono::high_resolution_clock::now();
        double totalMs = 0.0;

        for (int frame = 0; frame < MAX_FRAMES; frame++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            simCPUStep(sim);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            totalMs += ms;

            if (frame % PRINT_EVERY == 0 || frame == MAX_FRAMES - 1) {
                printf("%6d  %6.1f s  %6u  %.3f\n",
                       frame, frame * DT, sim->liveCount, ms);
                fflush(stdout);
                if (sim->liveCount == 0) {
                    printf("All agents evacuated at frame %d!\n", frame);
                    break;
                }
            }
        }

        double wallSec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wallStart).count();
        printf("\n--- Summary (CPU) ---\n");
        printf("  Agents evacuated : %d / %d\n", N_AGENTS - (int)sim->liveCount, N_AGENTS);
        printf("  Simulated time   : %.1f s\n", sim->params.frameIndex * DT);
        printf("  Wall-clock time  : %.2f s\n", wallSec);
        printf("  Avg CPU ms/frame : %.3f\n",
               sim->params.frameIndex > 0 ? totalMs / sim->params.frameIndex : 0.0);
        return 0;
    }

    // ── GPU path ─────────────────────────────────────────────────────────────
    Sim *sim = simCreate(N_AGENTS, hPosX, hPosY, exits, obstacles, ff);

    // Warm-up: one step to populate GPU caches and JIT-compile PTX
    simStep(sim);
    CK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CK(cudaEventCreate(&evStart));
    CK(cudaEventCreate(&evStop));

    double totalGpuMs = 0.0;
    auto wallStart = std::chrono::high_resolution_clock::now();

    printf("\nFrame   SimTime  Alive   GPU ms/frame\n");
    printf("------  -------  ------  ------------\n");

    for (int frame = 0; frame < MAX_FRAMES; frame++) {
        CK(cudaEventRecord(evStart));
        simStep(sim);
        CK(cudaEventRecord(evStop));
        CK(cudaEventSynchronize(evStop));

        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, evStart, evStop));
        totalGpuMs += ms;

        if (frame % PRINT_EVERY == 0 || frame == MAX_FRAMES - 1) {
            unsigned lc;
            CK(cudaMemcpy(&lc, sim->d_liveCount, sizeof(unsigned), cudaMemcpyDeviceToHost));
            printf("%6d  %6.1f s  %6u  %.3f\n", frame, frame * DT, lc, ms);
            fflush(stdout);
            if (lc == 0) { printf("All agents evacuated at frame %d!\n", frame); break; }
        }
    }

    double wallSec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wallStart).count();
    unsigned finalLive;
    CK(cudaMemcpy(&finalLive, sim->d_liveCount, sizeof(unsigned), cudaMemcpyDeviceToHost));

    printf("\n--- Summary (GPU) ---\n");
    printf("  Agents evacuated : %d / %d\n", N_AGENTS - (int)finalLive, N_AGENTS);
    printf("  Simulated time   : %.1f s\n", sim->params.frameIndex * DT);
    printf("  Wall-clock time  : %.2f s\n", wallSec);
    printf("  Avg GPU ms/frame : %.3f\n",
           sim->params.frameIndex > 0 ? totalGpuMs / sim->params.frameIndex : 0.0);

    CK(cudaEventDestroy(evStart));
    CK(cudaEventDestroy(evStop));
    return 0;
}
