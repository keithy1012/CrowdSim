#import "GPUSimulation.h"
#include "Simulation.h"    // world/behavior constants
#include "SharedTypes.h"
#include <random>
#include <queue>
#include <algorithm>
#include <cmath>

// Grid dimensions derived from world size and neighbor radius.
// cellSize == kNeighborRadius ensures a 3×3 cell search covers the full interaction circle.
static const int kGridWidth  = 26;  // ceil(1280 / 50)
static const int kGridHeight = 15;  // ceil( 720 / 50)
static const int kNumCells   = kGridWidth * kGridHeight; // 390
static const uint32_t kInitialObstacleCapacity = 64;

@implementation GPUSimulation {
    id<MTLDevice>               _device;

    // Phase 2 — kept for Phase 4 benchmarking reference
    id<MTLComputePipelineState> _steerPipeline;
    id<MTLComputePipelineState> _integratePipeline;

    // Phase 3 — grid build + grid-based steering
    id<MTLComputePipelineState> _clearGridPipeline;
    id<MTLComputePipelineState> _hashPipeline;
    id<MTLComputePipelineState> _prefixSumPipeline;
    id<MTLComputePipelineState> _scatterPipeline;
    id<MTLComputePipelineState> _reorderPipeline;
    id<MTLComputePipelineState> _steerGridPipeline;

    // Phase 7 — ORCA collision avoidance (replaces k_steerGrid when active)
    id<MTLComputePipelineState> _orcaPipeline;
    BOOL                        _useORCA;
    // Tiling benchmark — untiled baseline kept for comparison
    id<MTLComputePipelineState> _noTiledPipeline;
    BOOL                        _useTiling;

    // Evacuation mode
    id<MTLComputePipelineState> _checkExitPipeline;
    id<MTLBuffer> _activeBuffer;      // float[N]:  1.0 = alive, 0.0 = exited
    id<MTLBuffer> _sActive;           // float[N]:  sorted active flag (reorder output)
    id<MTLBuffer> _exitTimeBuffer;    // float[N]:  frame index when each agent exited
    id<MTLBuffer> _liveCountBuffer;   // uint32_t[1]: atomic live-agent counter
    id<MTLBuffer> _exitBuffer;        // EvacExit[kMaxEvacExits]
    uint32_t      _evacExitCount;
    BOOL          _evacuationMode;
    BOOL          _evacuationComplete;
    uint32_t      _evacSpawnCount;    // agents spawned in current run
    int           _evacSpawnFrame;    // frame index at spawn time

    // SoA agent data — shared storage; CPU writes once at init, GPU owns thereafter
    id<MTLBuffer> _posX, _posY;
    id<MTLBuffer> _velX, _velY;
    id<MTLBuffer> _targetX, _targetY;
    id<MTLBuffer> _maxSpeed;
    id<MTLBuffer> _radBuffer;
    id<MTLBuffer> _forceX, _forceY;
    id<MTLBuffer> _params;

    // Spatial hash grid — rebuilt each frame
    id<MTLBuffer> _cellID;           // uint[N]:   which cell each agent belongs to
    id<MTLBuffer> _cellCount;        // uint[G]:   agents per cell (written atomically)
    id<MTLBuffer> _cellStart;        // uint[G]:   sorted-array start index for each cell
    id<MTLBuffer> _insertPos;        // uint[G]:   atomic insert cursor per cell
    id<MTLBuffer> _sortedAgentIndex; // uint[N]:   agents sorted by cell

    // Sorted SoA — gathered by k_reorder so k_steerGrid reads sequentially
    id<MTLBuffer> _sPosX, _sPosY;    // float[N]:  positions in sorted order
    id<MTLBuffer> _sVelX, _sVelY;    // float[N]:  velocities in sorted order
    id<MTLBuffer> _sMaxSpeed;        // float[N]:  max speeds in sorted order

    // Phase 5 — static obstacle line segments (dynamic buffer, doubles when full)
    id<MTLBuffer> _obstacleBuffer;
    uint32_t      _obstacleCapacity;

    // Phase 6 — flow field navigation
    id<MTLBuffer> _flowFieldBuffer;  // float2[kNumCells]: one direction vector per grid cell
    float         _flowGoalX, _flowGoalY;
    BOOL          _flowGoalSet;      // NO until the user places the first goal
    BOOL          _flowFieldDirty;   // recompute on next encode if YES
    BOOL          _useFlowField;

    uint32_t _agentCount;
    uint32_t _obstacleCount;
    int      _frameIndex;
}

@synthesize agentCount             = _agentCount;
@synthesize obstacleCount          = _obstacleCount;
@synthesize posXBuffer             = _posX;
@synthesize posYBuffer             = _posY;
@synthesize radBuffer              = _radBuffer;
@synthesize velXBuffer             = _velX;
@synthesize velYBuffer             = _velY;
@synthesize maxSpeedBuffer         = _maxSpeed;
@synthesize obstacleBuffer         = _obstacleBuffer;
@synthesize activeBuffer           = _activeBuffer;
@synthesize evacExitBuffer         = _exitBuffer;
@synthesize evacExitCount          = _evacExitCount;
@synthesize useGridSteering        = _useGridSteering;
@synthesize flowGoalSet            = _flowGoalSet;
@synthesize flowGoalX              = _flowGoalX;
@synthesize flowGoalY              = _flowGoalY;
@synthesize evacuationComplete     = _evacuationComplete;
@synthesize evacuationCompleteCallback;
// Custom getters/setters below: useFlowField, useORCA, useTiling, evacuationMode

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                    agentCount:(uint32_t)count {
    self = [super init];
    if (!self) return nil;
    _device          = device;
    _agentCount      = count;
    _useGridSteering = YES;
    _useTiling       = YES;
    [self buildPipelines:library];
    [self allocateAndInitBuffers];
    return self;
}

// ── Pipeline setup ────────────────────────────────────────────────────────────

- (id<MTLComputePipelineState>)pipelineNamed:(NSString *)name
                                     library:(id<MTLLibrary>)lib {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { NSLog(@"[GPUSim] %@ not found in library", name); return nil; }
    NSError *err = nil;
    id<MTLComputePipelineState> ps =
        [_device newComputePipelineStateWithFunction:fn error:&err];
    if (err) NSLog(@"[GPUSim] %@: %@", name, err);
    return ps;
}

- (void)buildPipelines:(id<MTLLibrary>)library {
    _steerPipeline     = [self pipelineNamed:@"k_steer"     library:library];
    _integratePipeline = [self pipelineNamed:@"k_integrate" library:library];
    _clearGridPipeline = [self pipelineNamed:@"k_clearGrid" library:library];
    _hashPipeline      = [self pipelineNamed:@"k_hash"      library:library];
    _prefixSumPipeline = [self pipelineNamed:@"k_prefixSum" library:library];
    _scatterPipeline   = [self pipelineNamed:@"k_scatter"   library:library];
    _reorderPipeline   = [self pipelineNamed:@"k_reorder"   library:library];
    _steerGridPipeline = [self pipelineNamed:@"k_steerGrid" library:library];
    _orcaPipeline      = [self pipelineNamed:@"k_orca"               library:library];
    _noTiledPipeline   = [self pipelineNamed:@"k_steerGrid_notiled"  library:library];
    _checkExitPipeline = [self pipelineNamed:@"k_checkExit"          library:library];
}

// ── Buffer allocation + CPU initialisation ────────────────────────────────────

- (id<MTLBuffer>)sharedBuf:(NSUInteger)bytes {
    return [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

- (void)allocateAndInitBuffers {
    const NSUInteger floatSz = _agentCount * sizeof(float);
    const NSUInteger uintN   = _agentCount * sizeof(uint32_t);
    const NSUInteger uintG   = kNumCells   * sizeof(uint32_t);

    _posX     = [self sharedBuf:floatSz];
    _posY     = [self sharedBuf:floatSz];
    _velX     = [self sharedBuf:floatSz];
    _velY     = [self sharedBuf:floatSz];
    _targetX  = [self sharedBuf:floatSz];
    _targetY  = [self sharedBuf:floatSz];
    _maxSpeed = [self sharedBuf:floatSz];
    _radBuffer= [self sharedBuf:floatSz];
    _forceX   = [self sharedBuf:floatSz];
    _forceY   = [self sharedBuf:floatSz];
    _params   = [_device newBufferWithLength:sizeof(SimParams)
                                     options:MTLResourceStorageModeShared];

    _cellID           = [self sharedBuf:uintN];
    _cellCount        = [self sharedBuf:uintG];
    _cellStart        = [self sharedBuf:uintG];
    _insertPos        = [self sharedBuf:uintG];
    _sortedAgentIndex = [self sharedBuf:uintN];

    _sPosX     = [self sharedBuf:floatSz];
    _sPosY     = [self sharedBuf:floatSz];
    _sVelX     = [self sharedBuf:floatSz];
    _sVelY     = [self sharedBuf:floatSz];
    _sMaxSpeed = [self sharedBuf:floatSz];
    _sActive   = [self sharedBuf:floatSz];

    // Evacuation buffers — always allocated; active defaults to all-1 (everyone alive)
    _activeBuffer   = [self sharedBuf:floatSz];
    _exitTimeBuffer = [self sharedBuf:floatSz];
    _liveCountBuffer= [self sharedBuf:sizeof(uint32_t)];
    _exitBuffer     = [self sharedBuf:kMaxEvacExits * sizeof(struct EvacExit)];

    float *act = (float *)_activeBuffer.contents;
    for (uint32_t i = 0; i < _agentCount; i++) act[i] = 1.f;
    memset(_exitTimeBuffer.contents, 0, floatSz);
    *((uint32_t *)_liveCountBuffer.contents) = _agentCount;
    memset(_exitBuffer.contents, 0, kMaxEvacExits * sizeof(struct EvacExit));
    _evacExitCount   = 0;
    _evacuationMode  = NO;
    _evacuationComplete = NO;

    _obstacleCapacity = kInitialObstacleCapacity;
    _obstacleBuffer   = [self sharedBuf:_obstacleCapacity * sizeof(struct Obstacle)];
    _obstacleCount    = 0;

    // Flow field: float2 per cell, zero-initialised (no flow until goal is set)
    _flowFieldBuffer = [self sharedBuf:kNumCells * sizeof(float) * 2];
    memset(_flowFieldBuffer.contents, 0, kNumCells * sizeof(float) * 2);
    _flowGoalSet    = NO;
    _flowFieldDirty = NO;
    _useFlowField   = NO;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(0.f, Simulation::kWorldWidth);
    std::uniform_real_distribution<float> ry(0.f, Simulation::kWorldHeight);
    std::uniform_real_distribution<float> spd(80.f, 120.f);

    float *pX  = (float *)_posX.contents,    *pY  = (float *)_posY.contents;
    float *vX  = (float *)_velX.contents,    *vY  = (float *)_velY.contents;
    float *tX  = (float *)_targetX.contents, *tY  = (float *)_targetY.contents;
    float *ms  = (float *)_maxSpeed.contents;
    float *rad = (float *)_radBuffer.contents;
    float *fX  = (float *)_forceX.contents,  *fY  = (float *)_forceY.contents;

    for (uint32_t i = 0; i < _agentCount; i++) {
        pX[i]  = rx(rng); pY[i]  = ry(rng);
        vX[i]  = 0.f;     vY[i]  = 0.f;
        tX[i]  = rx(rng); tY[i]  = ry(rng);
        ms[i]  = spd(rng);
        rad[i] = 5.f;
        fX[i]  = 0.f;     fY[i]  = 0.f;
    }

    SimParams *p         = (SimParams *)_params.contents;
    p->agentCount        = (int)_agentCount;
    p->dt                = 1.f / 60.f;
    p->worldWidth        = Simulation::kWorldWidth;
    p->worldHeight       = Simulation::kWorldHeight;
    p->frameIndex        = 0;
    p->neighborRadius2   = Simulation::kNeighborRadius   * Simulation::kNeighborRadius;
    p->separationRadius  = Simulation::kSeparationRadius;
    p->separationRadius2 = Simulation::kSeparationRadius * Simulation::kSeparationRadius;
    p->arrivalRadius2    = Simulation::kArrivalRadius    * Simulation::kArrivalRadius;
    p->weightSeek        = 1.0f;
    p->weightSep         = 2.0f;
    p->weightAlign       = 0.4f;
    p->weightCohere      = 0.2f;
    p->gridWidth         = kGridWidth;
    p->gridHeight        = kGridHeight;
    p->numCells          = kNumCells;
    p->cellSize          = Simulation::kNeighborRadius;
    p->obstacleCount        = 0;
    p->obstacleAvoidRadius  = 40.f;
    p->obstacleAvoidRadius2 = 40.f * 40.f;
    p->useFlowField         = 0;
    p->useORCA              = 0;
    p->orcaTimeHorizon      = 1.5f;
    p->evacuationMode       = 0;
    p->exitCount            = 0;
    p->densityJamCount      = 10.f;
    p->densityRadius2       = 30.f * 30.f;
}

// ── Obstacle loading ──────────────────────────────────────────────────────────

- (void)growObstacleBufferTo:(uint32_t)needed {
    if (needed <= _obstacleCapacity) return;
    uint32_t cap = _obstacleCapacity;
    while (cap < needed) cap *= 2;
    id<MTLBuffer> newBuf = [self sharedBuf:cap * sizeof(struct Obstacle)];
    if (_obstacleCount > 0)
        memcpy(newBuf.contents, _obstacleBuffer.contents,
               _obstacleCount * sizeof(struct Obstacle));
    _obstacleBuffer   = newBuf;
    _obstacleCapacity = cap;
}

- (void)loadObstacles:(const struct Obstacle *)obstacles count:(uint32_t)count {
    [self growObstacleBufferTo:MAX(count, 1u)];
    if (count > 0)
        memcpy(_obstacleBuffer.contents, obstacles, count * sizeof(struct Obstacle));
    _obstacleCount = count;
    ((SimParams *)_params.contents)->obstacleCount = (int)count;
    if (_useFlowField && _flowGoalSet) _flowFieldDirty = YES;
}

- (void)appendObstacle:(struct Obstacle)obs {
    [self growObstacleBufferTo:_obstacleCount + 1];
    struct Obstacle *buf = (struct Obstacle *)_obstacleBuffer.contents;
    buf[_obstacleCount++] = obs;
    ((SimParams *)_params.contents)->obstacleCount = (int)_obstacleCount;
    if (_useFlowField && _flowGoalSet) _flowFieldDirty = YES;
}

// ── Flow field ────────────────────────────────────────────────────────────────

- (BOOL)useFlowField { return _useFlowField; }

- (void)setUseFlowField:(BOOL)on {
    _useFlowField = on;
    ((SimParams *)_params.contents)->useFlowField = on ? 1 : 0;
    if (on && _flowGoalSet) _flowFieldDirty = YES;
}

- (BOOL)useORCA { return _useORCA; }

- (void)setUseORCA:(BOOL)on {
    _useORCA = on;
    ((SimParams *)_params.contents)->useORCA = on ? 1 : 0;
}

- (BOOL)useTiling { return _useTiling; }
- (void)setUseTiling:(BOOL)on { _useTiling = on; }

- (BOOL)evacuationMode { return _evacuationMode; }
- (void)setEvacuationMode:(BOOL)on {
    _evacuationMode = on;
    ((SimParams *)_params.contents)->evacuationMode = on ? 1 : 0;
}

- (uint32_t)liveAgentCount {
    return *((uint32_t *)_liveCountBuffer.contents);
}

- (void)setFlowFieldGoal:(float)x y:(float)y {
    _flowGoalX   = x;
    _flowGoalY   = y;
    _flowGoalSet = YES;
    [self recomputeFlowField];
}

- (void)recomputeFlowField {
    // Requires at least one goal: either the traditional single goal or evac exits
    if (!_flowGoalSet && _evacExitCount == 0) return;

    const float cellSz = Simulation::kNeighborRadius;   // 50 px
    const float blockR = cellSz * 0.5f;

    // ── Build blocked map ──────────────────────────────────────────────────
    bool blocked[kNumCells] = {};
    if (_obstacleCount > 0) {
        const struct Obstacle *obs = (const struct Obstacle *)_obstacleBuffer.contents;
        for (int c = 0; c < kNumCells; c++) {
            float cx = ((c % kGridWidth) + 0.5f) * cellSz;
            float cy = ((c / kGridWidth) + 0.5f) * cellSz;
            for (uint32_t oi = 0; oi < _obstacleCount && !blocked[c]; oi++) {
                float ax = obs[oi].x0, ay = obs[oi].y0;
                float bx = obs[oi].x1, by = obs[oi].y1;
                float abx = bx-ax, aby = by-ay;
                float len2 = abx*abx + aby*aby;
                float t = len2 > 0.f
                    ? std::clamp(((cx-ax)*abx + (cy-ay)*aby) / len2, 0.f, 1.f)
                    : 0.f;
                float dx = cx-(ax+t*abx), dy = cy-(ay+t*aby);
                blocked[c] = (dx*dx + dy*dy) < blockR*blockR;
            }
        }
    }

    // ── Dijkstra (8-directional) — multi-source seeding ───────────────────
    static const int   NDX[] = {-1, 0, 1,-1, 1,-1, 0, 1};
    static const int   NDY[] = {-1,-1,-1, 0, 0, 1, 1, 1};
    static const float NDC[] = {1.414f,1.f,1.414f,1.f,1.f,1.414f,1.f,1.414f};

    float cost[kNumCells];
    std::fill(cost, cost+kNumCells, 1e30f);

    using PQ = std::priority_queue<std::pair<float,int>,
                                   std::vector<std::pair<float,int>>,
                                   std::greater<std::pair<float,int>>>;
    PQ pq;

    // Seed from evacuation exits (multi-source: nearest exit wins)
    if (_evacExitCount > 0) {
        const struct EvacExit *ex = (const struct EvacExit *)_exitBuffer.contents;
        for (uint32_t e = 0; e < _evacExitCount; e++) {
            int gcx = std::clamp((int)(ex[e].x / cellSz), 0, kGridWidth -1);
            int gcy = std::clamp((int)(ex[e].y / cellSz), 0, kGridHeight-1);
            int gc  = gcx + gcy * kGridWidth;
            blocked[gc] = false;   // exit cell always passable
            if (cost[gc] > 0.f) { cost[gc] = 0.f; pq.push({0.f, gc}); }
        }
    }

    // Single-goal fallback (non-evacuation flow field)
    if (_flowGoalSet && _evacExitCount == 0) {
        int gcx = std::clamp((int)(_flowGoalX / cellSz), 0, kGridWidth -1);
        int gcy = std::clamp((int)(_flowGoalY / cellSz), 0, kGridHeight-1);
        int gc  = gcx + gcy * kGridWidth;
        blocked[gc] = false;
        cost[gc]    = 0.f;
        pq.push({0.f, gc});
    }

    while (!pq.empty()) {
        auto [d, c] = pq.top(); pq.pop();
        if (d > cost[c]) continue;
        int cx = c % kGridWidth, cy = c / kGridWidth;
        for (int i = 0; i < 8; i++) {
            int nx = cx+NDX[i], ny = cy+NDY[i];
            if (nx < 0 || nx >= kGridWidth || ny < 0 || ny >= kGridHeight) continue;
            int nc = nx + ny*kGridWidth;
            if (blocked[nc]) continue;
            if (NDX[i] != 0 && NDY[i] != 0)
                if (blocked[(cx+NDX[i]) + cy*kGridWidth] ||
                    blocked[cx + (cy+NDY[i])*kGridWidth]) continue;
            float nd = d + NDC[i];
            if (nd < cost[nc]) { cost[nc] = nd; pq.push({nd, nc}); }
        }
    }

    // ── Derive flow vectors: steepest descent on cost field ───────────────
    struct FlowVec { float x, y; };
    FlowVec *ff = (FlowVec *)_flowFieldBuffer.contents;

    for (int c = 0; c < kNumCells; c++) {
        if (blocked[c] || cost[c] >= 1e29f) { ff[c] = {0.f,0.f}; continue; }
        int cx = c % kGridWidth, cy = c / kGridWidth;
        float best = cost[c];
        float bx = 0.f, by = 0.f;
        for (int i = 0; i < 8; i++) {
            int nx = cx+NDX[i], ny = cy+NDY[i];
            if (nx < 0 || nx >= kGridWidth || ny < 0 || ny >= kGridHeight) continue;
            if (cost[nx + ny*kGridWidth] < best) {
                best = cost[nx + ny*kGridWidth];
                bx = (float)NDX[i];
                by = (float)NDY[i];
            }
        }
        float len = std::sqrtf(bx*bx + by*by);
        ff[c] = len > 0.f ? FlowVec{bx/len, by/len} : FlowVec{0.f,0.f};
    }
}

// ── Evacuation API ────────────────────────────────────────────────────────────

- (void)addEvacExit:(float)x y:(float)y radius:(float)radius {
    if (_evacExitCount >= kMaxEvacExits) return;
    struct EvacExit *ex = (struct EvacExit *)_exitBuffer.contents;
    ex[_evacExitCount++] = { x, y, radius, 0.f };
    SimParams *p = (SimParams *)_params.contents;
    p->exitCount = (int)_evacExitCount;
    _flowFieldDirty = YES;
}

- (void)clearEvacExits {
    _evacExitCount = 0;
    ((SimParams *)_params.contents)->exitCount = 0;
    memset(_exitBuffer.contents, 0, kMaxEvacExits * sizeof(struct EvacExit));
    _flowFieldDirty = YES;
}

- (void)spawnEvacAgents:(uint32_t)count {
    if (count == 0 || count > _agentCount) count = _agentCount;

    // Build blocked map (same logic as recomputeFlowField)
    const float cellSz = Simulation::kNeighborRadius;
    const float blockR = cellSz * 0.5f;
    bool blocked[kNumCells] = {};
    if (_obstacleCount > 0) {
        const struct Obstacle *obs = (const struct Obstacle *)_obstacleBuffer.contents;
        for (int c = 0; c < kNumCells; c++) {
            float cx = ((c % kGridWidth) + 0.5f) * cellSz;
            float cy = ((c / kGridWidth) + 0.5f) * cellSz;
            for (uint32_t oi = 0; oi < _obstacleCount && !blocked[c]; oi++) {
                float ax = obs[oi].x0, ay = obs[oi].y0;
                float bx = obs[oi].x1, by = obs[oi].y1;
                float abx = bx-ax, aby = by-ay;
                float len2 = abx*abx + aby*aby;
                float t = len2 > 0.f
                    ? std::clamp(((cx-ax)*abx + (cy-ay)*aby) / len2, 0.f, 1.f)
                    : 0.f;
                float dx = cx-(ax+t*abx), dy = cy-(ay+t*aby);
                blocked[c] = (dx*dx + dy*dy) < blockR*blockR;
            }
        }
    }
    // Collect spawn-eligible cells
    std::vector<int> freeCells;
    freeCells.reserve(kNumCells);
    for (int c = 0; c < kNumCells; c++)
        if (!blocked[c]) freeCells.push_back(c);

    if (freeCells.empty()) return;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(0.f, cellSz);
    std::uniform_int_distribution<int>    pickCell(0, (int)freeCells.size() - 1);
    std::uniform_real_distribution<float> spd(80.f, 120.f);

    float *pX  = (float *)_posX.contents,  *pY  = (float *)_posY.contents;
    float *vX  = (float *)_velX.contents,  *vY  = (float *)_velY.contents;
    float *ms  = (float *)_maxSpeed.contents;
    float *fX  = (float *)_forceX.contents, *fY = (float *)_forceY.contents;
    float *act = (float *)_activeBuffer.contents;
    float *ext = (float *)_exitTimeBuffer.contents;

    for (uint32_t i = 0; i < count; i++) {
        int   c  = freeCells[pickCell(rng)];
        float cx = (c % kGridWidth) * cellSz + jitter(rng);
        float cy = (c / kGridWidth) * cellSz + jitter(rng);
        pX[i] = std::clamp(cx, 0.f, Simulation::kWorldWidth);
        pY[i] = std::clamp(cy, 0.f, Simulation::kWorldHeight);
        vX[i] = 0.f; vY[i] = 0.f;
        ms[i] = spd(rng);
        fX[i] = 0.f; fY[i] = 0.f;
        act[i] = 1.f;
        ext[i] = 0.f;
    }
    // Agents beyond `count` are inactive (invisible)
    for (uint32_t i = count; i < _agentCount; i++) {
        act[i] = 0.f;
        pX[i] = -9999.f; pY[i] = -9999.f;
    }

    *((uint32_t *)_liveCountBuffer.contents) = count;
    _evacSpawnCount  = count;
    _evacSpawnFrame  = _frameIndex;
    _evacuationComplete = NO;

    SimParams *p      = (SimParams *)_params.contents;
    p->agentCount     = (int)_agentCount;  // GPU always dispatches full N
    _evacuationMode   = YES;
    p->evacuationMode = 1;
    _useFlowField     = YES;
    p->useFlowField   = 1;
    _useORCA          = YES;
    p->useORCA        = 1;
    if (_evacExitCount > 0) [self recomputeFlowField];
}

- (void)resetEvacuation {
    _evacuationMode = NO;
    ((SimParams *)_params.contents)->evacuationMode = 0;
    _evacuationComplete = NO;
    float *act = (float *)_activeBuffer.contents;
    for (uint32_t i = 0; i < _agentCount; i++) act[i] = 1.f;
    *((uint32_t *)_liveCountBuffer.contents) = _agentCount;
}

- (EvacMetrics)computeEvacMetrics {
    EvacMetrics m = {};
    m.agentsSpawned = _evacSpawnCount;
    float *ext = (float *)_exitTimeBuffer.contents;
    float *act = (float *)_activeBuffer.contents;
    double sumTime = 0.0;
    uint32_t exited = 0;
    float dt = ((SimParams *)_params.contents)->dt;
    for (uint32_t i = 0; i < _evacSpawnCount; i++) {
        if (act[i] < 0.5f && ext[i] > 0.f) {
            double t = (ext[i] - (float)_evacSpawnFrame) * (double)dt;
            sumTime += t;
            exited++;
            if (t > m.totalTimeSeconds) m.totalTimeSeconds = t;
        }
    }
    m.agentsExited    = exited;
    m.avgTimeSeconds  = exited > 0 ? sumTime / exited : 0.0;
    return m;
}

// ── Per-frame encode ──────────────────────────────────────────────────────────

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt {
    if (!_integratePipeline) return;

    // Recompute flow field if obstacles changed since last encode
    if (_flowFieldDirty) {
        [self recomputeFlowField];
        _flowFieldDirty = NO;
    }

    SimParams *p  = (SimParams *)_params.contents;
    p->dt         = dt;
    p->frameIndex = _frameIndex++;

    MTLSize agentGrid = MTLSizeMake(_agentCount, 1, 1);
    MTLSize tg64      = MTLSizeMake(64, 1, 1);

    if (_useGridSteering) {
        if (!_steerGridPipeline) return;

        MTLSize cellGrid = MTLSizeMake(kNumCells, 1, 1);
        MTLSize prefGrid = MTLSizeMake(1, 1, 1);
        MTLSize tg1      = MTLSizeMake(1, 1, 1);

        // Pass 1 — zero per-cell counters
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_clearGridPipeline];
            [enc setBuffer:_cellCount offset:0 atIndex:0];
            [enc setBuffer:_insertPos offset:0 atIndex:1];
            [enc setBuffer:_params    offset:0 atIndex:2];
            [enc dispatchThreads:cellGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
        // Pass 2 — assign cell IDs, count agents per cell
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_hashPipeline];
            [enc setBuffer:_posX      offset:0 atIndex:0];
            [enc setBuffer:_posY      offset:0 atIndex:1];
            [enc setBuffer:_cellID    offset:0 atIndex:2];
            [enc setBuffer:_cellCount offset:0 atIndex:3];
            [enc setBuffer:_params    offset:0 atIndex:4];
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
        // Pass 3 — prefix sum → cellStart; seed insertPos (1 thread)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_prefixSumPipeline];
            [enc setBuffer:_cellCount  offset:0 atIndex:0];
            [enc setBuffer:_cellStart  offset:0 atIndex:1];
            [enc setBuffer:_insertPos  offset:0 atIndex:2];
            [enc setBuffer:_params     offset:0 atIndex:3];
            [enc dispatchThreads:prefGrid threadsPerThreadgroup:tg1];
            [enc endEncoding];
        }
        // Pass 4 — scatter agents into sorted positions
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_scatterPipeline];
            [enc setBuffer:_cellID           offset:0 atIndex:0];
            [enc setBuffer:_insertPos        offset:0 atIndex:1];
            [enc setBuffer:_sortedAgentIndex offset:0 atIndex:2];
            [enc setBuffer:_params           offset:0 atIndex:3];
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
        // Pass 5 — gather SoA into sorted order (includes active flag)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_reorderPipeline];
            [enc setBuffer:_sortedAgentIndex offset:0 atIndex:0];
            [enc setBuffer:_posX             offset:0 atIndex:1];
            [enc setBuffer:_posY             offset:0 atIndex:2];
            [enc setBuffer:_velX             offset:0 atIndex:3];
            [enc setBuffer:_velY             offset:0 atIndex:4];
            [enc setBuffer:_maxSpeed         offset:0 atIndex:5];
            [enc setBuffer:_sPosX            offset:0 atIndex:6];
            [enc setBuffer:_sPosY            offset:0 atIndex:7];
            [enc setBuffer:_sVelX            offset:0 atIndex:8];
            [enc setBuffer:_sVelY            offset:0 atIndex:9];
            [enc setBuffer:_sMaxSpeed        offset:0 atIndex:10];
            [enc setBuffer:_params           offset:0 atIndex:11];
            [enc setBuffer:_activeBuffer     offset:0 atIndex:12];
            [enc setBuffer:_sActive          offset:0 atIndex:13];
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
        // Pass 6 — steering: ORCA, tiled grid, or untiled grid
        {
            id<MTLComputePipelineState> steerPS;
            if (_useORCA && _orcaPipeline)
                steerPS = _orcaPipeline;
            else if (!_useTiling && _noTiledPipeline)
                steerPS = _noTiledPipeline;
            else
                steerPS = _steerGridPipeline;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:steerPS];
            [enc setBuffer:_sPosX            offset:0 atIndex:0];
            [enc setBuffer:_sPosY            offset:0 atIndex:1];
            [enc setBuffer:_sVelX            offset:0 atIndex:2];
            [enc setBuffer:_sVelY            offset:0 atIndex:3];
            [enc setBuffer:_sMaxSpeed        offset:0 atIndex:4];
            [enc setBuffer:_targetX          offset:0 atIndex:5];
            [enc setBuffer:_targetY          offset:0 atIndex:6];
            [enc setBuffer:_forceX           offset:0 atIndex:7];
            [enc setBuffer:_forceY           offset:0 atIndex:8];
            [enc setBuffer:_cellStart        offset:0 atIndex:9];
            [enc setBuffer:_cellCount        offset:0 atIndex:10];
            [enc setBuffer:_sortedAgentIndex offset:0 atIndex:11];
            [enc setBuffer:_params           offset:0 atIndex:12];
            [enc setBuffer:_obstacleBuffer   offset:0 atIndex:13];
            [enc setBuffer:_flowFieldBuffer  offset:0 atIndex:14];
            [enc setBuffer:_sActive          offset:0 atIndex:15];
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
    } else {
        // Phase 2 O(N²) path — used by Phase 4 benchmark for comparison
        if (!_steerPipeline) return;
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_steerPipeline];
            [enc setBuffer:_posX     offset:0 atIndex:0];
            [enc setBuffer:_posY     offset:0 atIndex:1];
            [enc setBuffer:_velX     offset:0 atIndex:2];
            [enc setBuffer:_velY     offset:0 atIndex:3];
            [enc setBuffer:_targetX  offset:0 atIndex:4];
            [enc setBuffer:_targetY  offset:0 atIndex:5];
            [enc setBuffer:_maxSpeed offset:0 atIndex:6];
            [enc setBuffer:_forceX   offset:0 atIndex:7];
            [enc setBuffer:_forceY   offset:0 atIndex:8];
            [enc setBuffer:_params   offset:0 atIndex:9];
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
    }

    // Pass 7 — integration (all paths)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_integratePipeline];
        [enc setBuffer:_posX         offset:0 atIndex:0];
        [enc setBuffer:_posY         offset:0 atIndex:1];
        [enc setBuffer:_velX         offset:0 atIndex:2];
        [enc setBuffer:_velY         offset:0 atIndex:3];
        [enc setBuffer:_forceX       offset:0 atIndex:4];
        [enc setBuffer:_forceY       offset:0 atIndex:5];
        [enc setBuffer:_maxSpeed     offset:0 atIndex:6];
        [enc setBuffer:_params       offset:0 atIndex:7];
        [enc setBuffer:_activeBuffer offset:0 atIndex:8];
        [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
        [enc endEncoding];
    }

    // Pass 8 — exit detection (evacuation mode only)
    if (_evacuationMode && !_evacuationComplete && _checkExitPipeline && _evacExitCount > 0) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_checkExitPipeline];
        [enc setBuffer:_posX            offset:0 atIndex:0];
        [enc setBuffer:_posY            offset:0 atIndex:1];
        [enc setBuffer:_activeBuffer    offset:0 atIndex:2];
        [enc setBuffer:_exitTimeBuffer  offset:0 atIndex:3];
        [enc setBuffer:_liveCountBuffer offset:0 atIndex:4];
        [enc setBuffer:_exitBuffer      offset:0 atIndex:5];
        [enc setBuffer:_params          offset:0 atIndex:6];
        [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
        [enc endEncoding];

        // Check completion on CPU after GPU finishes (read from previous frame — 1 frame lag)
        if (!_evacuationComplete) {
            uint32_t live = *((uint32_t *)_liveCountBuffer.contents);
            if (live == 0 && _evacSpawnCount > 0) {
                _evacuationComplete = YES;
                if (self.evacuationCompleteCallback) {
                    EvacMetrics m = [self computeEvacMetrics];
                    self.evacuationCompleteCallback(m);
                }
            }
        }
    }
}

@end
