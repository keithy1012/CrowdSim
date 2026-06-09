#import "GPUSimulation.h"
#include "Simulation.h"    // world/behavior constants
#include "SharedTypes.h"
#include <random>

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

    uint32_t _agentCount;
    uint32_t _obstacleCount;
    int      _frameIndex;
}

@synthesize agentCount      = _agentCount;
@synthesize obstacleCount   = _obstacleCount;
@synthesize posXBuffer      = _posX;
@synthesize posYBuffer      = _posY;
@synthesize radBuffer       = _radBuffer;
@synthesize velXBuffer      = _velX;
@synthesize velYBuffer      = _velY;
@synthesize maxSpeedBuffer  = _maxSpeed;
@synthesize obstacleBuffer  = _obstacleBuffer;
@synthesize useGridSteering = _useGridSteering;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                    agentCount:(uint32_t)count {
    self = [super init];
    if (!self) return nil;
    _device          = device;
    _agentCount      = count;
    _useGridSteering = YES;
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

    _obstacleCapacity = kInitialObstacleCapacity;
    _obstacleBuffer   = [self sharedBuf:_obstacleCapacity * sizeof(struct Obstacle)];
    _obstacleCount    = 0;

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
}

- (void)appendObstacle:(struct Obstacle)obs {
    [self growObstacleBufferTo:_obstacleCount + 1];
    struct Obstacle *buf = (struct Obstacle *)_obstacleBuffer.contents;
    buf[_obstacleCount++] = obs;
    ((SimParams *)_params.contents)->obstacleCount = (int)_obstacleCount;
}

// ── Per-frame encode ──────────────────────────────────────────────────────────

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt {
    if (!_integratePipeline) return;

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
        // Pass 5 — gather SoA into sorted order
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
            [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
            [enc endEncoding];
        }
        // Pass 6 — grid steering (sequential reads via sorted SoA)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:_steerGridPipeline];
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

    // Final pass — integration (same for both paths)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_integratePipeline];
        [enc setBuffer:_posX     offset:0 atIndex:0];
        [enc setBuffer:_posY     offset:0 atIndex:1];
        [enc setBuffer:_velX     offset:0 atIndex:2];
        [enc setBuffer:_velY     offset:0 atIndex:3];
        [enc setBuffer:_forceX   offset:0 atIndex:4];
        [enc setBuffer:_forceY   offset:0 atIndex:5];
        [enc setBuffer:_maxSpeed offset:0 atIndex:6];
        [enc setBuffer:_params   offset:0 atIndex:7];
        [enc dispatchThreads:agentGrid threadsPerThreadgroup:tg64];
        [enc endEncoding];
    }
}

@end
