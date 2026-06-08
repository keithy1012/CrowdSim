#import "GPUSimulation.h"
#include "Simulation.h"    // for world/behavior constants
#include "SharedTypes.h"
#include <random>

@implementation GPUSimulation {
    id<MTLDevice>               _device;
    id<MTLComputePipelineState> _steerPipeline;
    id<MTLComputePipelineState> _integratePipeline;

    // SoA agent data — shared storage; CPU writes once at init, GPU owns thereafter
    id<MTLBuffer> _posX, _posY;
    id<MTLBuffer> _velX, _velY;
    id<MTLBuffer> _targetX, _targetY;
    id<MTLBuffer> _maxSpeed;
    id<MTLBuffer> _radBuffer;
    id<MTLBuffer> _forceX, _forceY;
    id<MTLBuffer> _params;

    uint32_t _agentCount;
    int      _frameIndex;
}

// Map public property names to internal ivar names
@synthesize agentCount = _agentCount;
@synthesize posXBuffer = _posX;
@synthesize posYBuffer = _posY;
@synthesize radBuffer  = _radBuffer;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                    agentCount:(uint32_t)count {
    self = [super init];
    if (!self) return nil;
    _device     = device;
    _agentCount = count;
    [self buildPipelines:library];
    [self allocateAndInitBuffers];
    return self;
}

// ── Pipeline setup ────────────────────────────────────────────────────────────

- (void)buildPipelines:(id<MTLLibrary>)library {
    NSError *err = nil;

    id<MTLFunction> steerFn = [library newFunctionWithName:@"k_steer"];
    if (steerFn) {
        _steerPipeline = [_device newComputePipelineStateWithFunction:steerFn error:&err];
        if (err) NSLog(@"[GPUSim] k_steer: %@", err);
    } else NSLog(@"[GPUSim] k_steer not found in library");

    err = nil;
    id<MTLFunction> integrateFn = [library newFunctionWithName:@"k_integrate"];
    if (integrateFn) {
        _integratePipeline = [_device newComputePipelineStateWithFunction:integrateFn error:&err];
        if (err) NSLog(@"[GPUSim] k_integrate: %@", err);
    } else NSLog(@"[GPUSim] k_integrate not found in library");
}

// ── Buffer allocation + CPU initialisation ────────────────────────────────────

- (id<MTLBuffer>)sharedBuf:(NSUInteger)bytes {
    return [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

- (void)allocateAndInitBuffers {
    const NSUInteger sz = _agentCount * sizeof(float);

    _posX     = [self sharedBuf:sz];
    _posY     = [self sharedBuf:sz];
    _velX     = [self sharedBuf:sz];
    _velY     = [self sharedBuf:sz];
    _targetX  = [self sharedBuf:sz];
    _targetY  = [self sharedBuf:sz];
    _maxSpeed = [self sharedBuf:sz];
    _radBuffer= [self sharedBuf:sz];
    _forceX   = [self sharedBuf:sz];
    _forceY   = [self sharedBuf:sz];
    _params   = [_device newBufferWithLength:sizeof(SimParams)
                                     options:MTLResourceStorageModeShared];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(0.f, Simulation::kWorldWidth);
    std::uniform_real_distribution<float> ry(0.f, Simulation::kWorldHeight);
    std::uniform_real_distribution<float> spd(80.f, 120.f);

    float *pX  = (float *)_posX.contents,     *pY  = (float *)_posY.contents;
    float *vX  = (float *)_velX.contents,     *vY  = (float *)_velY.contents;
    float *tX  = (float *)_targetX.contents,  *tY  = (float *)_targetY.contents;
    float *ms  = (float *)_maxSpeed.contents;
    float *rad = (float *)_radBuffer.contents;
    float *fX  = (float *)_forceX.contents,   *fY  = (float *)_forceY.contents;

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
}

// ── Per-frame encode ──────────────────────────────────────────────────────────

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt {
    if (!_steerPipeline || !_integratePipeline) return;

    SimParams *p  = (SimParams *)_params.contents;
    p->dt         = dt;
    p->frameIndex = _frameIndex++;

    MTLSize grid = MTLSizeMake(_agentCount, 1, 1);
    MTLSize tg   = MTLSizeMake(64, 1, 1);

    // Pass 1 — Steering: O(N²) neighbour scan, writes forceX/Y and possibly targetX/Y
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
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // Pass 2 — Integration: reads forceX/Y, writes posX/Y and velX/Y
    // Separate encoder guarantees all steer writes are visible before integration reads.
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
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
}

@end
