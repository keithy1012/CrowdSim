#import "Renderer.h"
#import "GPUSimulation.h"
#include "Simulation.h"
#import <QuartzCore/QuartzCore.h>

// CPU-path buffer capacity (Phase 1 / Phase 4 benchmarking)
static const uint32_t kMaxAgentsCPU = 50000;

@interface Renderer ()
- (void)buildAgentPipelineWithView:(MTKView *)view;
- (void)buildObstaclePipelineWithView:(MTKView *)view;
@end

@implementation Renderer {
    id<MTLDevice>              _device;
    id<MTLCommandQueue>        _commandQueue;
    id<MTLLibrary>             _library;
    id<MTLRenderPipelineState> _agentPipeline;
    id<MTLRenderPipelineState> _obstaclePipeline;

    // CPU-path render buffers (separate SoA, matches shader buffer layout)
    id<MTLBuffer> _posXBuffer;
    id<MTLBuffer> _posYBuffer;
    id<MTLBuffer> _radBuffer;
    id<MTLBuffer> _vpBuffer;
    // CPU-path velocity stand-ins: zeros for vel, 1.0 for maxSpeed → agents render blue
    id<MTLBuffer> _cpuVelXBuffer;
    id<MTLBuffer> _cpuVelYBuffer;
    id<MTLBuffer> _cpuMaxSpeedBuffer;

    // Active simulation — at most one is non-nil
    Simulation     *_sim;       // CPU path (weak, owned by AppDelegate)
    GPUSimulation  *_gpuSim;    // GPU path (strong)

    CFTimeInterval _lastTime;
    uint32_t       _frameCount;
    CFTimeInterval _fpsTimer;
}

- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;

    _device       = view.device;
    _commandQueue = [_device newCommandQueue];

    NSURL *libURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    if (libURL) {
        NSError *err = nil;
        _library = [_device newLibraryWithURL:libURL error:&err];
        if (err) NSLog(@"[Renderer] Library error: %@", err);
    } else {
        NSLog(@"[Renderer] default.metallib not found in bundle.");
    }

    if (_library) {
        [self buildAgentPipelineWithView:view];
        [self buildObstaclePipelineWithView:view];
    }

    NSUInteger sz = kMaxAgentsCPU * sizeof(float);
    _posXBuffer = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    _posYBuffer = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    _radBuffer  = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    _vpBuffer   = [_device newBufferWithLength:sizeof(float) * 2
                                       options:MTLResourceStorageModeShared];

    // CPU-path velocity stand-ins: vel=0 → speed=0 → all agents appear blue
    _cpuVelXBuffer     = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    _cpuVelYBuffer     = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    _cpuMaxSpeedBuffer = [_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    memset(_cpuVelXBuffer.contents,     0, sz);
    memset(_cpuVelYBuffer.contents,     0, sz);
    float *msPtr = (float *)_cpuMaxSpeedBuffer.contents;
    for (uint32_t i = 0; i < kMaxAgentsCPU; i++) msPtr[i] = 1.f;

    view.clearColor               = MTLClearColorMake(0.05, 0.05, 0.10, 1.0);
    view.preferredFramesPerSecond = 60;

    return self;
}

- (void)buildObstaclePipelineWithView:(MTKView *)view {
    id<MTLFunction> vsFn = [_library newFunctionWithName:@"vs_obstacle"];
    id<MTLFunction> fsFn = [_library newFunctionWithName:@"fs_obstacle"];
    if (!vsFn || !fsFn) { NSLog(@"[Renderer] vs_obstacle / fs_obstacle not found."); return; }

    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction               = vsFn;
    pd.fragmentFunction             = fsFn;
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;

    NSError *err = nil;
    _obstaclePipeline = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (err) NSLog(@"[Renderer] Obstacle pipeline error: %@", err);
}

- (void)buildAgentPipelineWithView:(MTKView *)view {
    id<MTLFunction> vsFn = [_library newFunctionWithName:@"vs_agent"];
    id<MTLFunction> fsFn = [_library newFunctionWithName:@"fs_agent"];
    if (!vsFn || !fsFn) { NSLog(@"[Renderer] vs_agent / fs_agent not found."); return; }

    MTLRenderPipelineDescriptor *pd  = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction                = vsFn;
    pd.fragmentFunction              = fsFn;
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;

    NSError *err = nil;
    _agentPipeline = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (err) NSLog(@"[Renderer] Pipeline error: %@", err);
}

// ── Simulation wiring ─────────────────────────────────────────────────────────

- (void)setSimulation:(Simulation *)sim {
    _sim    = sim;
    _gpuSim = nil;

    if (!_sim) return;
    const auto &a = _sim->agents();
    uint32_t n = std::min(a.count(), kMaxAgentsCPU);
    float *rad = (float *)_radBuffer.contents;
    for (uint32_t i = 0; i < n; i++) rad[i] = a.radius[i];
}

- (void)setGPUSimulation:(GPUSimulation *)gpuSim {
    _gpuSim = gpuSim;
    _sim    = nil;
}

// ── Render loop ───────────────────────────────────────────────────────────────

- (void)drawInMTKView:(MTKView *)view {
    CFTimeInterval now = CACurrentMediaTime();
    float dt = (_lastTime > 0.0) ? (float)(now - _lastTime) : (1.f / 60.f);
    _lastTime = now;
    dt = fminf(dt, 0.05f);

    _frameCount++;
    if (now - _fpsTimer >= 1.0) {
        uint32_t n = _gpuSim ? _gpuSim.agentCount : (_sim ? _sim->agents().count() : 0);
        NSLog(@"[CrowdSim] %u agents  %.0f FPS  [%@]",
              n, _frameCount / (now - _fpsTimer),
              _gpuSim ? @"GPU" : @"CPU");
        _frameCount = 0;
        _fpsTimer   = now;
    }

    id<MTLCommandBuffer> cmd = [_commandQueue commandBuffer];

    if (_gpuSim) {
        // GPU path: encode both compute passes before the render pass
        [_gpuSim encodeToCommandBuffer:cmd dt:dt];
    } else if (_sim) {
        // CPU path: step on CPU, then upload positions
        _sim->update(dt);
        const auto &a  = _sim->agents();
        uint32_t    n  = std::min(a.count(), kMaxAgentsCPU);
        float *pX = (float *)_posXBuffer.contents;
        float *pY = (float *)_posYBuffer.contents;
        for (uint32_t i = 0; i < n; i++) { pX[i] = a.posX[i]; pY[i] = a.posY[i]; }
    }

    // Upload viewport in logical points so world coords [0,1280]×[0,720] fill the window.
    // drawableSize is in physical pixels (2× on Retina) — using it would confine agents
    // to the top-left quarter of the screen.
    float vp[2] = { (float)view.bounds.size.width, (float)view.bounds.size.height };
    memcpy(_vpBuffer.contents, vp, sizeof(vp));

    // Render pass
    MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
    if (!rpd || !view.currentDrawable) { [cmd commit]; return; }

    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];

    uint32_t agentCount = _gpuSim ? _gpuSim.agentCount
                                  : (_sim ? (uint32_t)_sim->agents().count() : 0);

    if (_agentPipeline && agentCount > 0) {
        // GPU sim owns its position/velocity buffers; CPU path uses local staging buffers
        id<MTLBuffer> pxBuf = _gpuSim ? _gpuSim.posXBuffer     : _posXBuffer;
        id<MTLBuffer> pyBuf = _gpuSim ? _gpuSim.posYBuffer     : _posYBuffer;
        id<MTLBuffer> rBuf  = _gpuSim ? _gpuSim.radBuffer      : _radBuffer;
        id<MTLBuffer> vxBuf = _gpuSim ? _gpuSim.velXBuffer     : _cpuVelXBuffer;
        id<MTLBuffer> vyBuf = _gpuSim ? _gpuSim.velYBuffer     : _cpuVelYBuffer;
        id<MTLBuffer> msBuf = _gpuSim ? _gpuSim.maxSpeedBuffer : _cpuMaxSpeedBuffer;

        [enc setRenderPipelineState:_agentPipeline];
        [enc setVertexBuffer:pxBuf     offset:0 atIndex:0];
        [enc setVertexBuffer:pyBuf     offset:0 atIndex:1];
        [enc setVertexBuffer:rBuf      offset:0 atIndex:2];
        [enc setVertexBuffer:_vpBuffer offset:0 atIndex:3];
        [enc setVertexBuffer:vxBuf     offset:0 atIndex:4];
        [enc setVertexBuffer:vyBuf     offset:0 atIndex:5];
        [enc setVertexBuffer:msBuf     offset:0 atIndex:6];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip
                vertexStart:0
                vertexCount:4
              instanceCount:agentCount];
    }

    // Draw obstacle line segments on top of agents
    if (_obstaclePipeline && _gpuSim && _gpuSim.obstacleCount > 0) {
        [enc setRenderPipelineState:_obstaclePipeline];
        [enc setVertexBuffer:_gpuSim.obstacleBuffer offset:0 atIndex:0];
        [enc setVertexBuffer:_vpBuffer              offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeLine
                vertexStart:0
                vertexCount:_gpuSim.obstacleCount * 2];
    }

    [enc endEncoding];
    [cmd presentDrawable:view.currentDrawable];
    [cmd commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}

@end
