#import "Renderer.h"
#include "Simulation.h"
#import <QuartzCore/QuartzCore.h>

static const uint32_t kMaxAgents = 10000;

@interface Renderer ()
- (void)buildAgentPipelineWithView:(MTKView *)view;
@end

@implementation Renderer {
    id<MTLDevice>              _device;
    id<MTLCommandQueue>        _commandQueue;
    id<MTLLibrary>             _library;
    id<MTLRenderPipelineState> _agentPipeline;

    id<MTLBuffer> _posBuffer;   // packed float2 [x,y] per agent
    id<MTLBuffer> _radBuffer;   // float radius per agent
    id<MTLBuffer> _vpBuffer;    // float2 viewport size

    Simulation    *_sim;        // weak — owned by AppDelegate
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

    if (_library) [self buildAgentPipelineWithView:view];

    _posBuffer = [_device newBufferWithLength:kMaxAgents * sizeof(float) * 2
                                      options:MTLResourceStorageModeShared];
    _radBuffer = [_device newBufferWithLength:kMaxAgents * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    _vpBuffer  = [_device newBufferWithLength:sizeof(float) * 2
                                      options:MTLResourceStorageModeShared];

    view.clearColor               = MTLClearColorMake(0.05, 0.05, 0.10, 1.0);
    view.preferredFramesPerSecond = 60;

    return self;
}

- (void)buildAgentPipelineWithView:(MTKView *)view {
    id<MTLFunction> vsFn = [_library newFunctionWithName:@"vs_agent"];
    id<MTLFunction> fsFn = [_library newFunctionWithName:@"fs_agent"];
    if (!vsFn || !fsFn) { NSLog(@"[Renderer] Shader functions not found."); return; }

    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction                            = vsFn;
    pd.fragmentFunction                          = fsFn;
    pd.colorAttachments[0].pixelFormat           = view.colorPixelFormat;

    NSError *err = nil;
    _agentPipeline = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (err) NSLog(@"[Renderer] Pipeline error: %@", err);
}

- (void)setSimulation:(Simulation *)sim {
    _sim = sim;

    // Radii are constant for Phase 1 — upload once
    if (_sim) {
        const auto &agents = _sim->agents();
        uint32_t n = std::min(agents.count(), kMaxAgents);
        float *rad = (float *)_radBuffer.contents;
        for (uint32_t i = 0; i < n; i++) rad[i] = agents.radius[i];
    }
}

- (void)drawInMTKView:(MTKView *)view {
    CFTimeInterval now = CACurrentMediaTime();
    float dt = (_lastTime > 0.0) ? (float)(now - _lastTime) : (1.f / 60.f);
    _lastTime = now;
    dt = fminf(dt, 0.05f);  // guard against large dt on resume

    // FPS log every second
    _frameCount++;
    if (now - _fpsTimer >= 1.0) {
        NSLog(@"[CrowdSim] %u agents  %.0f FPS", _sim ? _sim->agents().count() : 0,
              _frameCount / (now - _fpsTimer));
        _frameCount = 0;
        _fpsTimer   = now;
    }

    // Step simulation
    if (_sim) _sim->update(dt);

    // Upload viewport
    float vp[2] = { (float)view.drawableSize.width, (float)view.drawableSize.height };
    memcpy(_vpBuffer.contents, vp, sizeof(vp));

    // Upload positions
    uint32_t agentCount = 0;
    if (_sim) {
        const auto &agents = _sim->agents();
        agentCount = std::min(agents.count(), kMaxAgents);
        float *pos = (float *)_posBuffer.contents;
        for (uint32_t i = 0; i < agentCount; i++) {
            pos[2 * i]     = agents.posX[i];
            pos[2 * i + 1] = agents.posY[i];
        }
    }

    // Encode
    id<MTLCommandBuffer>     cmd = [_commandQueue commandBuffer];
    MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
    if (!rpd || !view.currentDrawable) { [cmd commit]; return; }

    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];

    if (_agentPipeline && agentCount > 0) {
        [enc setRenderPipelineState:_agentPipeline];
        [enc setVertexBuffer:_posBuffer offset:0 atIndex:0];
        [enc setVertexBuffer:_radBuffer offset:0 atIndex:1];
        [enc setVertexBuffer:_vpBuffer  offset:0 atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip
                vertexStart:0
                vertexCount:4
              instanceCount:agentCount];
    }

    [enc endEncoding];
    [cmd presentDrawable:view.currentDrawable];
    [cmd commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}

@end
