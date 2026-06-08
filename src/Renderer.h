#pragma once
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#ifdef __cplusplus
class Simulation;
#endif

@class GPUSimulation;

@interface Renderer : NSObject <MTKViewDelegate>
- (instancetype)initWithView:(MTKView *)view;
#ifdef __cplusplus
- (void)setSimulation:(Simulation *)sim;   // Phase 1 / 4 CPU path
#endif
- (void)setGPUSimulation:(GPUSimulation *)gpuSim; // Phase 2+ GPU path
@end
