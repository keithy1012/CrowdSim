#pragma once
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#ifdef __cplusplus
class Simulation;
#endif

@interface Renderer : NSObject <MTKViewDelegate>
- (instancetype)initWithView:(MTKView *)view;
#ifdef __cplusplus
- (void)setSimulation:(Simulation *)sim;
#endif
@end
