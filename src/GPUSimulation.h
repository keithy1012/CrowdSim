#pragma once
#import <Metal/Metal.h>

@interface GPUSimulation : NSObject

@property (nonatomic, readonly) uint32_t      agentCount;
@property (nonatomic, readonly) id<MTLBuffer> posXBuffer;
@property (nonatomic, readonly) id<MTLBuffer> posYBuffer;
@property (nonatomic, readonly) id<MTLBuffer> radBuffer;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                    agentCount:(uint32_t)count;

// Encodes steer + integrate compute passes into cmd.
// Call before the render pass so the GPU reads updated positions.
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt;

@end
