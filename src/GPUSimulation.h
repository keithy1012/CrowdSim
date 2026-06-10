#pragma once
#import <Metal/Metal.h>
#include "SharedTypes.h"

@interface GPUSimulation : NSObject

@property (nonatomic, readonly)  uint32_t      agentCount;
@property (nonatomic, readonly)  uint32_t      obstacleCount;
@property (nonatomic, readonly)  id<MTLBuffer> posXBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> posYBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> radBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> velXBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> velYBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> maxSpeedBuffer;
@property (nonatomic, readonly)  id<MTLBuffer> obstacleBuffer;
// NO = Phase 2 O(N²) path (k_steer); YES = Phase 3 spatial hash path (default)
@property (nonatomic)            BOOL          useGridSteering;
// YES = agents sample the flow field instead of seeking individual targets.
// Right-click sets the goal; right-click again updates it.
@property (nonatomic)            BOOL          useFlowField;
// YES = k_orca replaces k_steerGrid; agents use velocity-space LP for collision avoidance.
@property (nonatomic)            BOOL          useORCA;
// YES (default) = tiled k_steerGrid with threadgroup memory; NO = untiled baseline.
@property (nonatomic)            BOOL          useTiling;
@property (nonatomic, readonly)  BOOL          flowGoalSet;
@property (nonatomic, readonly)  float         flowGoalX;
@property (nonatomic, readonly)  float         flowGoalY;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                    agentCount:(uint32_t)count;

// Replace the active obstacle set (line segments). Pass count=0 to clear.
- (void)loadObstacles:(const struct Obstacle *)obstacles count:(uint32_t)count;

// Set the shared navigation goal for all agents when useFlowField is YES.
// Triggers an immediate BFS over the 26×15 grid; obstacle cells are treated as impassable.
- (void)setFlowFieldGoal:(float)x y:(float)y;

// Append a single user-drawn segment to the current set (no-op if buffer is full).
- (void)appendObstacle:(struct Obstacle)obs;

// Encodes steer + integrate compute passes into cmd.
// Call before the render pass so the GPU reads updated positions.
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt;

@end
