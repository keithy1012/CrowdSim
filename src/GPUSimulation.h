#pragma once
#import <Metal/Metal.h>
#include "SharedTypes.h"

typedef struct {
    double   totalTimeSeconds;   // time from spawn to last agent exiting
    double   avgTimeSeconds;     // mean exit time across all agents
    uint32_t agentsSpawned;
    uint32_t agentsExited;
} EvacMetrics;

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
@property (nonatomic, readonly)  id<MTLBuffer> activeBuffer;    // float[N]: 1=alive, 0=exited
@property (nonatomic, readonly)  id<MTLBuffer> evacExitBuffer;  // EvacExit[kMaxEvacExits]
@property (nonatomic, readonly)  uint32_t      evacExitCount;
// NO = Phase 2 O(N²) path (k_steer); YES = Phase 3 spatial hash path (default)
@property (nonatomic)            BOOL          useGridSteering;
// YES = agents sample the flow field instead of seeking individual targets.
@property (nonatomic)            BOOL          useFlowField;
// YES = k_orca replaces k_steerGrid; agents use velocity-space LP for collision avoidance.
@property (nonatomic)            BOOL          useORCA;
// YES (default) = tiled k_steerGrid with threadgroup memory; NO = untiled baseline.
@property (nonatomic)            BOOL          useTiling;
@property (nonatomic, readonly)  BOOL          flowGoalSet;
@property (nonatomic, readonly)  float         flowGoalX;
@property (nonatomic, readonly)  float         flowGoalY;

// ── Evacuation mode ────────────────────────────────────────────────────────────
// Enabled while the evacuation simulation is running.  Turns on active-flag
// filtering, density-dependent speed, exit detection, and no world-wrapping.
@property (nonatomic)            BOOL          evacuationMode;
// Number of agents still alive (decremented by k_checkExit on the GPU).
@property (nonatomic, readonly)  uint32_t      liveAgentCount;
// YES once liveAgentCount reaches 0.  Latches true; reset by resetEvacuation.
@property (nonatomic, readonly)  BOOL          evacuationComplete;
// Called once on the main thread when the last agent exits.
@property (nonatomic, copy)      void (^evacuationCompleteCallback)(EvacMetrics);

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

// Set the shared navigation goal for all agents when useFlowField is YES.
// Also used by evacuation mode — provide one goal for single-exit flows.

// ── Evacuation API ─────────────────────────────────────────────────────────────

// Add an exit zone at (x,y) with given radius.  Triggers flow-field recompute.
- (void)addEvacExit:(float)x y:(float)y radius:(float)radius;

// Remove all exit zones.
- (void)clearEvacExits;

// Spawn `count` agents in non-blocked grid cells.  Initialises active flags,
// resets liveCount, enables evacuationMode.  Call after exits are placed.
- (void)spawnEvacAgents:(uint32_t)count;

// Clear agents and live-count but keep walls and exits.  Disables evacuationMode.
- (void)resetEvacuation;

// Compute and return metrics from the last completed evacuation.
- (EvacMetrics)computeEvacMetrics;

// Encodes steer + integrate compute passes into cmd.
// Call before the render pass so the GPU reads updated positions.
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt;

@end
