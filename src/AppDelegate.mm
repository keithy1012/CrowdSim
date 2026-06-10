#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"
#import "GPUSimulation.h"

static constexpr uint32_t kDefaultAgentCount = 100000;

// ── Scenario definitions ──────────────────────────────────────────────────────

static const Obstacle kScenario2[] = {
    {0.f, 240.f, 768.f, 240.f},
    {512.f, 480.f, 1280.f, 480.f},
};

static const Obstacle kScenario3[] = {
    {247.f, 220.f, 287.f, 220.f}, {287.f, 220.f, 287.f, 260.f},
    {287.f, 260.f, 247.f, 260.f}, {247.f, 260.f, 247.f, 220.f},
    {620.f, 220.f, 660.f, 220.f}, {660.f, 220.f, 660.f, 260.f},
    {660.f, 260.f, 620.f, 260.f}, {620.f, 260.f, 620.f, 220.f},
    {993.f, 220.f, 1033.f, 220.f}, {1033.f, 220.f, 1033.f, 260.f},
    {1033.f, 260.f, 993.f, 260.f}, {993.f, 260.f, 993.f, 220.f},
    {247.f, 460.f, 287.f, 460.f}, {287.f, 460.f, 287.f, 500.f},
    {287.f, 500.f, 247.f, 500.f}, {247.f, 500.f, 247.f, 460.f},
    {620.f, 460.f, 660.f, 460.f}, {660.f, 460.f, 660.f, 500.f},
    {660.f, 500.f, 620.f, 500.f}, {620.f, 500.f, 620.f, 460.f},
    {993.f, 460.f, 1033.f, 460.f}, {1033.f, 460.f, 1033.f, 500.f},
    {1033.f, 500.f, 993.f, 500.f}, {993.f, 500.f, 993.f, 460.f},
};

// ─────────────────────────────────────────────────────────────────────────────

static const float kMinSegLen = 12.f;

typedef NS_ENUM(int, EvacPhase) {
    kEvacOff = 0,
    kEvacIdle,      // designing layout: draw walls, place exits, set agent count
    kEvacRunning,   // simulation in progress
    kEvacComplete   // all agents exited; showing results
};

@implementation AppDelegate {
    NSWindow      *_window;
    MTKView       *_view;
    Renderer      *_renderer;
    GPUSimulation *_gpuSim;

    // Normal-mode drawing state
    NSPoint _lastDrawPt;
    BOOL    _isDrawing;
    int     _currentScenario;

    // Evacuation mode state
    EvacPhase  _evacPhase;
    uint32_t   _evacAgentCount;  // agents to spawn (default 500)
    NSTimer   *_hudTimer;        // updates window title ~4×/sec
}

// ── Coordinate conversion ─────────────────────────────────────────────────────

- (NSPoint)worldPointFromEvent:(NSEvent *)event {
    NSPoint vp = [_view convertPoint:event.locationInWindow fromView:nil];
    return NSMakePoint(vp.x, _view.bounds.size.height - vp.y);
}

// ── Window title helpers ──────────────────────────────────────────────────────

- (void)updateNormalModeTitle {
    [_window setTitle:@"CrowdSim  ·  1/2/3: scene  ·  LDrag: draw  ·  RClick: flow goal"
     "  ·  F: flow  ·  O: ORCA  ·  C: clear  ·  E: evacuation mode"];
}

- (void)updateEvacIdleTitle {
    [_window setTitle:[NSString stringWithFormat:
        @"EVAC DESIGN  ·  LDrag: walls  ·  RClick: exit (%.0fpx)  ·  +/-: agents (%u)"
         "  ·  Space: start  ·  C: clear  ·  E: exit evac mode",
        50.f, _evacAgentCount]];
}

- (void)updateEvacRunningTitle {
    [_window setTitle:[NSString stringWithFormat:
        @"EVAC RUNNING  ·  Agents: %u / %u  ·  E: exit mode",
        _gpuSim.liveAgentCount, _evacAgentCount]];
}

- (void)updateEvacCompleteTitle:(EvacMetrics)m {
    [_window setTitle:[NSString stringWithFormat:
        @"EVAC COMPLETE  ·  Total: %.1fs  ·  Avg: %.1fs  ·  Exited: %u/%u"
         "  ·  R: reset  ·  E: exit mode",
        m.totalTimeSeconds, m.avgTimeSeconds, m.agentsExited, m.agentsSpawned]];
}

// ── Mouse drawing ─────────────────────────────────────────────────────────────

- (void)handleMouseEvent:(NSEvent *)event {
    NSPoint viewPt = [_view convertPoint:event.locationInWindow fromView:nil];
    if (!NSPointInRect(viewPt, _view.bounds)) return;

    switch (event.type) {

    // ── Left button: draw obstacle segments (normal + evac idle) ─────────────
    case NSEventTypeLeftMouseDown:
        if (_evacPhase == kEvacRunning || _evacPhase == kEvacComplete) break;
        _lastDrawPt = [self worldPointFromEvent:event];
        _isDrawing  = YES;
        break;

    case NSEventTypeLeftMouseDragged: {
        if (!_isDrawing) break;
        if (_evacPhase == kEvacRunning || _evacPhase == kEvacComplete) break;
        NSPoint cur = [self worldPointFromEvent:event];
        float dx = (float)(cur.x - _lastDrawPt.x);
        float dy = (float)(cur.y - _lastDrawPt.y);
        if (dx*dx + dy*dy < kMinSegLen * kMinSegLen) break;
        Obstacle seg = {
            (float)_lastDrawPt.x, (float)_lastDrawPt.y,
            (float)cur.x,         (float)cur.y
        };
        [_gpuSim appendObstacle:seg];
        _lastDrawPt = cur;
        break;
    }

    case NSEventTypeLeftMouseUp:
        _isDrawing = NO;
        break;

    // ── Right-click: flow goal (normal) or place exit (evac idle) ────────────
    case NSEventTypeRightMouseDown: {
        NSPoint wp = [self worldPointFromEvent:event];
        if (_evacPhase == kEvacIdle) {
            [_gpuSim addEvacExit:(float)wp.x y:(float)wp.y radius:50.f];
            [self updateEvacIdleTitle];
        } else if (_evacPhase == kEvacOff) {
            [_gpuSim setFlowFieldGoal:(float)wp.x y:(float)wp.y];
            if (!_gpuSim.useFlowField) {
                _gpuSim.useFlowField = YES;
                NSLog(@"[CrowdSim] Flow field ON");
            }
        }
        break;
    }

    default: break;
    }
}

// ── Scenario loading ──────────────────────────────────────────────────────────

- (void)loadScenario:(int)idx {
    _currentScenario = idx;
    switch (idx) {
    case 0: [_gpuSim loadObstacles:NULL          count:0];  break;
    case 1: [_gpuSim loadObstacles:kScenario2    count:2];  break;
    case 2: [_gpuSim loadObstacles:kScenario3    count:24]; break;
    default: return;
    }
}

// ── Evacuation helpers ────────────────────────────────────────────────────────

- (void)enterEvacMode {
    _evacPhase       = kEvacIdle;
    _evacAgentCount  = MIN(500u, _gpuSim.agentCount);
    [_gpuSim clearEvacExits];
    [self updateEvacIdleTitle];

    __weak AppDelegate *ws = self;
    _hudTimer = [NSTimer scheduledTimerWithTimeInterval:0.25
        repeats:YES block:^(NSTimer *) {
        AppDelegate *s = ws;
        if (!s) return;
        if (s->_evacPhase == kEvacRunning)
            [s updateEvacRunningTitle];
    }];
}

- (void)exitEvacMode {
    [_hudTimer invalidate]; _hudTimer = nil;
    _evacPhase = kEvacOff;
    [_gpuSim resetEvacuation];
    [_gpuSim clearEvacExits];
    _gpuSim.evacuationMode = NO;
    _gpuSim.useFlowField   = NO;
    _gpuSim.useORCA        = NO;
    [self updateNormalModeTitle];
}

- (void)startEvacuation {
    if (_evacAgentCount == 0) return;
    _evacPhase = kEvacRunning;

    __weak AppDelegate *ws = self;
    _gpuSim.evacuationCompleteCallback = ^(EvacMetrics m) {
        AppDelegate *s = ws;
        if (!s) return;
        s->_evacPhase = kEvacComplete;
        [s updateEvacCompleteTitle:m];
        NSLog(@"[EVAC] Complete — total %.2fs, avg %.2fs, exited %u/%u",
              m.totalTimeSeconds, m.avgTimeSeconds, m.agentsExited, m.agentsSpawned);
    };

    [_gpuSim spawnEvacAgents:_evacAgentCount];
    [self updateEvacRunningTitle];
}

- (void)resetEvacuation {
    [_gpuSim resetEvacuation];
    _evacPhase = kEvacIdle;
    [self updateEvacIdleTitle];
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSRect frame = NSMakeRect(0, 0, 1280, 720);

    _window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:NSWindowStyleMaskTitled
               | NSWindowStyleMaskClosable
               | NSWindowStyleMaskResizable
               | NSWindowStyleMaskMiniaturizable
        backing:NSBackingStoreBuffered
        defer:NO];

    [_window center];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { NSLog(@"[AppDelegate] Metal not supported."); [NSApp terminate:nil]; return; }

    _view     = [[MTKView alloc] initWithFrame:frame device:device];
    _renderer = [[Renderer alloc] initWithView:_view];

    NSURL *libURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    NSError *err  = nil;
    id<MTLLibrary> library = libURL ? [device newLibraryWithURL:libURL error:&err] : nil;
    if (err) NSLog(@"[AppDelegate] Library error: %@", err);

    uint32_t n = _agentCount > 0 ? _agentCount : kDefaultAgentCount;
    _gpuSim = [[GPUSimulation alloc] initWithDevice:device library:library agentCount:n];

    [_renderer setGPUSimulation:_gpuSim];
    _view.delegate = _renderer;

    [_window setContentView:_view];
    [_window makeKeyAndOrderFront:nil];
    [self updateNormalModeTitle];

    // ── Event monitors ────────────────────────────────────────────────────────
    __weak AppDelegate *weakSelf = self;

    [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                          handler:^NSEvent *(NSEvent *event) {
        AppDelegate *s = weakSelf;
        if (!s) return event;
        NSString *ch = event.charactersIgnoringModifiers;

        // ── E: toggle evacuation mode ─────────────────────────────────────
        if ([ch isEqualToString:@"e"] || [ch isEqualToString:@"E"]) {
            if (s->_evacPhase == kEvacOff) [s enterEvacMode];
            else                           [s exitEvacMode];
            return event;
        }

        // ── Keys active only in evacuation mode ───────────────────────────
        if (s->_evacPhase != kEvacOff) {
            if ([ch isEqualToString:@" "]) {
                if (s->_evacPhase == kEvacIdle) [s startEvacuation];
            } else if ([ch isEqualToString:@"r"] || [ch isEqualToString:@"R"]) {
                if (s->_evacPhase == kEvacComplete || s->_evacPhase == kEvacRunning)
                    [s resetEvacuation];
            } else if ([ch isEqualToString:@"c"] || [ch isEqualToString:@"C"]) {
                if (s->_evacPhase == kEvacIdle) {
                    [s loadScenario:s->_currentScenario];
                    [s->_gpuSim clearEvacExits];
                    [s updateEvacIdleTitle];
                }
            } else if ([ch isEqualToString:@"+"] || [ch isEqualToString:@"="]) {
                if (s->_evacPhase == kEvacIdle) {
                    s->_evacAgentCount = MIN(s->_evacAgentCount + 100, s->_gpuSim.agentCount);
                    [s updateEvacIdleTitle];
                }
            } else if ([ch isEqualToString:@"-"]) {
                if (s->_evacPhase == kEvacIdle && s->_evacAgentCount > 100)
                    s->_evacAgentCount -= 100;
                else if (s->_evacPhase == kEvacIdle) s->_evacAgentCount = 100;
                [s updateEvacIdleTitle];
            }
            return event;
        }

        // ── Normal mode keys ──────────────────────────────────────────────
        if      ([ch isEqualToString:@"1"]) [s loadScenario:0];
        else if ([ch isEqualToString:@"2"]) [s loadScenario:1];
        else if ([ch isEqualToString:@"3"]) [s loadScenario:2];
        else if ([ch isEqualToString:@"c"] || [ch isEqualToString:@"C"])
            [s loadScenario:s->_currentScenario];
        else if ([ch isEqualToString:@"f"] || [ch isEqualToString:@"F"]) {
            s->_gpuSim.useFlowField = !s->_gpuSim.useFlowField;
            NSLog(@"[CrowdSim] Flow field %@", s->_gpuSim.useFlowField ? @"ON" : @"OFF");
        }
        else if ([ch isEqualToString:@"o"] || [ch isEqualToString:@"O"]) {
            s->_gpuSim.useORCA = !s->_gpuSim.useORCA;
            NSLog(@"[CrowdSim] ORCA %@", s->_gpuSim.useORCA ? @"ON" : @"OFF");
        }
        return event;
    }];

    NSEventMask drawMask = NSEventMaskLeftMouseDown
                         | NSEventMaskLeftMouseDragged
                         | NSEventMaskLeftMouseUp
                         | NSEventMaskRightMouseDown;
    [NSEvent addLocalMonitorForEventsMatchingMask:drawMask
                                          handler:^NSEvent *(NSEvent *event) {
        [weakSelf handleMouseEvent:event];
        return event;
    }];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app {
    return YES;
}

@end
