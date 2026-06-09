#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"
#import "GPUSimulation.h"

static constexpr uint32_t kDefaultAgentCount = 100000;

// ── Scenario definitions ──────────────────────────────────────────────────────

// Scenario 1: Open Field — no obstacles

// Scenario 2: Barrier — two staggered horizontal walls force a zigzag path.
static const Obstacle kScenario2[] = {
    {0.f, 240.f, 768.f, 240.f},
    {512.f, 480.f, 1280.f, 480.f},
};

// Scenario 3: Pillars — 6 square pillars (40×40 px) in a 3×2 grid.
static const Obstacle kScenario3[] = {
    // Pillar (267, 240)
    {247.f, 220.f, 287.f, 220.f}, {287.f, 220.f, 287.f, 260.f},
    {287.f, 260.f, 247.f, 260.f}, {247.f, 260.f, 247.f, 220.f},
    // Pillar (640, 240)
    {620.f, 220.f, 660.f, 220.f}, {660.f, 220.f, 660.f, 260.f},
    {660.f, 260.f, 620.f, 260.f}, {620.f, 260.f, 620.f, 220.f},
    // Pillar (1013, 240)
    {993.f, 220.f, 1033.f, 220.f}, {1033.f, 220.f, 1033.f, 260.f},
    {1033.f, 260.f, 993.f, 260.f}, {993.f, 260.f, 993.f, 220.f},
    // Pillar (267, 480)
    {247.f, 460.f, 287.f, 460.f}, {287.f, 460.f, 287.f, 500.f},
    {287.f, 500.f, 247.f, 500.f}, {247.f, 500.f, 247.f, 460.f},
    // Pillar (640, 480)
    {620.f, 460.f, 660.f, 460.f}, {660.f, 460.f, 660.f, 500.f},
    {660.f, 500.f, 620.f, 500.f}, {620.f, 500.f, 620.f, 460.f},
    // Pillar (1013, 480)
    {993.f, 460.f, 1033.f, 460.f}, {1033.f, 460.f, 1033.f, 500.f},
    {1033.f, 500.f, 993.f, 500.f}, {993.f, 500.f, 993.f, 460.f},
};

// ─────────────────────────────────────────────────────────────────────────────

static const float kMinSegLen = 12.f;  // min px between draw sample points

@implementation AppDelegate {
    NSWindow      *_window;
    MTKView       *_view;
    Renderer      *_renderer;
    GPUSimulation *_gpuSim;

    // Drawing state
    NSPoint _lastDrawPt;
    BOOL    _isDrawing;
    int     _currentScenario;
}

// ── Coordinate conversion ─────────────────────────────────────────────────────

// NSView origin is bottom-left, Y up. World origin is top-left, Y down.
- (NSPoint)worldPointFromEvent:(NSEvent *)event {
    NSPoint vp = [_view convertPoint:event.locationInWindow fromView:nil];
    return NSMakePoint(vp.x, _view.bounds.size.height - vp.y);
}

// ── Mouse drawing ─────────────────────────────────────────────────────────────

- (void)handleMouseEvent:(NSEvent *)event {
    // Ignore events outside the view (e.g. title-bar clicks)
    NSPoint viewPt = [_view convertPoint:event.locationInWindow fromView:nil];
    if (!NSPointInRect(viewPt, _view.bounds)) return;

    switch (event.type) {
    case NSEventTypeLeftMouseDown:
        _lastDrawPt = [self worldPointFromEvent:event];
        _isDrawing  = YES;
        break;

    case NSEventTypeLeftMouseDragged: {
        if (!_isDrawing) break;
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
    static const char *names[] = { "Open Field", "Barrier", "Pillars" };
    [_window setTitle:[NSString stringWithFormat:
        @"CrowdSim — %s  ·  Draw: drag mouse  ·  C: clear drawing",
        names[idx]]];
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

    [_window setTitle:@"CrowdSim  ·  1/2/3: scene  ·  Drag: draw obstacle  ·  C: clear"];
    [_window center];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"[AppDelegate] Metal is not supported on this device.");
        [NSApp terminate:nil];
        return;
    }

    _view     = [[MTKView alloc] initWithFrame:frame device:device];
    _renderer = [[Renderer alloc] initWithView:_view];

    NSURL *libURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    NSError *err  = nil;
    id<MTLLibrary> library = libURL ? [device newLibraryWithURL:libURL error:&err] : nil;
    if (err) NSLog(@"[AppDelegate] Library error: %@", err);

    uint32_t n = _agentCount > 0 ? _agentCount : kDefaultAgentCount;
    _gpuSim = [[GPUSimulation alloc] initWithDevice:device
                                            library:library
                                         agentCount:n];

    [_renderer setGPUSimulation:_gpuSim];
    _view.delegate = _renderer;

    [_window setContentView:_view];
    [_window makeKeyAndOrderFront:nil];

    // ── Event monitors ────────────────────────────────────────────────────────
    __weak AppDelegate *weakSelf = self;

    // Keyboard: 1/2/3 switch scene, C clears user-drawn obstacles
    [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                          handler:^NSEvent *(NSEvent *event) {
        NSString *ch = event.charactersIgnoringModifiers;
        if      ([ch isEqualToString:@"1"]) [weakSelf loadScenario:0];
        else if ([ch isEqualToString:@"2"]) [weakSelf loadScenario:1];
        else if ([ch isEqualToString:@"3"]) [weakSelf loadScenario:2];
        else if ([ch isEqualToString:@"c"] ||
                 [ch isEqualToString:@"C"]) { AppDelegate *s = weakSelf; [s loadScenario:s->_currentScenario]; }
        return event;
    }];

    // Mouse: left-drag draws obstacle segments directly onto the simulation
    NSEventMask drawMask = NSEventMaskLeftMouseDown
                         | NSEventMaskLeftMouseDragged
                         | NSEventMaskLeftMouseUp;
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
