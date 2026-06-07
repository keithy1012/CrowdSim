#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"
#include "Simulation.h"

// Increase to 10000 to benchmark Phase 1 target; lower values are smoother in Debug builds
static constexpr uint32_t kAgentCount = 1000;

@implementation AppDelegate {
    NSWindow   *_window;
    MTKView    *_view;
    Renderer   *_renderer;
    Simulation *_sim;
}

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

    [_window setTitle:@"CrowdSim"];
    [_window center];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"[AppDelegate] Metal is not supported on this device.");
        [NSApp terminate:nil];
        return;
    }

    _sim = new Simulation(kAgentCount);

    _view          = [[MTKView alloc] initWithFrame:frame device:device];
    _renderer      = [[Renderer alloc] initWithView:_view];
    [_renderer setSimulation:_sim];
    _view.delegate = _renderer;

    [_window setContentView:_view];
    [_window makeKeyAndOrderFront:nil];
}

- (void)dealloc {
    delete _sim;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app {
    return YES;
}

@end
