#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"
#import "GPUSimulation.h"

// Phase 3 — spatial hash grid target
static constexpr uint32_t kAgentCount = 100000;

@implementation AppDelegate {
    NSWindow      *_window;
    MTKView       *_view;
    Renderer      *_renderer;
    GPUSimulation *_gpuSim;
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

    _view     = [[MTKView alloc] initWithFrame:frame device:device];
    _renderer = [[Renderer alloc] initWithView:_view];

    // Library is loaded by Renderer; we need it here to build GPU simulation pipelines.
    // Load from the same bundle location so both share the compiled shaders.
    NSURL *libURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    NSError *err  = nil;
    id<MTLLibrary> library = libURL ? [device newLibraryWithURL:libURL error:&err] : nil;
    if (err) NSLog(@"[AppDelegate] Library error: %@", err);

    _gpuSim = [[GPUSimulation alloc] initWithDevice:device
                                            library:library
                                         agentCount:kAgentCount];

    [_renderer setGPUSimulation:_gpuSim];
    _view.delegate = _renderer;

    [_window setContentView:_view];
    [_window makeKeyAndOrderFront:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app {
    return YES;
}

@end
