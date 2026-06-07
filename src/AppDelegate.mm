#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"

@implementation AppDelegate {
    NSWindow *_window;
    MTKView  *_view;
    Renderer *_renderer;
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

    _view          = [[MTKView alloc] initWithFrame:frame device:device];
    _renderer      = [[Renderer alloc] initWithView:_view];
    _view.delegate = _renderer;

    [_window setContentView:_view];
    [_window makeKeyAndOrderFront:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app {
    return YES;
}

@end
