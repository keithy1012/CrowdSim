#import "Renderer.h"

@implementation Renderer {
    id<MTLDevice>       _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary>      _library;
}

- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;

    _device       = view.device;
    _commandQueue = [_device newCommandQueue];

    // Load pre-compiled shader library from the app bundle (requires full Xcode build)
    NSURL *libURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    if (libURL) {
        NSError *err = nil;
        _library = [_device newLibraryWithURL:libURL error:&err];
        if (err) NSLog(@"[Renderer] Failed to load metallib: %@", err);
    } else {
        NSLog(@"[Renderer] No metallib found — shaders not compiled yet.");
    }

    view.clearColor              = MTLClearColorMake(0.05, 0.05, 0.10, 1.0);
    view.preferredFramesPerSecond = 60;

    return self;
}

- (void)drawInMTKView:(MTKView *)view {
    id<MTLCommandBuffer>        cmd = [_commandQueue commandBuffer];
    MTLRenderPassDescriptor    *rpd = view.currentRenderPassDescriptor;
    if (!rpd || !view.currentDrawable) return;

    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
    // Phase 2: add render commands here
    [enc endEncoding];

    [cmd presentDrawable:view.currentDrawable];
    [cmd commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}

@end
