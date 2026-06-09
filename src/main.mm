#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import "AppDelegate.h"
#import "Benchmark.h"
#include <cstring>

int main(int argc, const char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            @autoreleasepool {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                if (!device) {
                    fprintf(stderr, "Metal not supported.\n");
                    return 1;
                }
                // The binary is at CrowdSim.app/Contents/MacOS/CrowdSim;
                // the metallib lives two levels up in Resources/.
                NSString *binPath = [NSString stringWithUTF8String:argv[0]];
                NSString *resDir  = [[[binPath stringByDeletingLastPathComponent]
                                       stringByDeletingLastPathComponent]
                                      stringByAppendingPathComponent:@"Resources"];
                NSURL *libURL = [NSURL fileURLWithPath:
                    [resDir stringByAppendingPathComponent:@"default.metallib"]];
                NSError *err = nil;
                id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
                if (!lib) {
                    fprintf(stderr, "Could not load Metal library: %s\n",
                            err.localizedDescription.UTF8String);
                    return 1;
                }
                return runBenchmark(device, lib);
            }
        }
    }

    @autoreleasepool {
        NSApplication *app      = [NSApplication sharedApplication];
        AppDelegate   *delegate = [[AppDelegate alloc] init];

        for (int i = 1; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--agents") == 0) {
                int n = atoi(argv[i + 1]);
                if (n > 0) delegate.agentCount = (uint32_t)n;
                break;
            }
        }

        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}
