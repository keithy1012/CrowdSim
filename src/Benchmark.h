#pragma once
#import <Metal/Metal.h>

int runBenchmark(id<MTLDevice> device, id<MTLLibrary> library);
int runTilingBenchmark(id<MTLDevice> device, id<MTLLibrary> library);
