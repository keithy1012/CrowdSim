#pragma once
#import <AppKit/AppKit.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
// Set before [app run]. 0 = use compiled-in default (100 000).
@property (nonatomic) uint32_t agentCount;
@end
