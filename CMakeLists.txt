cmake_minimum_required(VERSION 3.20)
project(CrowdSim LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_library(METAL_LIB      Metal       REQUIRED)
find_library(METALKIT_LIB   MetalKit    REQUIRED)
find_library(APPKIT_LIB     AppKit      REQUIRED)
find_library(FOUNDATION_LIB Foundation  REQUIRED)
find_library(QUARTZCORE_LIB QuartzCore  REQUIRED)

set(SOURCES
    src/main.mm
    src/AppDelegate.mm
    src/Renderer.mm
    src/Simulation.cpp
    src/GPUSimulation.mm
)

add_executable(CrowdSim MACOSX_BUNDLE ${SOURCES})

target_include_directories(CrowdSim PRIVATE
    src
    # Required when building against Command Line Tools SDK without full Xcode
    ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1
)

target_compile_options(CrowdSim PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

target_link_libraries(CrowdSim
    ${METAL_LIB}
    ${METALKIT_LIB}
    ${APPKIT_LIB}
    ${FOUNDATION_LIB}
    ${QUARTZCORE_LIB}
)

set_target_properties(CrowdSim PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER  com.keithyao.crowdsim
    MACOSX_BUNDLE_BUNDLE_NAME     CrowdSim
    MACOSX_BUNDLE_BUNDLE_VERSION  "0.1"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
)

# --- Metal shader compilation ---
# Requires full Xcode (not just Command Line Tools).
# Install via: https://apps.apple.com/app/xcode/id497799835
execute_process(
    COMMAND xcrun --sdk macosx --find metal
    OUTPUT_VARIABLE METAL_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE METAL_FOUND
)

if(METAL_FOUND EQUAL 0)
    set(METAL_SRC  ${CMAKE_SOURCE_DIR}/shaders/Shaders.metal)
    set(AIR_FILE   ${CMAKE_BINARY_DIR}/Shaders.air)
    set(METALLIB   ${CMAKE_BINARY_DIR}/default.metallib)

    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC}
                -I ${CMAKE_SOURCE_DIR}/src
                -o ${AIR_FILE}
        DEPENDS ${METAL_SRC} ${CMAKE_SOURCE_DIR}/src/SharedTypes.h
        COMMENT "Compiling Metal shaders"
    )

    add_custom_command(
        OUTPUT ${METALLIB}
        COMMAND xcrun -sdk macosx metallib ${AIR_FILE} -o ${METALLIB}
        DEPENDS ${AIR_FILE}
        COMMENT "Linking Metal library"
    )

    add_custom_target(MetalShaders ALL DEPENDS ${METALLIB})
    add_dependencies(CrowdSim MetalShaders)

    add_custom_command(TARGET CrowdSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${METALLIB}
            $<TARGET_BUNDLE_CONTENT_DIR:CrowdSim>/Resources/default.metallib
        COMMENT "Copying metallib into app bundle"
    )
else()
    message(WARNING
        "Metal compiler not found — shader compilation skipped.\n"
        "Install Xcode from the App Store to enable Metal shaders."
    )
endif()
