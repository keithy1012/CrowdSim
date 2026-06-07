#include <metal_stdlib>
using namespace metal;

// ── Phase 1: instanced circle rendering ─────────────────────────────────────

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Each instance is one agent.
// Buffer 0: packed float2 positions [x0,y0, x1,y1, ...]
// Buffer 1: float radii
// Buffer 2: float2 viewport (pixels)
vertex VertexOut vs_agent(
    uint               vid  [[vertex_id]],
    uint               iid  [[instance_id]],
    constant float2   *pos  [[buffer(0)]],
    constant float    *rad  [[buffer(1)]],
    constant float2   &vp   [[buffer(2)]]
) {
    // Unit quad as triangle strip (vid 0-3)
    float2 offsets[4] = {
        float2(-1.f, -1.f),
        float2(-1.f,  1.f),
        float2( 1.f, -1.f),
        float2( 1.f,  1.f)
    };

    float2 offset   = offsets[vid];
    float2 worldPos = pos[iid] + offset * rad[iid];

    // World [0, vp] → NDC [-1, 1]; flip Y (world Y is down, NDC Y is up)
    float2 ndc = (worldPos / vp) * 2.f - 1.f;
    ndc.y      = -ndc.y;

    VertexOut out;
    out.position = float4(ndc, 0.f, 1.f);
    out.uv       = offset;
    return out;
}

fragment float4 fs_agent(VertexOut in [[stage_in]]) {
    // Discard pixels outside the unit circle
    if (length(in.uv) > 1.f) discard_fragment();
    return float4(1.f, 1.f, 1.f, 1.f);
}

// ── Phase 2: GPU compute passes will be added here ───────────────────────────
// Pass 1 — Build spatial grid
// Pass 2 — Neighbor search
// Pass 3 — Steering update
// Pass 4 — Physics integration
