#version 450

#define SHADOW_MAP_CASCADE_COUNT 3

precision highp float;

layout(binding = 0) uniform sampler2D texture_depth;
layout(binding = 1) uniform sampler2D texture_albedo;
layout(binding = 2) uniform sampler2D texture_normal;
layout(binding = 9) uniform sampler2D texture_emissive;

layout(location = 0) in vec2 vertex_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 3) uniform GlobalUniform {
    mat4 inverse_view_proj;
    vec3 camera_position;
} global_uniform;

layout(set = 0, binding = 11) uniform samplerCube sampler_irradiance;
layout(set = 0, binding = 12) uniform samplerCube sampler_prefiltered;
layout(set = 0, binding = 13) uniform sampler2D sampler_brdf_lut;

#include "pbr_lighting.h"

layout(set = 0, binding = 4) uniform LightsInfo {
    Light directional_lights[MAX_LIGHT_COUNT];
    Light point_lights[MAX_LIGHT_COUNT];
    Light spot_lights[MAX_LIGHT_COUNT];
} lights_info;

layout(set = 0, binding = 5) uniform ShadowUniform {
    vec4 cascade_far_depths;
    mat4 light_matrices[SHADOW_MAP_CASCADE_COUNT];
} shadow_uniform;

layout(set = 0, binding = 6) uniform sampler2DArrayShadow shadow_sampler;

layout(constant_id = 0) const uint DIRECTIONAL_LIGHT_COUNT = 0U;
layout(constant_id = 1) const uint POINT_LIGHT_COUNT = 0U;
layout(constant_id = 2) const uint SPOT_LIGHT_COUNT = 0U;

#extension GL_EXT_nonuniform_qualifier : require
layout(set = 1, binding = 10) uniform sampler2DShadow global_textures[];

float calculateShadow(highp vec3 position, uint cascade_index) {
    vec4 projected_coord = shadow_uniform.light_matrices[cascade_index] * vec4(position, 1.0);
    projected_coord /= projected_coord.w;
    projected_coord.xy = 0.5 * projected_coord.xy + 0.5;

    if (projected_coord.x < 0.0 || projected_coord.x > 1.0 ||
        projected_coord.y < 0.0 || projected_coord.y > 1.0) {
        return 1.0;
    }

    float shadow = 0.0;
    int sample_count = 0;

    const int KERNEL_SIZE = 5;
    const float SHADOW_MAP_RESOLUTION = 2048.0;
    const float TEXEL_SIZE = 1.0 / SHADOW_MAP_RESOLUTION;
    const float SAMPLE_OFFSET = TEXEL_SIZE;
    const float DEPTH_BIAS = 0.000005;

    for (int x = -KERNEL_SIZE / 2; x <= KERNEL_SIZE / 2; x++) {
        for (int y = -KERNEL_SIZE / 2; y <= KERNEL_SIZE / 2; y++) {
            vec2 texture_offset = vec2(float(x), float(y)) * SAMPLE_OFFSET;
            float shadow_sample = texture(
                shadow_sampler,
                vec4(projected_coord.xy + texture_offset, cascade_index, projected_coord.z - DEPTH_BIAS)
            );
            shadow += shadow_sample;
            sample_count++;
        }
    }

    shadow /= float(sample_count);
    return shadow;
}

vec3 calculateReinhardExtended(vec3 hdr_color, float white_point) {
    vec3 numerator = hdr_color * (1.0 + (hdr_color / (white_point * white_point)));
    return numerator / (1.0 + hdr_color);
}

void main() {
    // Retrieve position from depth
    vec2 screen_uv = gl_FragCoord.xy / vec2(textureSize(texture_depth, 0));
    float depth = texture(texture_depth, screen_uv).x;

#ifdef USE_IBL
    if (depth < 0.000000001) {
        vec4 clip = vec4(vertex_uv * 2.0 - 1.0, 0.0, 1.0);
        vec4 world_direction = global_uniform.inverse_view_proj * clip;
        vec3 view_direction = normalize(world_direction.xyz / world_direction.w);

        vec3 background = textureLod(sampler_prefiltered, view_direction, 1.0).rgb;
        frag_color = vec4(calculateReinhardExtended(background, 1.0), 1.0);
        return;
    }
#endif

    vec4 clip = vec4(vertex_uv * 2.0 - 1.0, texture(texture_depth, screen_uv).x, 1.0);
    highp vec4 world_position_w = global_uniform.inverse_view_proj * clip;
    highp vec3 world_position = world_position_w.xyz / world_position_w.w;

    vec4 albedo_roughness = texture(texture_albedo, screen_uv);
    vec3 albedo = albedo_roughness.rgb;

    // Transform normal from [0,1] to [-1,1] range
    vec4 normal_metallic = texture(texture_normal, screen_uv);
    vec3 normal = normalize(2.0 * normal_metallic.xyz - 1.0);

    // Calculate shadow cascade index
    uint cascade_index = 0;
    for (uint i = 0; i < SHADOW_MAP_CASCADE_COUNT; ++i) {
        if (texture(texture_depth, screen_uv).x < shadow_uniform.cascade_far_depths[i]) {
            cascade_index = i;
        }
    }

    // Calculate lighting
    vec3 total_light = vec3(0.0);
    float roughness = albedo_roughness.w;
    float metallic = normal_metallic.w;
    vec3 camera_position = global_uniform.camera_position.xyz;

    for (uint i = 0U; i < DIRECTIONAL_LIGHT_COUNT; ++i) {
        vec3 light_contribution = calculateDirectionalLight(
            lights_info.directional_lights[i],
            world_position,
            normal,
            albedo,
            metallic,
            roughness,
            camera_position
        );
        if (i == 0U) {
            light_contribution *= calculateShadow(world_position, cascade_index);
        }
        total_light += light_contribution;
    }

    for (uint i = 0U; i < POINT_LIGHT_COUNT; ++i) {
        total_light += calculatePointLight(
            lights_info.point_lights[i],
            world_position,
            normal,
            albedo,
            metallic,
            roughness,
            camera_position
        );
    }

    for (uint i = 0U; i < SPOT_LIGHT_COUNT; ++i) {
        total_light += calculateSpotLight(
            lights_info.spot_lights[i],
            world_position,
            normal,
            albedo,
            metallic,
            roughness,
            camera_position
        );
    }

    vec3 emissive = texture(texture_emissive, screen_uv).xyz;
    float luminance = dot(emissive, vec3(0.299, 0.587, 0.114));
    float emissive_gain = 1.0 + smoothstep(0.5, 1.0, luminance) * 150.0;
    emissive *= emissive_gain;

    vec3 final_color = total_light + emissive;
    vec3 ambient_color = vec3(0.2) * albedo.xyz;

#ifdef USE_IBL
    vec3 view_direction = normalize(camera_position - world_position);
    vec3 ibl = calculateIBL(normal, view_direction, albedo, metallic, roughness);
    final_color += ibl * 1.0;
#else
    final_color += ambient_color;
#endif

#ifdef SHOW_CASCADE_VIEW
    vec3 cascade_overlay = vec3(0.0);
    if (cascade_index == 0) {
        cascade_overlay = vec3(0.2, 0.3, 0.6);
    } else if (cascade_index == 1) {
        cascade_overlay = vec3(0.3, 0.6, 0.3);
    } else if (cascade_index == 2) {
        cascade_overlay = vec3(0.6, 0.4, 0.2);
    } else if (cascade_index == 3) {
        cascade_overlay = vec3(0.6, 0.3, 0.6);
    }
    final_color = mix(final_color, final_color + cascade_overlay, 0.3);
#endif

    final_color = calculateReinhardExtended(final_color, 1.0);
    frag_color = vec4(final_color, 1.0);
}