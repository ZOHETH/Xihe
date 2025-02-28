#version 450

precision highp float;

#extension GL_EXT_nonuniform_qualifier : require


layout (location = 0) in vec4 in_pos;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_normal;

layout (location = 0) out vec4 o_albedo;
layout (location = 1) out vec4 o_normal;

layout(set = 0, binding = 1) uniform GlobalUniform {
    mat4 model;
    mat4 view_proj;
    vec3 camera_position;
} global_uniform;

layout (set = 1, binding = 10 ) uniform sampler2D global_textures[];

layout(push_constant, std430) uniform PBRMaterialUniform {
    // x = diffuse index, y = roughness index, z = normal index, w = occlusion index.
	// Occlusion and roughness are encoded in the same texture
	uvec4       texture_indices;
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
} pbr_material_uniform;

void main(void)
{
    vec3 normal = normalize(in_normal);
    // Transform normals from [-1, 1] to [0, 1]
    o_normal = vec4(0.5 * normal + 0.5, 1.0);

    vec4 base_color = vec4(1.0, 0.0, 0.0, 1.0);

#ifdef HAS_BASE_COLOR_TEXTURE
    base_color = texture(global_textures[nonuniformEXT(pbr_material_uniform.texture_indices.x)], in_uv);
#else
    base_color = pbr_material_uniform.base_color_factor;
#endif

    o_albedo = base_color;
}