const float PI = 3.14159265359;

struct Light
{
	vec4 position;           // position.w represents type of light
	vec4 color;              // color.w represents light intensity
	vec4 direction;          // direction.w represents range
	vec2 cone_angles;        // x: light inner cone angle, y: light outer cone angle
};

float calculateGGXDistribution(float n_dot_h, float roughness)
{
	float alpha           = roughness * roughness;
	float alpha_squared   = alpha * alpha;
	float n_dot_h_squared = n_dot_h * n_dot_h;
	float denominator     = (n_dot_h_squared * (alpha_squared - 1.0) + 1.0);
	return alpha_squared / (PI * denominator * denominator);
}

float calculateSmithGeometry(float n_dot_v, float n_dot_l, float roughness)
{
	float r   = roughness + 1.0;
	float k   = (r * r) / 8.0;
	float g1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
	float g1l = n_dot_l / (n_dot_l * (1.0 - k) + k);
	return g1v * g1l;
}

vec3 calculateSchlickFresnel(float cos_theta, vec3 f0)
{
	return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

vec3 calculatePBR(vec3 light_dir, vec3 view_dir, vec3 normal, vec3 light_color,
                  vec3 albedo, float metallic, float roughness)
{
	vec3 half_vector = normalize(view_dir + light_dir);

	float n_dot_v = max(dot(normal, view_dir), 0.001);
	float n_dot_l = max(dot(normal, light_dir), 0.001);
	float n_dot_h = max(dot(normal, half_vector), 0.0);
	float l_dot_h = max(dot(light_dir, half_vector), 0.0);

	vec3 f0 = mix(vec3(0.04), albedo, metallic);

	// Cook-Torrance BRDF
	float distribution = calculateGGXDistribution(n_dot_h, roughness);
	float geometry     = calculateSmithGeometry(n_dot_v, n_dot_l, roughness);
	vec3  fresnel      = calculateSchlickFresnel(l_dot_h, f0);

	vec3 specular = (distribution * geometry * fresnel) /
	                max(4.0 * n_dot_v * n_dot_l, 0.001);
	vec3 k_diffuse = (vec3(1.0) - fresnel) * (1.0 - metallic);
	vec3 diffuse   = k_diffuse * albedo / PI;

	return (diffuse + specular) * light_color * n_dot_l;
}

vec3 calculateDirectionalLight(Light light, vec3 position, vec3 normal,
                               vec3 albedo, float metallic, float roughness,
                               vec3 camera_pos)
{
	vec3 light_dir       = normalize(-light.direction.xyz);
	vec3 view_dir        = normalize(camera_pos - position);
	vec3 light_intensity = light.color.rgb * light.color.w;

	return calculatePBR(light_dir, view_dir, normal, light_intensity,
	                    albedo, metallic, roughness);
}

vec3 calculatePointLight(Light light, vec3 position, vec3 normal,
                         vec3 albedo, float metallic, float roughness,
                         vec3 camera_pos)
{
	vec3  light_to_frag = light.position.xyz - position;
	float distance      = length(light_to_frag);
	vec3  light_dir     = normalize(light_to_frag);

	float attenuation     = 1.0 / (distance * distance * 0.01);
	vec3  light_intensity = light.color.rgb * light.color.w * attenuation;
	vec3  view_dir        = normalize(camera_pos - position);

	return calculatePBR(light_dir, view_dir, normal, light_intensity,
	                    albedo, metallic, roughness);
}

vec3 calculateSpotLight(Light light, vec3 position, vec3 normal,
                        vec3 albedo, float metallic, float roughness,
                        vec3 camera_pos)
{
	vec3  light_to_frag = light.position.xyz - position;
	float distance      = length(light_to_frag);
	vec3  light_dir     = normalize(light_to_frag);

	vec3  light_to_pixel = -light_dir;
	float theta          = dot(light_to_pixel, normalize(light.direction.xyz));
	float intensity      = smoothstep(light.cone_angles.y, light.cone_angles.x,
	                                  acos(theta));

	float attenuation     = 1.0 / (distance * distance * 0.01);
	vec3  light_intensity = light.color.rgb * light.color.w * intensity * attenuation;
	vec3  view_dir        = normalize(camera_pos - position);

	return calculatePBR(light_dir, view_dir, normal, light_intensity,
	                    albedo, metallic, roughness);
}

vec3 calculateReflectionVector(vec3 normal, vec3 view_dir)
{
	return reflect(-view_dir, normal);
}

float calculateLODLevel(float roughness, float num_mip_levels)
{
	return roughness * (num_mip_levels - 1);
}

vec3 calculateSchlickFresnelRoughness(float cos_theta, vec3 f0, float roughness)
{
	float r1 = 1.0 - roughness;
	return f0 + (max(vec3(r1), f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

vec3 calculateUncharted2Tonemap(vec3 color)
{
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;
	const float W = 11.2;
	return ((color * (A * color + C * B) + D * E) /
	        (color * (A * color + B) + D * F)) -
	       E / F;
}

vec3 tonemap(vec3 color)
{
	vec3 mapped_color = calculateUncharted2Tonemap(color * 4.5);
	mapped_color      = mapped_color * (1.0f / calculateUncharted2Tonemap(vec3(11.2f)));
	return mapped_color;
}

vec3 calculateIBL(vec3 normal, vec3 view_dir, vec3 albedo,
                  float metallic, float roughness)
{
	vec3  f0      = mix(vec3(0.04), albedo, metallic);
	float n_dot_v = max(dot(normal, view_dir), 0.001);

	vec3 reflection = reflect(-view_dir, normal);

	vec3 irradiance = tonemap(texture(sampler_irradiance, normal).rgb);
	vec3 diffuse    = irradiance * albedo;

	float lod               = roughness * 10;
	vec3  prefiltered_color = tonemap(
        textureLod(sampler_prefiltered, reflection, lod).rgb);

	vec2 brdf = texture(sampler_brdf_lut,
	                    vec2(n_dot_v, 1.0 - roughness))
	                .rg;

	vec3 fresnel = calculateSchlickFresnelRoughness(n_dot_v, f0, roughness);

	vec3 k_specular = fresnel;
	vec3 k_diffuse  = (1.0 - k_specular) * (1.0 - metallic);

	vec3 specular = prefiltered_color * (f0 * brdf.x + brdf.y);

	return (k_diffuse * diffuse + specular);
}