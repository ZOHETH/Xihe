#include "sample_app.h"

#include "backend/shader_compiler/glsl_compiler.h"
#include "rendering/passes/bloom_pass.h"
#include "rendering/passes/cascade_shadow_pass.h"
#include "rendering/passes/clustered_lighting_pass.h"
#include "rendering/passes/geometry_pass.h"
#include "rendering/passes/mesh_draw_preparation.h"
#include "rendering/passes/mesh_pass.h"
#include "rendering/passes/meshlet_pass.h"
#include "rendering/passes/pointshadows_pass.h"
#include "rendering/passes/test_pass.h"
#include "scene_graph/components/camera.h"
#include "scene_graph/components/light.h"
#include "scene_graph/components/mesh.h"

#define EX

namespace xihe
{
using namespace rendering;

SampleApp::SampleApp()
{
	add_device_extension(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
	add_device_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	add_device_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
	add_device_extension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);

	backend::GlslCompiler::set_target_environment(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);
}

bool SampleApp::prepare(Window *window)
{
	if (!XiheApp::prepare(window))
	{
		return false;
	}

	asset_loader_ = std::make_unique<AssetLoader>(*device_);

	load_scene("scenes/sponza/Sponza01.gltf");
	// load_scene("scenes/cube.gltf");
	assert(scene_ && "Scene not loaded");
	update_bindless_descriptor_sets();
	gpu_scene_ = std::make_unique<GpuScene>(*device_);
	gpu_scene_->initialize(*scene_);

	// auto *skybox_texture = asset_loader_->load_texture_cube(*scene_, "skybox", "textures/uffizi_rgba16f_cube.ktx");

	auto light_pos   = glm::vec3(-150.0f, 188.0f, -225.0f);
	auto light_color = glm::vec3(1.0, 1.0, 1.0);

	// Magic numbers used to offset lights in the Sponza scene
	/*for (int i = -4; i < 4; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			glm::vec3 pos = light_pos;
			pos.x += i * 400;
			pos.z += j * 150;
			pos.y = 8;
			for (int k = 0; k < 6; ++k)
			{
				pos.y         = pos.y + (k * 50);
				light_color.x = static_cast<float>(rand()) / (RAND_MAX);
				light_color.y = static_cast<float>(rand()) / (RAND_MAX);
				light_color.z = static_cast<float>(rand()) / (RAND_MAX);
				sg::LightProperties props;
				props.color     = light_color;
				props.intensity = 2.0f;
				props.range     = 700.f;
				add_point_light(*scene_, pos, props);
			}
		}
	}*/

	for (int i = -4; i < 4; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			glm::vec3 pos = light_pos;
			pos.x += i * 400;
			pos.z += j * (225 + 140);
			pos.y = 8;

			for (int k = 0; k < 3; ++k)
			{
				pos.y = pos.y + (k * 100);

				light_color.x = static_cast<float>(rand()) / (RAND_MAX);
				light_color.y = static_cast<float>(rand()) / (RAND_MAX);
				light_color.z = static_cast<float>(rand()) / (RAND_MAX);

				sg::LightProperties props;
				props.color     = light_color;
				props.intensity = 1.f;
				props.range     = 700.f;
				add_point_light(*scene_, pos, props);
			}
		}
	}

	auto &camera_node = sg::add_free_camera(*scene_, "main_camera", render_context_->get_surface_extent());
	auto  camera      = &camera_node.get_component<sg::Camera>();
	camera_           = camera;

	auto  cascade_script   = std::make_unique<sg::CascadeScript>("", *scene_, *dynamic_cast<sg::PerspectiveCamera *>(camera));
	auto *p_cascade_script = cascade_script.get();
	scene_->add_component(std::move(cascade_script));

	// shadow pass
	{
		PassAttachment shadow_attachment_0{AttachmentType::kDepth, "shadowmap"};
		shadow_attachment_0.extent_desc                    = ExtentDescriptor::Fixed({kShadowmapResolution, kShadowmapResolution, 1});
		shadow_attachment_0.image_properties.array_layers  = 3;
		shadow_attachment_0.image_properties.current_layer = 0;
		shadow_attachment_0.image_properties.n_use_layer   = 1;

		PassAttachment shadow_attachment_1                 = shadow_attachment_0;
		shadow_attachment_1.image_properties.current_layer = 1;

		PassAttachment shadow_attachment_2                 = shadow_attachment_0;
		shadow_attachment_2.image_properties.current_layer = 2;

		auto shadow_pass_0 = std::make_unique<CascadeShadowPass>(scene_->get_components<sg::Mesh>(), *p_cascade_script, 0);
		graph_builder_->add_pass("Shadow 0", std::move(shadow_pass_0))
		    .attachments({{shadow_attachment_0}})
		    .shader({"shadow/csm.vert", "shadow/csm.frag"})
		    .finalize();

		auto shadow_pass_1 = std::make_unique<CascadeShadowPass>(scene_->get_components<sg::Mesh>(), *p_cascade_script, 1);
		graph_builder_->add_pass("Shadow 1", std::move(shadow_pass_1))
		    .attachments({{shadow_attachment_1}})
		    .shader({"shadow/csm.vert", "shadow/csm.frag"})
		    .finalize();

		auto shadow_pass_2 = std::make_unique<CascadeShadowPass>(scene_->get_components<sg::Mesh>(), *p_cascade_script, 2);
		graph_builder_->add_pass("Shadow 2", std::move(shadow_pass_2))
		    .attachments({{shadow_attachment_2}})
		    .shader({"shadow/csm.vert", "shadow/csm.frag"})
		    .finalize();

		/*auto test_pass = std::make_unique<TestPass>();
		graph_builder_->add_pass("Test", std::move(test_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "per-light meshlet indies", .buffer_size = 256 * 4}})
		    .shader({"shadow/test.comp"})
		    .finalize();*/

		auto point_shadows_culling_pass = std::make_unique<PointShadowsCullingPass>(*gpu_scene_, scene_->get_components<sg::Light>());
		graph_builder_->add_pass("Point Light Shadows Culling", std::move(point_shadows_culling_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "meshlet instances", .buffer_size = kMaxPointLightCount * kMaxPerLightMeshletCount * 8},
		                {.type = BindableType::kStorageBufferWriteClear, .name = "per-light meshlet indies", .buffer_size = (kMaxPointLightCount + 1) * 2 * 4}})
		    .shader({"shadow/pointshadows_culling.comp"})
		    .finalize();

		auto point_shadows_commands_generation_pass = std::make_unique<PointShadowsCommandsGenerationPass>();
		graph_builder_->add_pass("Point Light Shadows Commands Generation", std::move(point_shadows_commands_generation_pass))
		    .bindables({{.type = BindableType::kStorageBufferRead, .name = "per-light meshlet indies"},
		                {.type = BindableType::kStorageBufferWrite, .name = "meshlet draw command", .buffer_size = kMaxPointLightCount * 6 * 16}})
		    .shader({"shadow/pointshadows_commands_generation.comp"})
		    .finalize();

		PassAttachment point_shadows_attachment{AttachmentType::kDepth, "point shadowmaps"};
		point_shadows_attachment.extent_desc                    = ExtentDescriptor::Fixed({1024, 1024, 1});
		point_shadows_attachment.image_properties.array_layers  = PointShadowsResources::get().get_point_light_count() * 6;
		point_shadows_attachment.image_properties.current_layer = 0;
		point_shadows_attachment.image_properties.n_use_layer   = PointShadowsResources::get().get_point_light_count() * 6;

		auto point_shadows_pass = std::make_unique<PointShadowsPass>(*gpu_scene_, scene_->get_components<sg::Light>());
		graph_builder_->add_pass("Point Light Shadows", std::move(point_shadows_pass))
		    .bindables({
		        {.type = BindableType::kStorageBufferRead, .name = "meshlet instances"},
		        {.type = BindableType::kStorageBufferRead, .name = "per-light meshlet indies"},
		        {.type = BindableType::kIndirectBuffer, .name = "meshlet draw command"},
		    })
		    .attachments({point_shadows_attachment})
		    .shader({"shadow/pointshadows.task", "shadow/pointshadows.mesh"})
		    .finalize();
		;
	}

	// geometry pass
	{
		/*auto geometry_pass = std::make_unique<GeometryPass>(scene_->get_components<sg::Mesh>(), *camera);

		graph_builder_->add_pass("Geometry", std::move(geometry_pass))

		    .attachments({{AttachmentType::kDepth, "depth"},
		                  {AttachmentType::kColor, "albedo"},
		                  {AttachmentType::kColor, "normal", vk::Format::eA2B10G10R10UnormPack32}})

		    .shader({"deferred/geometry.vert", "deferred/geometry.frag"})

		    .finalize();*/
#ifdef EX
		auto mesh_preparation_pass = std::make_unique<MeshDrawPreparationPass>(*gpu_scene_);
		graph_builder_->add_pass("Mesh Draw Preparation", std::move(mesh_preparation_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "draw command", .buffer_size = gpu_scene_->get_instance_count() * sizeof(MeshDrawCommand)}})
		    .shader({"mesh_shading/prepare_mesh_draws.comp"})
		    .finalize();
#endif

#ifdef EX
		auto geometry_pass = std::make_unique<MeshPass>(*gpu_scene_, *camera);
#else
		auto geometry_pass = std::make_unique<MeshletPass>(scene_->get_components<sg::Mesh>(), *camera);
#endif
		graph_builder_->add_pass("Geometry", std::move(geometry_pass))
#ifdef EX
		    .bindables({{.type = BindableType::kStorageBufferRead, .name = "draw command"}})
#endif

		    .attachments({{AttachmentType::kDepth, "depth"},
		                  {AttachmentType::kColor, "albedo"},
		                  {AttachmentType::kColor, "normal", vk::Format::eA2B10G10R10UnormPack32}})
#ifdef EX
		    .shader({"deferred/geometry_indirect.task", "deferred/geometry_indirect.mesh", "deferred/geometry_indirect.frag"})
#else
		    .shader({"deferred/geometry_mesh.task", "deferred/geometry_mesh.mesh", "deferred/geometry_mesh.frag"})
#endif
		    .finalize();
	}

	// lighting pass
	{
		auto lighting_pass = std::make_unique<ClusteredLightingPass>(scene_->get_components<sg::Light>(), *camera, p_cascade_script);

		graph_builder_->add_pass("Lighting", std::move(lighting_pass))

		    .bindables({{BindableType::kSampled, "depth"},
		                {BindableType::kSampled, "albedo"},
		                {BindableType::kSampled, "normal"},
		                {BindableType::kSampled, "shadowmap"},
		                {BindableType::kSampledCube, "point shadowmaps"}})

		    .attachments({{AttachmentType::kColor, "lighting", vk::Format::eR16G16B16A16Sfloat}})

		    .shader({"deferred/lighting.vert", "deferred/clustered_lighting.frag"})

		    .finalize();
	}

	// bloom pass
	{
		auto extract_pass = std::make_unique<BloomExtractPass>();

		graph_builder_->add_pass("Bloom Extract", std::move(extract_pass))
		    .bindables({{BindableType::kSampled, "lighting"},
		                {BindableType::kStorageWrite, "bloom_extract", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.5, 0.5)}})
		    .shader({"post_processing/bloom/threshold.comp"})
		    .finalize();

		auto downsample_pass0 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 0", std::move(downsample_pass0))
		    .bindables({{BindableType::kSampled, "bloom_extract"},
		                {BindableType::kStorageWrite, "bloom_down_sample_0", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.25, 0.25)}})
		    .shader({"post_processing/bloom/blur_down_first.comp"})
		    .finalize();

		auto downsample_pass1 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 1", std::move(downsample_pass1))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_0"},
		                {BindableType::kStorageWrite, "bloom_down_sample_1", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.125, 0.125)}})
		    .shader({"post_processing/bloom/blur_down.comp"})
		    .finalize();

		auto downsample_pass2 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 2", std::move(downsample_pass2))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_1"},
		                {BindableType::kStorageWrite, "bloom_down_sample_2", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.0625, 0.0625)}})
		    .shader({"post_processing/bloom/blur_down.comp"})
		    .finalize();

		auto upsample_pass0 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 0", std::move(upsample_pass0))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_2"},
		                {BindableType::kStorageWrite, "bloom_up_sample_0", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.125, 0.125)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();

		auto upsample_pass1 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 1", std::move(upsample_pass1))
		    .bindables({{BindableType::kSampled, "bloom_up_sample_0"},
		                {BindableType::kStorageWrite, "bloom_up_sample_1", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.25, 0.25)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();

		auto upsample_pass2 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 2", std::move(upsample_pass2))
		    .bindables({{BindableType::kSampled, "bloom_up_sample_1"},
		                {BindableType::kStorageWrite, "bloom_up_sample_2", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.5, 0.5)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();
	}

	gui_ = std::make_unique<Gui>(*this, *window, stats_.get());
	// composite pass
	{
		auto composite_pass = std::make_unique<BloomCompositePass>();
		graph_builder_->add_pass("Bloom Composite", std::move(composite_pass))
		    .bindables({{BindableType::kSampled, "lighting"},
		                {BindableType::kSampled, "bloom_up_sample_2"}})
		    .shader({"post_processing/bloom_composite.vert", "post_processing/bloom_composite.frag"})
		    .gui(gui_.get())
		    .present()
		    .finalize();
	}
	{
	}

	graph_builder_->build();

	return true;
}

void SampleApp::update(float delta_time)
{
	/*MeshletPass::show_meshlet_view(show_meshlet_view_, *scene_);
	MeshletPass::freeze_frustum(freeze_frustum_, camera_);*/
	MeshPass::show_meshlet_view(show_meshlet_view_);
	MeshPass::freeze_frustum(freeze_frustum_, camera_);
	LightingPass::show_cascade_view(show_cascade_view_);
	XiheApp::update(delta_time);
}

void SampleApp::request_gpu_features(backend::PhysicalDevice &gpu)
{
	XiheApp::request_gpu_features(gpu);

	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShader);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShaderQueries);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, taskShader);

	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan11Features, shaderDrawParameters);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceFragmentShadingRateFeaturesKHR, primitiveFragmentShadingRate);
	// REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceFragmentShadingRateFeaturesKHR, attachmentFragmentShadingRate);
}

void SampleApp::draw_gui()
{
	gui_->show_stats(*stats_);

	gui_->show_views_window(
	    /* body = */ [this]() {
		    ImGui::Checkbox("Meshlet", &show_meshlet_view_);
		    ImGui::Checkbox("视域静留", &freeze_frustum_);
		    ImGui::Checkbox("级联阴影", &show_cascade_view_);
	    },
	    /* lines = */ 2);
}
}        // namespace xihe

std::unique_ptr<xihe::Application> create_application()
{
	return std::make_unique<xihe::SampleApp>();
}