#include "preprocess_app.h"

#include "ktx.h"
#include "ktxvulkan.h"
#include "rendering/passes/bloom_pass.h"
#include "rendering/passes/cascade_shadow_pass.h"
#include "rendering/passes/geometry_pass.h"
#include "rendering/passes/lighting_pass.h"
#include "rendering/passes/skybox_pass.h"
#include "scene_graph/gltf_loader.h"
#include "scene_graph/components/image.h"

namespace xihe
{

uint32_t get_format_size(vk::Format format)
{
	switch (format)
	{
		case vk::Format::eR32G32B32A32Sfloat:
			return 16;
		case vk::Format::eR16G16B16A16Sfloat:
			return 8;
		default:
			throw std::runtime_error("Unsupported format");
	}
}

void download_image(backend::ImageView &image_view, backend::Device &device)
{
	vk::DeviceSize total_size = 0;

	auto                             subresource_range = image_view.get_subresource_range();
	std::vector<vk::BufferImageCopy> buffer_copy_regions(subresource_range.levelCount);

	for (uint32_t mip = 0; mip < subresource_range.levelCount; mip++)
	{
		auto &copy_region       = buffer_copy_regions[mip];
		auto  extent            = image_view.get_image().get_extent();
		copy_region.imageExtent = vk::Extent3D{std::max(extent.width >> mip, 1u), std::max(extent.height >> mip, 1u), extent.depth};

		copy_region.bufferRowLength                 = copy_region.imageExtent.width;
		copy_region.bufferImageHeight               = copy_region.imageExtent.height;
		copy_region.bufferOffset                    = total_size;
		copy_region.imageSubresource.aspectMask     = subresource_range.aspectMask;
		copy_region.imageSubresource.mipLevel       = mip;
		copy_region.imageSubresource.baseArrayLayer = subresource_range.baseArrayLayer;
		copy_region.imageSubresource.layerCount     = subresource_range.layerCount;

		total_size += get_format_size(image_view.get_format()) * copy_region.imageExtent.width * copy_region.imageExtent.height * copy_region.imageExtent.depth * subresource_range.layerCount;
	}

	backend::BufferBuilder builder{total_size};
	builder.with_usage(vk::BufferUsageFlagBits::eTransferDst);
	builder.with_vma_usage(VMA_MEMORY_USAGE_GPU_TO_CPU);
	builder.with_vma_flags(VMA_ALLOCATION_CREATE_MAPPED_BIT);
	auto dst_buffer = builder.build(device);

	backend::CommandBuffer &command_buffer = device.request_command_buffer();

	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	{
		common::BufferMemoryBarrier memory_barrier{};
		memory_barrier.src_access_mask = vk::AccessFlagBits2::eNone;
		memory_barrier.dst_access_mask = vk::AccessFlagBits2::eTransferWrite;
		memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eTransfer;
		memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eTransfer;
		command_buffer.buffer_memory_barrier(dst_buffer, 0, total_size, memory_barrier);
	}

	{
		common::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = vk::ImageLayout::eUndefined;
		memory_barrier.new_layout      = vk::ImageLayout::eTransferSrcOptimal;
		memory_barrier.dst_access_mask = vk::AccessFlagBits2::eTransferRead;
		memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;
		memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eTransfer;
		command_buffer.image_memory_barrier(image_view, memory_barrier);
	}

	command_buffer.copy_image_to_buffer(image_view.get_image(), vk::ImageLayout::eTransferSrcOptimal, dst_buffer, buffer_copy_regions);

	{
		common::BufferMemoryBarrier memory_barrier{};
		memory_barrier.src_access_mask = vk::AccessFlagBits2::eTransferWrite;
		memory_barrier.dst_access_mask = vk::AccessFlagBits2::eHostRead;
		memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eTransfer;
		memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eHost;
		command_buffer.buffer_memory_barrier(dst_buffer, 0, total_size, memory_barrier);
	}

	{
		common::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout     = vk::ImageLayout::eTransferSrcOptimal;
		memory_barrier.new_layout     = vk::ImageLayout::eShaderReadOnlyOptimal;
		memory_barrier.src_stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
		memory_barrier.dst_stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
		command_buffer.image_memory_barrier(image_view, memory_barrier);
	}

	command_buffer.end();

	const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
	queue.submit(command_buffer, device.request_fence());

	device.wait_idle();

	auto raw_data = dst_buffer.map();

	ktxTexture2         *ktx_texture;
	ktxTextureCreateInfo create_info{};
	create_info.vkFormat        = static_cast<VkFormat>(image_view.get_format());
	create_info.baseWidth       = image_view.get_image().get_extent().width;
	create_info.baseHeight      = image_view.get_image().get_extent().height;
	create_info.baseDepth       = image_view.get_image().get_extent().depth;
	create_info.numDimensions   = 2;
	create_info.numFaces        = image_view.get_subresource_range().layerCount;
	create_info.numLevels       = image_view.get_subresource_range().levelCount;
	create_info.numLayers       = 1;
	create_info.isArray         = KTX_FALSE;
	create_info.generateMipmaps = KTX_FALSE;

	auto error = ktxTexture2_Create(&create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &ktx_texture);

	// ktxTexture_construct(&ktx_texture, &create_info, nullptr);

	VkDeviceSize current_offset = 0;
	for (uint32_t level = 0; level < subresource_range.levelCount; level++)
	{
		uint32_t width  = std::max(1u, create_info.baseWidth >> level);
		uint32_t height = std::max(1u, create_info.baseHeight >> level);

		VkDeviceSize level_size = get_format_size(image_view.get_format()) *
		                          width * height * create_info.baseDepth;

		for (uint32_t layer = 0; layer < subresource_range.layerCount; layer++)
		{
			ktxTexture_SetImageFromMemory(
			    (ktxTexture *) ktx_texture,
			    level,        // mipLevel
			    0,            // layer
			    layer,        // face
			    raw_data + current_offset,
			    level_size);

			current_offset += level_size;
		}
	}

	ktxTexture_WriteToNamedFile((ktxTexture *) ktx_texture, "assets/textures/output.ktx2");

	ktxTexture_Destroy((ktxTexture *) ktx_texture);
}

PreprocessApp::PreprocessApp()
{}

bool PreprocessApp::prepare(Window *window)
{
	using namespace rendering;

	if (!XiheApp::prepare(window))
	{
		return false;
	}

	load_scene("scenes/helmet.gltf");

	assert(scene_ && "Scene not loaded");

	update_bindless_descriptor_sets();

	xihe::GltfLoader loader(*device_);
	skybox_mesh_ = loader.minimal_read_model("scenes/cube.gltf");

	asset_loader_ = std::make_unique<AssetLoader>(*device_);

	textures_.environment_cube = asset_loader_->load_texture_cube(*scene_, "env_cube", "textures/concrete.ktx");
	// textures_.environment_cube = asset_loader_->load_texture_cube(*scene_, "env_cube", "textures/output.ktx2");

	vk::SamplerCreateInfo default_sampler_info;
	default_sampler_info.magFilter     = vk::Filter::eLinear;
	default_sampler_info.minFilter     = vk::Filter::eLinear;
	default_sampler_info.mipmapMode    = vk::SamplerMipmapMode::eLinear;
	default_sampler_info.addressModeU  = vk::SamplerAddressMode::eClampToEdge;
	default_sampler_info.addressModeV  = vk::SamplerAddressMode::eClampToEdge;
	default_sampler_info.addressModeW  = vk::SamplerAddressMode::eClampToEdge;
	default_sampler_info.minLod        = 0.0f;
	default_sampler_info.maxLod        = 1.0f;
	default_sampler_info.maxAnisotropy = 1.0f;
	default_sampler_info.borderColor   = vk::BorderColor::eFloatOpaqueWhite;

	for (uint32_t target = 0; target < kPrefilter + 1; target++)
	{
		vk::Format format;
		int32_t    dim;

		switch (target)
		{
			case kIrradiance:
				format = vk::Format::eR32G32B32A32Sfloat;
				dim    = 64;
				break;
			case kPrefilter:
				format = vk::Format::eR16G16B16A16Sfloat;
				dim    = 512;
				break;
		}

		PrefilterPass::num_mips = static_cast<uint32_t>(floor(log2(dim))) + 1;

		backend::ImageBuilder image_builder(dim, dim);
		image_builder.with_format(format);
		image_builder.with_array_layers(6);
		image_builder.with_mip_levels(PrefilterPass::num_mips);
		image_builder.with_usage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
		image_builder.with_flags(vk::ImageCreateFlagBits::eCubeCompatible);
		image_builder.with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

		vk::SamplerCreateInfo sampler_info = default_sampler_info;
		sampler_info.maxLod                = static_cast<float>(PrefilterPass::num_mips);

		switch (target)
		{
			case kIrradiance:
				textures_.irradiance_cube.image      = image_builder.build_unique(*device_);
				textures_.irradiance_cube.image_view = std::make_unique<backend::ImageView>(
				    *textures_.irradiance_cube.image, vk::ImageViewType::eCube);
				textures_.irradiance_cube.sampler = std::make_unique<backend::Sampler>(*device_, sampler_info);
				break;
			case kPrefilter:
				textures_.prefiltered_cube.image      = image_builder.build_unique(*device_);
				textures_.prefiltered_cube.image_view = std::make_unique<backend::ImageView>(
				    *textures_.prefiltered_cube.image, vk::ImageViewType::eCube);
				textures_.prefiltered_cube.sampler = std::make_unique<backend::Sampler>(*device_, sampler_info);
				break;
		}

		for (uint32_t m = 0; m < PrefilterPass::num_mips; m++)
		{
			for (uint32_t f = 0; f < 6; f++)

			{
				std::string suffix         = "_mip" + to_string(m) + "_face" + to_string(f);
				auto        prefilter_pass = std::make_unique<PrefilterPass>(*skybox_mesh_, *textures_.environment_cube, m, f, static_cast<PreprocessType>(target));

				std::string attachment_name;

				std::unique_ptr<backend::ImageView> copy_dst_image_view;

				switch (target)
				{
					case kIrradiance:
						attachment_name     = "irradiance_rt";
						copy_dst_image_view = std::make_unique<backend::ImageView>(*textures_.irradiance_cube.image, vk::ImageViewType::e2D, vk::Format::eUndefined, m, f, 1, 1);
						break;
					case kPrefilter:
						attachment_name     = "prefilter_rt";
						copy_dst_image_view = std::make_unique<backend::ImageView>(*textures_.prefiltered_cube.image, vk::ImageViewType::e2D, vk::Format::eUndefined, m, f, 1, 1);
						break;
				}
				PassAttachment attachment{AttachmentType::kColor, attachment_name + suffix};

				uint32_t mip_dim = dim * std::pow(0.5, m);

				attachment.extent_desc = ExtentDescriptor::Fixed({mip_dim, mip_dim, 1});
				attachment.format      = format;
				attachment.is_external = true;

				vk::ImageCopy copy_region{};
				copy_region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
				copy_region.srcSubresource.baseArrayLayer = 0;
				copy_region.srcSubresource.mipLevel       = 0;
				copy_region.srcSubresource.layerCount     = 1;
				copy_region.srcOffset                     = vk::Offset3D{0, 0, 0};

				copy_region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
				copy_region.dstSubresource.baseArrayLayer = f;
				copy_region.dstSubresource.mipLevel       = m;
				copy_region.dstSubresource.layerCount     = 1;
				copy_region.dstOffset                     = vk::Offset3D{0, 0, 0};

				copy_region.extent = vk::Extent3D{mip_dim, mip_dim, 1};

				switch (target)
				{
					case kIrradiance:
						graph_builder_->add_pass("irradiance" + suffix, std::move(prefilter_pass))
						    .attachments({{attachment}})
						    .shader({"preprocess/filtercube.vert", "preprocess/irradiancecube.frag"})
						    .copy(0, std::move(copy_dst_image_view), copy_region)
						    .finalize();
						break;
					case kPrefilter:
						graph_builder_->add_pass("prefilter" + suffix, std::move(prefilter_pass))
						    .attachments({{attachment}})
						    .shader({"preprocess/filtercube.vert", "preprocess/prefilterenvmap.frag"})
						    .copy(0, std::move(copy_dst_image_view), copy_region)
						    .finalize();
						break;
				}
			}
		}
	}

	{
		backend::ImageBuilder image_builder(512, 512);
		image_builder.with_format(vk::Format::eR16G16Sfloat);
		image_builder.with_usage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
		image_builder.with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

		vk::SamplerCreateInfo sampler_info = default_sampler_info;

		textures_.lut_brdf.image      = image_builder.build_unique(*device_);
		textures_.lut_brdf.image_view = std::make_unique<backend::ImageView>(
		    *textures_.lut_brdf.image, vk::ImageViewType::e2D);
		textures_.lut_brdf.sampler = std::make_unique<backend::Sampler>(*device_, sampler_info);

		auto           brdf_pass = std::make_unique<BrdfLutPass>();
		PassAttachment attachment{AttachmentType::kColor, "brdu_lut"};
		attachment.extent_desc = ExtentDescriptor::Fixed(textures_.lut_brdf.image->get_extent());
		attachment.format      = textures_.lut_brdf.image->get_format();
		attachment.is_external = true;

		vk::ImageCopy copy_region{};
		copy_region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		copy_region.srcSubresource.baseArrayLayer = 0;
		copy_region.srcSubresource.mipLevel       = 0;
		copy_region.srcSubresource.layerCount     = 1;
		copy_region.srcOffset                     = vk::Offset3D{0, 0, 0};

		copy_region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		copy_region.dstSubresource.baseArrayLayer = 0;
		copy_region.dstSubresource.mipLevel       = 0;
		copy_region.dstSubresource.layerCount     = 1;
		copy_region.dstOffset                     = vk::Offset3D{0, 0, 0};

		copy_region.extent = textures_.lut_brdf.image->get_extent();

		auto copy_dst_image_view = std::make_unique<backend::ImageView>(*textures_.lut_brdf.image, vk::ImageViewType::e2D, vk::Format::eUndefined, 0, 0, 1, 1);

		graph_builder_->add_pass("brdf_lut_pass", std::move(brdf_pass))
		    .attachments({attachment})
		    .shader({"preprocess/genbrdflut.vert", "preprocess/genbrdflut.frag"})
		    .copy(0, std::move(copy_dst_image_view), copy_region)
		    .finalize();
	}

	graph_builder_->build();

	render_graph_->execute(false);

	get_device()->get_handle().waitIdle();

	// download_image(*textures_.prefiltered_cube.image_view, *device_);

	graph_builder_.reset();
	render_graph_.reset();

	render_graph_  = std::make_unique<rendering::RenderGraph>(*render_context_, stats_.get());
	graph_builder_ = std::make_unique<rendering::GraphBuilder>(*render_graph_, *render_context_);

	auto &camera_node = sg::add_free_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.1f, 1.0f);
	auto  camera      = &camera_node.get_component<sg::Camera>();

	//auto skybox_pass = std::make_unique<SkyboxPass>(*skybox_mesh_, *textures_.environment_cube, *camera);
	////auto skybox_pass = std::make_unique<SkyboxPass>(*scene_->get_components<sg::Mesh>()[0], *textures_.environment_cube, *camera);
	//graph_builder_->add_pass("skybox_pass", std::move(skybox_pass))
	//    .shader({"skybox.vert", "skybox.frag"})
	//    .present()
	//    .finalize();

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

	}

	// geometry pass
	{
		auto geometry_pass = std::make_unique<GeometryPass>(scene_->get_components<sg::Mesh>(), *camera);

		graph_builder_->add_pass("Geometry", std::move(geometry_pass))

		    .attachments({{AttachmentType::kDepth, "depth"},
		                  {AttachmentType::kColor, "albedo"},
		                  {AttachmentType::kColor, "normal", vk::Format::eA2B10G10R10UnormPack32},
		                  {AttachmentType::kColor, "emission"}})

		    .shader({"deferred/geometry.vert", "deferred/geometry.frag"})

		    .finalize();
	}

	// lighting pass
	{
		auto lighting_pass = std::make_unique<LightingPass>(scene_->get_components<sg::Light>(), *camera, p_cascade_script, &textures_.irradiance_cube, &textures_.prefiltered_cube, &textures_.lut_brdf);

		graph_builder_->add_pass("Lighting", std::move(lighting_pass))

		    .bindables({{BindableType::kSampled, "depth"},
		                {BindableType::kSampled, "albedo"},
		                {BindableType::kSampled, "normal"},
		                {BindableType::kSampled, "emission"},
		                {BindableType::kSampled, "shadowmap"}})

		    .attachments({{AttachmentType::kColor, "lighting", vk::Format::eR16G16B16A16Sfloat}})

		    .shader({"deferred/lighting.vert", "deferred/lighting.frag"})

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


	graph_builder_->build();
	return true;
}

void PreprocessApp::update(float delta_time)
{
	XiheApp::update(delta_time);
}

void PreprocessApp::request_gpu_features(backend::PhysicalDevice &gpu)
{
	XiheApp::request_gpu_features(gpu);
}

void PreprocessApp::draw_gui()
{
	gui_->show_stats(*stats_);
}

}        // namespace xihe

//std::unique_ptr<xihe::Application> create_application()
//{
//	return std::make_unique<xihe::PreprocessApp>();
//}
