#include "test_app.h"

#include "scene_graph/components/camera.h"

namespace xihe
{
SparseImagePass::SparseImagePass(backend::Device &device)
{
	std::array<SimpleVertex, 4> vertices;
	vertices[0].pos = {-100.0f, -100.0f};
	vertices[1].pos = {100.0f, -100.0f};
	vertices[2].pos = {100.0f, 100.0f};
	vertices[3].pos = {-100.0f, 100.0f};

	vertices[0].uv = {0.0f, 0.0f};
	vertices[1].uv = {1.0f, 0.0f};
	vertices[2].uv = {1.0f, 1.0f};
	vertices[3].uv = {0.0f, 1.0f};

	{
		backend::BufferBuilder buffer_builder(sizeof(vertices[0]) * vertices.size());

		buffer_builder.with_usage(vk::BufferUsageFlagBits::eVertexBuffer).with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		vertex_buffer_ = buffer_builder.build_unique(device);
		vertex_buffer_->update(vertices);
	}

	std::array<uint16_t, 6U> indices = {0U, 1U, 2U, 2U, 3U, 0U};
	index_count_                     = indices.size();
	{
		backend::BufferBuilder buffer_builder(sizeof(indices[0]) * indices.size());
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eIndexBuffer).with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		index_buffer_ = buffer_builder.build_unique(device);
		index_buffer_->update(indices);
	}
}

void SparseImagePass::execute(backend::CommandBuffer &command_buffer, rendering::RenderFrame &active_frame, std::vector<rendering::ShaderBindable> input_bindables)
{
	RenderPass::execute(command_buffer, active_frame, input_bindables);
}

TestApp::TestApp()
{
}

bool TestApp::prepare(Window *window)
{
	if (!XiheApp::prepare(window))
	{
		return false;
	}

	virtual_texture_.raw_data_image = sg::Image::load("vulkan_logo", "/textures/vulkan_logo_full.ktx", sg::Image::ContentType::kColor);

	assert(virtual_texture_.raw_data_image->get_format() == vk::Format::eR8G8B8A8Srgb);

	const auto extent       = virtual_texture_.raw_data_image->get_extent();
	virtual_texture_.width  = extent.width;
	virtual_texture_.height = extent.height;

	auto &camera_node = sg::add_free_camera(*scene_, "default_camera", render_context_->get_surface_extent());
	auto  camera      = &camera_node.get_component<sg::Camera>();

	render_context_->create_sparse_bind_queue();

	return true;
}

void TestApp::update(float delta_time)
{
	XiheApp::update(delta_time);
}

void TestApp::request_gpu_features(backend::PhysicalDevice &gpu)
{
	XiheApp::request_gpu_features(gpu);
}

void TestApp::draw_gui()
{
	XiheApp::draw_gui();
}
}        // namespace xihe