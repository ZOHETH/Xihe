#pragma once

#include "virtual_texture.h"
#include "xihe_app.h"
#include "rendering/passes/render_pass.h"

namespace xihe
{

class SparseImagePass : public rendering::RenderPass
{
  public:
	SparseImagePass(backend::Device &device);

	~SparseImagePass() override = default;

	void execute(backend::CommandBuffer &command_buffer, rendering::RenderFrame &active_frame, std::vector<rendering::ShaderBindable> input_bindables) override;

  private:
	struct SimpleVertex
	{
		glm::vec2 pos;
		glm::vec2 uv;
	};

	VirtualTexture virtual_texture_;

	std::unique_ptr<backend::Buffer> vertex_buffer_;
	std::unique_ptr<backend::Buffer> index_buffer_;

	uint32_t index_count_{};
};

class TestApp : public XiheApp
{
  public:
	TestApp();

	~TestApp() override = default;

	bool prepare(Window *window) override;

	void update(float delta_time) override;

	void request_gpu_features(backend::PhysicalDevice &gpu) override;

  private:
	void draw_gui() override;

	VirtualTexture virtual_texture_;
};
}        // namespace xihe
