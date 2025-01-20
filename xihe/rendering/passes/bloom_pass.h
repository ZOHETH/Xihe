#pragma once

#include "render_pass.h"

namespace xihe::rendering
{
struct alignas(16) CommonUniforms
{
	glm::uvec2 resolution;
	glm::vec2  inv_resolution;
	glm::vec2  inv_input_resolution;
};

struct ExtractPush
{
	float threshold      = 1.0f;
	float soft_threshold = 2.0f;
	float intensity      = 0.5f;
	float saturation     = 1.0f;
};

struct CompositePush
{
	float bloom_strength = 0.2f;
	float exposure       = 1.5f;
};

class BloomComputePass : public RenderPass
{
  public:
	BloomComputePass() = default;
	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

  protected:
	CommonUniforms uniforms_;
};

class BloomExtractPass : public BloomComputePass
{
  public:
	BloomExtractPass() = default;
	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;
};

class BloomDownsamplePass : public BloomComputePass
{
  public:
	BloomDownsamplePass(float filter_radius);
	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

  private:
	float filter_radius_ = 1.0;
};

class BloomCompositePass : public RenderPass
{
  public:
	BloomCompositePass() = default;
	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;
};

}        // namespace xihe::rendering