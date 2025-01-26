#pragma once
#include "scene_graph/components/image.h"

namespace xihe
{
constexpr uint8_t kPagesPerAlloc = 50U;

struct MipProperties
{
	size_t num_rows;
	size_t num_columns;
	size_t mip_num_pages;
	size_t mip_base_page_index;
	size_t width;
	size_t height;
};

struct TextureBlock
{
	bool operator<(TextureBlock const &other) const;

	size_t row;
	size_t column;
	double old_mip_level;
	double new_mip_level;
	bool   on_screen;
};

struct MemPageDescription
{
	size_t  x;
	size_t  y;
	uint8_t mip_level;
};

struct Point
{
	double x;
	double y;
	bool   on_screen;
};

struct MipBlock
{
	double mip_level;
	bool   on_screen;
};

struct MemSector;

struct PageInfo
{
	std::shared_ptr<MemSector> memory_sector{nullptr};
	uint32_t                   offset = 0U;
};

struct Page
{
	bool valid{false};        // bound via queueBindSparse

	bool     gen_mip_required{false};
	bool     fixed{false};
	PageInfo page_memory_info;

	std::set<std::tuple<uint8_t, size_t, size_t>> render_required_set;
};

/**
 * \brief Memory pool allocator designed for Sparse Binding in virtual texturing system.
 * This allocator manages continuous GPU memory blocks (MemSector) divided into fixed-size pages.
 * It uses a LIFO strategy for memory allocation to maintain memory locality and efficient reuse.
 */
struct MemAllocInfo
{
	vk::Device device{VK_NULL_HANDLE};
	uint64_t   page_size{0U};
	uint32_t   memory_type_index{0U};
	size_t     pages_per_allocation{0U};

	/**
	 * Memory allocation strategy:
	 * 1. Tries to reuse existing memory sectors first (LIFO order)
	 * 2. Creates new memory sector when necessary
	 * 3. Updates mapping between virtual pages and physical memory
	 */
	void get_allocation(PageInfo &page_memory_info, size_t page_index);

	uint32_t get_size() const;

	std::list<std::weak_ptr<MemSector>> &get_memory_sectors();

  private:
	// Use weak_ptr for memory sectors to allow automatic cleanup
	// when memory sectors are no longer needed
	std::list<std::weak_ptr<MemSector>> memory_sectors;
};

/**
 * \brief Represents a continuous block of GPU memory for sparse binding
 *
 * MemSector manages a fixed-size block of GPU memory divided into pages.
 * It tracks both available memory locations and their usage by virtual pages.
 */
struct MemSector : public MemAllocInfo
{
	vk::DeviceMemory memory{VK_NULL_HANDLE};

	/**
	 * \brief Tracks available offsets in this memory sector
	 * - Each offset represents the starting position of a page in the continuous memory block
	 * - Offsets are removed when pages are allocated and added back when pages are freed
	 * - The distance between offsets equals page_size
	 */
	std::set<uint32_t> available_offsets;

	/**
	 * \brief Tracks which virtual pages are using this memory sector
	 * - Maps physical memory pages to virtual texture pages
	 * - Used for resource management and cleanup
	 * - Helps maintain the relationship between virtual pages and physical memory
	 */
	std::set<size_t> virtual_page_indices;

	MemSector(MemAllocInfo &mem_alloc_info);

	~MemSector();
};

struct MemSectorCompare
{
	bool operator()(const std::weak_ptr<MemSector> &left, const std::weak_ptr<MemSector> &right);
};

struct VirtualTexture
{
	std::unique_ptr<backend::Image> texture_image;
	vk::ImageView                   texture_image_view = VK_NULL_HANDLE;
	MemAllocInfo                    memory_allocations;

	size_t width  = 0U;
	size_t height = 0U;

	size_t page_size = 0U;

	uint8_t                    base_mip_level = 0U;
	uint8_t                    mip_levels     = 0U;
	std::vector<MipProperties> mip_properties;

	std::vector<std::vector<MipBlock>> current_mip_table;
	std::vector<std::vector<MipBlock>> new_mip_table;

	std::unique_ptr<sg::Image> raw_data_image;

	std::vector<Page> page_table;

	std::set<TextureBlock> texture_block_update_set;

	std::set<size_t> update_set;

	vk::SparseImageFormatProperties format_properties{};

	std::vector<vk::SparseImageMemoryBind> sparse_image_memory_binds;

	size_t num_vertical_blocks   = 50U;
	size_t num_horizontal_blocks = 50U;

	void create_sparse_texture_image(backend::Device &device);

  private:
	void     reset_mip_table();
	uint32_t get_mip_level(size_t page_index) const;
};

struct CalculateMipLevelData
{
	std::vector<std::vector<Point>>    mesh{0};
	std::vector<std::vector<MipBlock>> mip_table;

	uint32_t vertical_num_blocks{0U};
	uint32_t horizontal_num_blocks{0U};

	uint8_t mip_levels;

	std::vector<float> ax_vertical;
	std::vector<float> ax_horizontal;

	glm::mat4 mvp_transform{0};

	vk::Extent2D texture_base_dim{};
	vk::Extent2D screen_base_dim{};

	CalculateMipLevelData() = default;

	CalculateMipLevelData(const glm::mat4 &mvp_transform, const vk::Extent2D &texture_base_dim, const vk::Extent2D &screen_base_dim, uint32_t vertical_num_blocks, uint32_t horizontal_num_blocks, uint8_t mip_levels);

	void calculate_mesh_coordinates();
	void calculate_mip_levels();
};
}        // namespace xihe
