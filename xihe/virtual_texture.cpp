#include "virtual_texture.h"

namespace xihe
{
bool TextureBlock::operator<(TextureBlock const &other) const
{
	if (new_mip_level == other.new_mip_level)
	{
		if (column == other.column)
		{
			return row < other.row;
		}
		return column < other.column;
	}
	return new_mip_level < other.new_mip_level;
}

void MemAllocInfo::get_allocation(PageInfo &page_memory_info, size_t page_index)
{
	// Check if we need to create a new memory sector:
	// - First allocation: no memory sectors yet
	// - Invalid sector: previous sector was released
	// - Full sector: no available space in current sector
	if (memory_sectors.empty() || memory_sectors.front().expired() || memory_sectors.front().lock()->available_offsets.empty())
	{
		// Create new memory sector and allocate from it
		page_memory_info.memory_sector = std::make_shared<MemSector>(*this);
		page_memory_info.offset        = *(page_memory_info.memory_sector->available_offsets.begin());

		page_memory_info.memory_sector->available_offsets.erase(page_memory_info.offset);
		page_memory_info.memory_sector->virtual_page_indices.insert(page_index);

		memory_sectors.push_front(page_memory_info.memory_sector);
	}
	else
	{
		// Reuse existing memory sector
		const auto ptr = memory_sectors.front().lock();

		page_memory_info.memory_sector = ptr;
		page_memory_info.offset        = *(page_memory_info.memory_sector->available_offsets.begin());

		page_memory_info.memory_sector->available_offsets.erase(page_memory_info.offset);
		page_memory_info.memory_sector->virtual_page_indices.insert(page_index);
	}
}

uint32_t MemAllocInfo::get_size() const
{
	return static_cast<uint32_t>(memory_sectors.size());
}

std::list<std::weak_ptr<MemSector>> &MemAllocInfo::get_memory_sectors()
{
	return memory_sectors;
}

MemSector::MemSector(MemAllocInfo &mem_alloc_info) :
    MemAllocInfo(mem_alloc_info)
{
	vk::MemoryAllocateInfo memory_allocate_info;
	memory_allocate_info.allocationSize  = page_size * pages_per_allocation;
	memory_allocate_info.memoryTypeIndex = memory_type_index;

	memory = device.allocateMemory(memory_allocate_info, nullptr);

	for (size_t i = 0; i < pages_per_allocation; ++i)
	{
		available_offsets.insert(static_cast<uint32_t>(i * page_size));
	}
}

MemSector::~MemSector()
{
	device.waitIdle();
	device.freeMemory(memory, nullptr);
}

bool MemSectorCompare::operator()(const std::weak_ptr<MemSector> &left, const std::weak_ptr<MemSector> &right)
{
	if (left.expired())
	{
		return false;
	}
	else if (right.expired())
	{
		return true;
	}
	return left.lock()->available_offsets.size() > right.lock()->available_offsets.size();
}

void VirtualTexture::create_sparse_texture_image(backend::Device &device)
{
	base_mip_level = 0U;
	mip_levels     = 5U;
	{
		backend::ImageBuilder image_builder(width, height);
		image_builder.with_usage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled);
		image_builder.with_flags(vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency);
		image_builder.with_format(raw_data_image->get_format());
		image_builder.with_sharing_mode(vk::SharingMode::eExclusive);
		texture_image = image_builder.build_unique(device);
	}

	std::vector<vk::SparseImageMemoryRequirements> sparse_image_memory_requirements =
	    device.get_handle().getImageSparseMemoryRequirements(texture_image->get_handle());

	vk::MemoryRequirements memory_requirements = device.get_handle().getImageMemoryRequirements(texture_image->get_handle());

	format_properties = sparse_image_memory_requirements[0].formatProperties;

	// imageGranularity represents the minimum sparse image block dimensions
	// For sparse images, memory is allocated in blocks rather than as a continuous chunk
	// The imageGranularity defines the size of these memory blocks:
	// - width: minimum width of a sparse image block
	// - height: minimum height of a sparse image block
	// These values are hardware-dependent and represent the smallest unit of sparse allocation
	uint32_t sparse_block_height = format_properties.imageGranularity.height;
	uint32_t sparse_block_width  = format_properties.imageGranularity.width;

	page_size = sparse_block_height * sparse_block_width * 4U;

	size_t num_total_pages    = 0U;
	size_t current_mip_height = height;
	size_t current_mip_width  = width;

	mip_properties.resize(mip_levels);

	for (uint32_t mip_level = 0U; mip_level < mip_levels; ++mip_level)
	{
		size_t num_rows = (current_mip_height + sparse_block_height - 1) / sparse_block_height;
		size_t num_cols = (current_mip_width + sparse_block_width - 1) / sparse_block_width;

		num_total_pages += num_rows * num_cols;

		mip_properties[mip_level].width               = current_mip_width;
		mip_properties[mip_level].height              = current_mip_height;
		mip_properties[mip_level].num_rows            = num_rows;
		mip_properties[mip_level].num_columns         = num_cols;
		mip_properties[mip_level].mip_num_pages       = num_rows * num_cols;
		mip_properties[mip_level].mip_base_page_index = mip_level > 0U ? mip_properties[mip_level - 1U].mip_base_page_index + mip_properties[mip_level - 1U].mip_num_pages : 0U;

		if (current_mip_height > 1U)
		{
			current_mip_height >>= 1U;
		}
		if (current_mip_width > 1U)
		{
			current_mip_width >>= 1U;
		}
	}
	width  = mip_properties[0].width;
	height = mip_properties[0].height;

	page_table.resize(num_total_pages);
	sparse_image_memory_binds.resize(num_total_pages);

	reset_mip_table();

	memory_allocations.device = device.get_handle();
	memory_allocations.page_size = page_size;
	memory_allocations.pages_per_allocation = kPagesPerAlloc;

	for (size_t page_index = 0U; page_index < page_table.size(); ++page_index)
	{
		uint32_t mip_level = get_mip_level(page_index);
		// todo
	}
}

void VirtualTexture::reset_mip_table()
{
	current_mip_table.clear();
	current_mip_table.resize(num_vertical_blocks);

	new_mip_table.clear();
	new_mip_table.resize(num_vertical_blocks);

	for (size_t y = 0U; y< num_vertical_blocks; ++y)
	{
		current_mip_table[y].resize(num_horizontal_blocks);
		new_mip_table[y].resize(num_horizontal_blocks);

		for (size_t x = 0U; x < num_horizontal_blocks; ++x)
		{
			current_mip_table[y][x].on_screen = false;
			new_mip_table[y][x].on_screen     = false;
		}
	}

	for (auto &page : page_table)
	{
		if (!page.fixed)
		{
			page.render_required_set.clear();
		}
	}
}

uint32_t VirtualTexture::get_mip_level(size_t page_index) const
{
	uint8_t mip_level = 0U;
	if (mip_levels == 1U)
	{
		return base_mip_level;
	}
	for (uint8_t i = base_mip_level; i < mip_levels; ++i)
	{
		if (page_index < mip_properties[i].mip_base_page_index + mip_properties[i].mip_num_pages)
		{
			mip_level = i;
			break;
		}
	}
	return mip_level;
}

CalculateMipLevelData::CalculateMipLevelData(const glm::mat4 &mvp_transform, const vk::Extent2D &texture_base_dim, const vk::Extent2D &screen_base_dim, uint32_t vertical_num_blocks, uint32_t horizontal_num_blocks, uint8_t mip_levels) :
    mesh(vertical_num_blocks + 1U),
    vertical_num_blocks(vertical_num_blocks),
    horizontal_num_blocks(horizontal_num_blocks),
    mip_levels(mip_levels),
    ax_vertical(horizontal_num_blocks + 1U),
    ax_horizontal(vertical_num_blocks + 1U),
    mvp_transform(mvp_transform),
    texture_base_dim(texture_base_dim),
    screen_base_dim(screen_base_dim)
{
	for (auto &row : mesh)
	{
		row.resize(horizontal_num_blocks + 1U);
	}
}
}        // namespace xihe
