#pragma once

#include <cstdint>

struct my_buf {
	my_buf *prev_my_buf = nullptr;
	my_buf *next_my_buf = nullptr;
};
