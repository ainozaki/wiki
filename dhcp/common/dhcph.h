#pragma once

#include <stdint.h>

struct mydhcph {
	uint8_t type;
	uint8_t code;
	uint16_t ttl;
	uint32_t ipaddr;
	uint32_t netmask;
};

enum dhcph_type {
	UNALLOCATED,
	DISCOVER,
	OFFER,
	REQUEST,
	ACK,
	RELEASE,
};

enum dhcph_code {
	SUCCESS,
	UNAVAIRABLE,
	REQUEST_ALLOCATE,
	REQUEST_EXTEND,
	WRONG_MESSAGE,
};

void print_dhcpmsg(struct mydhcph *hdr, int if_send);
