#pragma once

#define DEFAULT_TTL 64

enum server_state {
  INIT,
  WAIT_DISCOVER,
  WAIT_REQUEST,
  ALLOCATED,
};

struct client {
	struct client *fp;
	struct client *bp;
	short status;
	int ttlcounter;
	// network order
	struct in_addr id;
	struct in_addr addr;
	struct in_addr netmask;
	uint16_t ttl;
};
