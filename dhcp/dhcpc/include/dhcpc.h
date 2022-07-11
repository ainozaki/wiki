#pragma once

enum client_state {
  INIT,
  WAIT_OFFER,
  WAIT_ACK,
  ALLOCATED,
};
