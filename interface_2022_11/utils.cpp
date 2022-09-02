#include "utils.h"

#include <cstdint>
#include <cstdio>

uint8_t mac_addr_toa_string_pool_index = 0;
char mac_addr_toa_string_pool
    [4][18]; // 18バイト(xxx.xxx.xxx.xxxの文字数+1)の領域を4つ確保

/**
 * MACアドレスから文字列に変換
 * @param addr
 * @return
 */
const char *mac_addr_toa(const uint8_t *addr) {
  mac_addr_toa_string_pool_index++;
  mac_addr_toa_string_pool_index %= 4;
  sprintf(mac_addr_toa_string_pool[mac_addr_toa_string_pool_index],
          "%02x:%02x:%02x:%02x:%02x:%02x", addr[0], addr[1], addr[2], addr[3],
          addr[4], addr[5]);
  return mac_addr_toa_string_pool[mac_addr_toa_string_pool_index];
}
