#include "ethernet.h"

#include "my_buf.h"
#include "utils.h"
#include <cstring>

/**
 * イーサネットの受信処理
 * @param dev 受信したデバイス
 * @param buffer 受信したデータのバイト列
 * @param len 受信したデータの長さ
 */
void ethernet_input(net_device *dev, uint8_t *buffer, ssize_t len) {
  printf("ethernet received!\n");

  /*
// 送られてきた通信をイーサネットのフレームとして解釈する
auto *header = reinterpret_cast<ethernet_header *>(buffer);
uint16_t ethernet_type = ntohs(header->type); //
イーサネットタイプを抜き出すし、ホストバイトオーダーに変換

// 自分のMACアドレス宛てかブロードキャストの通信かを確認する
if(memcmp(header->dest_address, dev->mac_address, 6) != 0 and
memcmp(header->dest_address, ETHERNET_ADDRESS_BROADCAST, 6) != 0){
return;
}

LOG_ETHERNET("Received ethernet frame type %04x from %s to %s\n",
   ethernet_type, mac_addr_toa(header->src_address),
   mac_addr_toa(header->dest_address));

// イーサネットタイプの値から上位プロトコルを特定する
switch(ethernet_type){
case ETHERNET_TYPE_ARP: // イーサネットタイプがARPのものだったら
return arp_input(
      dev,
      buffer + ETHERNET_HEADER_SIZE,
      len - ETHERNET_HEADER_SIZE
); // Ethernetヘッダを外してARP処理へ
case ETHERNET_TYPE_IP: // イーサネットタイプがIPのものだったら
return ip_input(
      dev,
      buffer + ETHERNET_HEADER_SIZE,
      len - ETHERNET_HEADER_SIZE
); // Ethernetヘッダを外してIP処理へ
default: // 知らないイーサネットタイプだったら
LOG_ETHERNET("Received unhandled ethernet type %04x\n", ethernet_type);
return;
}
          */
}
