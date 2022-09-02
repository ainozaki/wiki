#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <ifaddrs.h>
#include <iostream>
#include <linux/if_ether.h>
#include <net/if.h>
#include <netpacket/packet.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include "ethernet.h"
#include "net.h"
#include "utils.h"

#define ENABLE_INTERFACES                                                      \
  {                                                                            \
    "router1-host1", "router1-router2", "router1-router4", "router1-router3",  \
        "router1-br0"                                                          \
  }

net_device *net_dev_list;

/*
        デバイス依存のデータ
*/
struct net_device_data {
  int fd;
};

int net_device_transmit(struct net_device *dev, my_buf *buf) { return 0; }
int net_device_poll(net_device *dev) {
  uint8_t buffer[1550];
  ssize_t n = recv(((net_device_data *)dev->data)->fd, buffer, sizeof(buffer),
                   0); // socketから受信
  if (n == -1) {
    if (errno == EAGAIN) {
      return 0;
    } else {
      return -1;
    }
  }
  ethernet_input(dev, buffer, n); // 受信したデータをイーサネットに送る
  return 0;
}

bool is_enable_interface(const char *ifname) {
  char enable_interfaces[][IF_NAMESIZE] = ENABLE_INTERFACES;

  for (int i = 0; i < sizeof(enable_interfaces) / IF_NAMESIZE; i++) {
    if (strcmp(enable_interfaces[i], ifname) == 0) {
      return true;
    }
  }
  return false;
}

int main() {

  struct ifreq ifr {};
  struct ifaddrs *addrs;

  // ネットワークインターフェースを情報を取得
  getifaddrs(&addrs);

  for (ifaddrs *tmp = addrs; tmp; tmp = tmp->ifa_next) {
    if (tmp->ifa_addr && tmp->ifa_addr->sa_family == AF_PACKET) {

      // ioctlでコントロールするインターフェースを設定
      memset(&ifr, 0, sizeof(ifr));
      strcpy(ifr.ifr_name, tmp->ifa_name);

      // 有効化するインターフェースか確認
      if (!is_enable_interface(tmp->ifa_name)) {
        printf("Skipped to enable interface %s\n", tmp->ifa_name);
        continue;
      }

      // Socketをオープン
      int sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
      if (sock == -1) {
        perror("socket");
        continue;
      }

      // インターフェースのMACアドレスを取得
      if (ioctl(sock, SIOCGIFHWADDR, &ifr) != 0) {
        perror("ioctl");
        close(sock);
        continue;
      }

      // net_device構造体を作成
      auto *dev =
          (net_device *)calloc(1, sizeof(net_device) + sizeof(net_device_data));
      dev->ops.transmit = net_device_transmit; // 送信用の関数を設定
      dev->ops.poll = net_device_poll;         // 受信用の関数を設定

      ((net_device_data *)dev->data)->fd = sock; //
      strcpy(dev->ifname, tmp->ifa_name);

      memcpy(dev->mac_address, &ifr.ifr_hwaddr.sa_data[0], 6);

      // インターフェースのインデックスを取得
      if (ioctl(sock, SIOCGIFINDEX, &ifr) == -1) {
        close(sock);
        continue;
      }

      sockaddr_ll addr{};
      memset(&addr, 0x00, sizeof(addr));
      addr.sll_family = AF_PACKET;
      addr.sll_protocol = htons(ETH_P_ALL);
      addr.sll_ifindex = ifr.ifr_ifindex;
      if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        close(sock);
        free(dev);
        continue;
      }

      printf("[DEV] Created dev %s sock %d addr %s \n", dev->ifname, sock,
             mac_addr_toa(dev->mac_address));

      // 連結させる
      net_device *next;
      next = net_dev_list;
      net_dev_list = dev;
      dev->next = next;

      // ノンブロッキングに設定
      int val = fcntl(sock, F_GETFL, 0);      // File descriptorのflagを取得
      fcntl(sock, F_SETFL, val | O_NONBLOCK); // Non blockingのbitをセット
    }

    // 確保されていたメモリを解放
    // freeifaddrs(addrs);

    // 1つも有効なインターフェースをが無かったら終了
    if (net_dev_list == nullptr) {
      printf("No interface is enabled!\n");
      return 0;
    }

    while (true) {
      // インターフェースから通信を受信
      for (net_device *a = net_dev_list; a; a = a->next) {
        a->ops.poll(a);
      }
    }
  }
}
