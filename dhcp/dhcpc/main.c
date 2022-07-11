#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "dhcpc.h"
#include "dhcph.h"

/*
// Internet address
struct in_addr {
        in_addr_t s_addr;
};

// Socket address, internet style.
struct sockaddr_in {
        uint8_t sin_len;
        sa_family_t sin_family; // アドレスファミリー (AF_INET)
        in_port_t sin_port; // ポート番号
        struct in_addr sin_addr; // IP アドレス
        char sin_zero[8];
};
*/

int status, st;
struct in_addr server_addr;

const char *status_tbl[] = {
    "INIT",
    "WAIT_OFFER",
    "WAIT_ACK",
    "ALLOCATED",
};

void change_status(int nextst) {
  printf("STATUS CHANGED: [%s] -> [%s]\n", status_tbl[status],
         status_tbl[nextst]);
  status = nextst;
}

void init(){
  // socket
  if ((st = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    fprintf(stderr, "cannot open socket");
    return;
	}
}

void send_to_server(const struct mydhcph *msg, size_t msglen) {
  int count;
  struct sockaddr_in skt;
  in_port_t port;

  // set sockaddr_in
  memset(&skt, 0, sizeof(skt));
  skt.sin_family = AF_INET;
  port = 51230;
  skt.sin_port = htons(port);
  skt.sin_addr.s_addr = server_addr.s_addr;

  // send
  if ((count = sendto(st, msg, msglen, 0, (struct sockaddr *)&skt,
                      sizeof(skt))) < 0) {
    perror("sendto");
    return;
  }
  print_dhcpmsg((struct mydhcph *)msg, /*if_send=*/1);
}

struct mydhcph *wait_server() {
  int count;
  struct sockaddr_in skt;
  socklen_t sktlen;
  char *rbuf = malloc(sizeof(char) * 512);

  printf("waiting for server...\n");

  // recv
  sktlen = sizeof(skt);
  if ((count = recvfrom(st, rbuf, sizeof(rbuf), 0, (struct sockaddr *)&skt,
                        &sktlen)) < 0) {
    perror("recvfrom");
    return NULL;
  }
  print_dhcpmsg((struct mydhcph *)rbuf, /*if_send=*/0);

  return (struct mydhcph *)rbuf;
}

void send_discover() {
  struct mydhcph header = {0};
  header.type = DISCOVER;
  send_to_server(&header, sizeof(header));
}

void wait_offer(){
	struct mydhcph *msg;
	msg = wait_server();
	if (!msg){
		return;
	}
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <server-IP-address>\n", argv[0]);
    return 1;
  }
  printf("starting dhcpc\n");

  const char *server_ip = argv[1];
  if (inet_aton(server_ip, &server_addr) != 1) {
    fprintf(stderr, "cannot convert %s\n", server_ip);
    return 1;
  }

  status = INIT;

  for (;;) {
    switch (status) {
    case INIT:
			init();
      send_discover();
      change_status(WAIT_OFFER);
			break;
    case WAIT_OFFER:
      wait_offer();
      // send_request();
      for (;;)
        ;
      break;
    default:
      fprintf(stderr, "unknown status %d\n", status);
    }
  }
  return 0;
}
