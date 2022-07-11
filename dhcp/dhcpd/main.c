#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "dhcpd.h"

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
int status;

const char *status_tbl[] = {
	"INIT",
	"WAIT_DISCOVER",
	"WAIT_REQUEST",
	"ALLOCATED",
};

void change_status(int nextst){
	printf("STATUS: [%s] -> [%s]\n", status_tbl[status], status_tbl[nextst]);
	status = nextst;
}

void wait_client() {
  int s, count;
  struct sockaddr_in myskt;
  struct sockaddr_in skt;
  socklen_t sktlen;
  in_port_t port;
  char rbuf[512];

  printf("waiting for client...\n");

  // socket
  if ((s = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket");
    return;
  }

  // bind
  memset(&myskt, 0, sizeof(myskt));
  myskt.sin_family = AF_INET;
  port = 51230;
  myskt.sin_port = htons(port);
  myskt.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(s, (struct sockaddr *)&myskt, sizeof(myskt)) < 0) {
    perror("bind");
    return;
  }

  // recv
  sktlen = sizeof(skt);
  if ((count = recvfrom(s, rbuf, sizeof(rbuf), 0, (struct sockaddr *)&skt,
                        &sktlen)) < 0) {
    perror("recvfrom");
    return;
  }
  printf("received %d byte\n", count);

  // send
}

int main() {
  printf("starting dhcpd\n");

	status = INIT;
	
	for(;;){
		switch (status){
			case INIT:
  			wait_client();
				change_status(WAIT_DISCOVER);
				for(;;);
		}
	}

  return 0;
}
