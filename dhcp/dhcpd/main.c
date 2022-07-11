#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "dhcpd.h"
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
// TODO: create client list
struct client client;
struct in_addr client_addr;
int client_port;

const char *status_tbl[] = {
    "INIT",
    "WAIT_DISCOVER",
    "WAIT_REQUEST",
    "ALLOCATED",
};

void change_status(int nextst) {
  printf("STATUS CHANGED: [%s] -> [%s]\n", status_tbl[status], status_tbl[nextst]);
  status = nextst;
}

void init(){
  struct sockaddr_in myskt;
  in_port_t port;
	printf("Inialize server\n");
  // socket
  if ((st = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket");
    return;
	}
  
	// bind
  memset(&myskt, 0, sizeof(myskt));
  myskt.sin_family = AF_INET;
  port = 51230;
  myskt.sin_port = htons(port);
  myskt.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(st, (struct sockaddr *)&myskt, sizeof(myskt)) < 0) {
    perror("bind");
    return;
  }
}

struct mydhcph *wait_client() {
  int count;
  struct sockaddr_in skt;
  socklen_t sktlen;
  char *rbuf = malloc(sizeof(char) * 512);

  printf("waiting for client...\n");

  // recv
  sktlen = sizeof(skt);
  if ((count = recvfrom(st, rbuf, sizeof(rbuf), 0, (struct sockaddr *)&skt,
                        &sktlen)) < 0) {
    perror("recvfrom");
    return NULL;
  }
	client_port = skt.sin_port;
	client_addr = skt.sin_addr;
	printf("received from %s:%d\n", inet_ntoa(client_addr), client_port);
	print_dhcpmsg((struct mydhcph *)rbuf, /*if_send=*/0);

	return (struct mydhcph *)rbuf;
}

void send_to_client(struct client *client, const struct mydhcph *msg, size_t msglen){
	int count;
	struct sockaddr_in skt;

	// set sockaddr_in
	memset(&skt, 0, sizeof(skt));
	skt.sin_family = AF_INET;
	skt.sin_port = htons(client_port);
	skt.sin_addr = client_addr;
	
	// send
	if ((count = sendto(st, msg, msglen, 0, (struct sockaddr *)&skt, sizeof(skt))) < 0){
		perror("sendto");
		return;
	}
	print_dhcpmsg((struct mydhcph *)msg, /*if_send=*/1);
	printf("send to %s:%d\n", inet_ntoa(client_addr), client_port);
}

void wait_discover(){
	struct mydhcph *msg;

	msg = wait_client();
	if (!msg){
		return;
	}

	client.status = 0;
	client.ttlcounter = DEFAULT_TTL;
	memcpy(&(client.id), &client_addr, 4);
	client.ttl = DEFAULT_TTL;

	free(msg);
}

void send_offer(){
	struct in_addr addr, netmask;

	struct mydhcph msg = {0};
	msg.type = OFFER;
	msg.code = SUCCESS;
	msg.ttl = DEFAULT_TTL;
	// TODO: create avirable address list
	inet_aton("192.168.210.100", &addr);
	memcpy((void *)&msg.ipaddr, (void *)&addr, 4);
	inet_aton("255.255.255.0", &netmask);
	memcpy((void *)&msg.netmask, (void *)&netmask, 4);
	send_to_client(&client, &msg, sizeof(msg));
}

int main() {
  printf("starting dhcpd\n");

  status = INIT;

  for (;;) {
    switch (status) {
    case INIT:
			init();
      change_status(WAIT_DISCOVER);
			break;
		case WAIT_DISCOVER:
      wait_discover();
			sleep(1);
			send_offer();
      change_status(WAIT_REQUEST);
      for (;;)
        ;
    }
  }

  return 0;
}
