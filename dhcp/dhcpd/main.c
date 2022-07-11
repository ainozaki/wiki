#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

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
int status;
// TODO: create client list
struct client client;

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
	printf("Inialize server\n");
}

struct mydhcph *wait_client() {
  int s, count;
  struct sockaddr_in myskt;
  struct sockaddr_in skt;
  socklen_t sktlen;
  in_port_t port;
  char *rbuf = malloc(sizeof(char) * 512);

  printf("waiting for client...\n");

  // socket
  if ((s = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket");
    return NULL;
  }

  // bind
  memset(&myskt, 0, sizeof(myskt));
  myskt.sin_family = AF_INET;
  port = 51230;
  myskt.sin_port = htons(port);
  myskt.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(s, (struct sockaddr *)&myskt, sizeof(myskt)) < 0) {
    perror("bind");
    return NULL;
  }

  // recv
  sktlen = sizeof(skt);
  if ((count = recvfrom(s, rbuf, sizeof(rbuf), 0, (struct sockaddr *)&skt,
                        &sktlen)) < 0) {
    perror("recvfrom");
    return NULL;
  }
	print_dhcpmsg((struct mydhcph *)rbuf, /*if_send=*/0);

	return (struct mydhcph *)rbuf;
}

void send_to_client(struct client *client, const struct mydhcph *msg, size_t msglen){
	int s, count;
	struct sockaddr_in skt;
	in_port_t port;

	// socket
	if ((s = socket(AF_INET, SOCK_DGRAM, 0)) < 0){
		fprintf(stderr, "cannot open socket");
		return;
	}

	// set sockaddr_in
	memset(&skt, 0, sizeof(skt));
	skt.sin_family = AF_INET;
	port = 51230;
	skt.sin_port = htons(port);
	skt.sin_addr = client->id;
	
	// send
	if ((count = sendto(s, msg, msglen, 0, (struct sockaddr *)&skt, sizeof(skt))) < 0){
		perror("sendto");
		return;
	}
	print_dhcpmsg((struct mydhcph *)msg, /*if_send=*/1);
}

void wait_discover(){
	struct mydhcph *msg;

	msg = wait_client();
	if (!msg){
		return;
	}

	client.status = 0;
	client.ttlcounter = DEFAULT_TTL;
	memcpy(&(client.id), &msg->ipaddr, 4);
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
	msg.ipaddr = *(uint32_t *)&addr;
	inet_aton("255.255.255.0", &netmask);
	msg.netmask = *(uint32_t *)&netmask;
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
			send_offer();
      change_status(WAIT_REQUEST);
      for (;;)
        ;
    }
  }

  return 0;
}
