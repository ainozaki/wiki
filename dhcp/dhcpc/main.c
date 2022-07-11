#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

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

int status;
struct in_addr server_addr;

const char *status_tbl[] = {
	"INIT",
	"WAIT_OFFER",
	"WAIT_ACK",
	"ALLOCATED",
};

void change_status(int nextst){
	printf("STATUS CHANGED: [%s] -> [%s]\n", status_tbl[status], status_tbl[nextst]);
	status = nextst;
}

void send_to_server(const struct mydhcph *msg, size_t msglen){
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
	skt.sin_addr.s_addr = server_addr.s_addr;
	
	// send
	if ((count = sendto(s, msg, msglen, 0, (struct sockaddr *)&skt, sizeof(skt))) < 0){
		perror("sendto");
		return;
	}
	print_dhcpmsg((struct mydhcph *)msg, /*if_send=*/1);
}

void send_discover(){
	struct mydhcph header = {0};
	header.type = DISCOVER;
	send_to_server(&header, sizeof(header));
}

int main(int argc, char *argv[]){
	if (argc < 2){
		fprintf(stderr, "usage: %s <server-IP-address>\n", argv[0]);
		return 1;
	}
	printf("starting dhcpc\n");
	
	const char *server_ip = argv[1];
	if (inet_aton(server_ip, &server_addr) != 1){
		fprintf(stderr, "cannot convert %s\n", server_ip);
		return 1;
	}

	status = INIT;

	for (;;){
		switch (status){
			case INIT:
				// connect to server
				send_discover();
				change_status(WAIT_OFFER);
				for(;;);
				break;
			default:
				fprintf(stderr, "unknown status %d\n", status);
		}
	}
	return 0;
}
