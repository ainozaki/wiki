#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

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

void send_to_server(const char *server_ip){
	int s, count, datalen;
	struct sockaddr_in skt;
	in_port_t port;
	struct in_addr ipaddr;
	char sbuf[512];

	printf("sending to %s...\n", server_ip);

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
	if (inet_aton(server_ip, &ipaddr) != 1){
		fprintf(stderr, "cannot convert %s\n", server_ip);
		return;
	}
	skt.sin_addr.s_addr = ipaddr.s_addr;
	
	// send
	memset(sbuf, 'a', 512);
	datalen = 128;
	if ((count = sendto(s, sbuf, datalen, 0, (struct sockaddr *)&skt, sizeof(skt))) < 0){
		perror("sendto");
		return;
	}
	printf("send %d byte\n", count);
}

int main(int argc, char *argv[]){
	if (argc < 2){
		fprintf(stderr, "usage: %s <server-IP-address>\n", argv[0]);
		return 1;
	}
	printf("starting dhcpc\n");

	// connect to server
	send_to_server(argv[1]);

	return 0;
}
