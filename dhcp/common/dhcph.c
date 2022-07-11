#include "dhcph.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

void print_dhcpmsg(struct mydhcph *hdr, int if_send) {
  struct in_addr ipaddr;
  struct in_addr netmask;

	if (if_send){
		printf("SEND: ");
	}else{
		printf("RECEIVED: ");
	}

  ipaddr = *(struct in_addr *)&(hdr->ipaddr);
  netmask = *(struct in_addr *)&(hdr->netmask);
  printf("type=%d, code=%d, ttl=%d, ipaddr=%s, netmask=%s\n", hdr->type,
         hdr->code, hdr->ttl, inet_ntoa(ipaddr), inet_ntoa(netmask));
}
