#include "def.h"

#include <stdio.h>
#include <string.h>

int get_command(char *buf, char *argv[]){
	ssize_t readlen;
	size_t len;
	char *token;
	int index = 0;

	/* get input */
	readlen = getline(&buf, &len, stdin);
	if (readlen == -1){
		fprintf(stderr, "Invalid input\n");
		return -1;
	}

	/* parse */
	const char* delimiter = " \n";
	token = strtok(buf, delimiter);
	if (!token){
		return -1;
	}
	while(token){
		argv[index++] = token;
		token = strtok(NULL, delimiter);
	}
	/* Add NULL at the end of argv list */
	argv[index] = NULL;
	
	show_argv_list(argv);
	return 0;
}

void show_argv_list(char *argv[]){
	int n = 0;
	while (argv[n]){
		printf("%2d: %s\n", n, argv[n]);
		n++;
	}
}
