#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "def.h"

#define TOKEN_MAX 32
#define CWD_SIZE_MAX 64

int exit_flag = 0;

static void show_prompt(char *buf){
  printf("%s > ", getcwd(buf, CWD_SIZE_MAX));
}

int main(int argc, char **argv){
  char* prompt_buf = malloc(CWD_SIZE_MAX);

	while(!exit_flag){
		char *buf = NULL;
		char *argvlist[TOKEN_MAX];

		show_prompt(prompt_buf);

		// input
		argvlist[0] = malloc(sizeof(char *) * TOKEN_MAX);
		if (get_command(buf, argvlist) != 0){
			continue;
		}

		// execute
		if (execute_builtin(argvlist) == 0){
			continue;
		}

		// free
		// free argv list
		free(buf);
	}

  free(prompt_buf);
}
