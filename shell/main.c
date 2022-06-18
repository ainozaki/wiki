// 61915407 野崎愛

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "def.h"

#define TOKEN_MAX 32
#define CWD_SIZE_MAX 64

int exit_flag = 0;

static void show_prompt(char *buf) {
  printf("%s > ", getcwd(buf, CWD_SIZE_MAX));
}

int main(int argc, char **argv) {
	if (argc != 1){
		fprintf(stderr, "usage: %s\n", argv[0]);
		exit(1);
	}

  char *prompt_buf = malloc(CWD_SIZE_MAX);
  char *argv_token[TOKEN_MAX];

  init_jobs();
  argv_token[0] = malloc(sizeof(char *) * TOKEN_MAX);

  while (!exit_flag) {
    char *buf = NULL;

    show_prompt(prompt_buf);

    // ignore signals
    ignore_signal(SIGINT);
    ignore_signal(SIGTSTP);
    ignore_signal(SIGTTIN);
    ignore_signal(SIGTTOU);

    // input
    if (get_command(buf, argv_token) != 0) {
      continue;
    }

    // execute builtin
    if (execute_builtin(argv_token) == 0) {
      continue;
    }

    // execute
    tursh_exec(argv_token);

    // free
    free(buf);
  }

  // TODO: free argv_token
  printf("terminating shell...\n");
  free(prompt_buf);
}
