#include "def.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CWD_SIZE_MAX 64

static void do_cd(char **argv) {
  if (!argv[1]) {
    printf("usage: cd <path>\n");
    return;
  }
  if (chdir(argv[1]) != 0) {
    perror("cd");
    return;
  }
}

static void do_pwd() {
  char *buf = malloc(CWD_SIZE_MAX);
  printf("%s\n", getcwd(buf, CWD_SIZE_MAX));
  free(buf);
}

int execute_builtin(char **argv) {
  int pid, jobid;

  if (!strncmp(argv[0], "cd", 3)) {
    do_cd(argv);
    return 0;
  } else if (!strncmp(argv[0], "pwd", 4)) {
    do_pwd();
    return 0;
  } else if (!strncmp(argv[0], "jobs", 5)) {
    show_jobs();
    return 0;
  } else if (!strncmp(argv[0], "fg", 3)) {
    if (!argv[1]) {
      fprintf(stderr, "usage: fg <jobid>\n");
      return 1;
    }
    jobid = strtol(argv[1], NULL, 10);
    pid = find_pid_from_jobid(jobid);
    if (pid == -1) {
      fprintf(stderr, "Cannot find specified jobid %d.\n", jobid);
      return 1;
    }
    set_fg(pid);
  } else if (!strncmp(argv[0], "exit", 5)) {
    exit_flag = 1;
    exit(0);
  } else {
    /* Not built-in command */
    return 1;
  }
  return 0;
}
