#include "def.h"

#include <assert.h>
#include <signal.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

void ignore_signal(int signal) {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_handler = SIG_IGN;
  sigaction(signal, &act, NULL);
}

void default_signal(int signal) {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_handler = SIG_DFL;
  sigaction(signal, &act, NULL);
}

void set_handler(int signal, void *handler) {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_handler = handler;
	act.sa_flags = SA_RESTART;
  sigaction(signal, &act, NULL);
}

static void collect_zombie() {
	sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGCHLD);
  sigaddset(&set, SIGTSTP);
  sigaddset(&set, SIGINT);
  sigprocmask(SIG_BLOCK, &set, NULL);
  pid_t pid;
  while ((pid = waitpid((pid_t)-1, NULL, WNOHANG)) > 0) {
    delete_job(pid);
  }
	sigprocmask(SIG_UNBLOCK, &set, NULL);
}

void sigchld_handler(){
	collect_zombie();
}
