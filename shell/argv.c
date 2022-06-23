#include "def.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int get_command(char *buf, char *argv[]) {
  ssize_t readlen;
  size_t len;
  char *token;
  int index = 0;

  /* get input */
  if ((readlen = getline(&buf, &len, stdin))  == -1){
    perror("readlen\n");
    return -1;
	}

  /* parse */
  const char *delimiter = " \n";
  token = strtok(buf, delimiter);
  if (!token) {
    return -1;
  }
  while (token) {
    argv[index++] = token;
    token = strtok(NULL, delimiter);
  }
  /* Add NULL at the end of argv list */
  argv[index] = NULL;

  // show_argv_list(argv);
  return 0;
}

void show_argv_list(char *argv[]) {
  int n = 0;
  while (argv[n]) {
    printf("%2d: %s\n", n, argv[n]);
    n++;
  }
}

int get_argc(char *argv[]) {
  int n = 0;
  while (argv[n]) {
    n++;
    continue;
  }
  return n;
}

int search_argv(char *argv[], char *arg) {
	int i;
  for (i = 0; i < get_argc(argv); i++) {
    if (strcmp(argv[i], arg) == 0) {
      return i;
    }
  }
  return -1;
}

void delete_argv(char *argv[], int pos, int size) {
	int i;
  int argc = get_argc(argv);
  for (i = pos + size; i < argc; i++) {
    argv[i - size] = argv[i];
  }
  argv[argc - size] = NULL;
}

void extract_argv(char *dst[], char *src[], int pos, int size) {
	int i;
  for (i = 0; i < size; i++) {
    dst[i] = src[pos + i];
  }
  dst[size] = NULL;
}

void dump_argv(char *name, char **argv) {
  printf("---------------\n");
  printf("%s\n", name);
	int i;
  for (i = 0; i < get_argc(argv); i++) {
    printf("%s[%d] =  %s\n", name, i, argv[i]);
  }
  printf("---------------\n");
}

void free_argv(char **argv) {
	int i;
  if (argv) {
    for (i = 0; argv[i]; i++) {
      printf("free %s\n", argv[i]);
      free(argv[i]);
    }
    free(argv);
  }
}
