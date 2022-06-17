#pragma once 

extern int exit_flag;

/* argv */
void show_argv_list(char *argv[]);
int get_command(char *buf, char *argv[]);

/* builtin */
int execute_builtin(char** argv);
