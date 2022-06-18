#pragma once 

extern int exit_flag;

#define PATH_TOKEN_MAX  32
#define COMMAND_LEN_MAX 64
#define PATH_LEN_MAX    256

/* argv */
void show_argv_list(char *argv[]);
int get_command(char *buf, char *argv[]);
int get_argc(char* argv[]);
int search_argv(char* argv[], char* arg);
void delete_argv(char* argv[], int pos, int size);
void extract_argv(char* dst[], char* src[], int pos, int size);
void dump_argv(char* name, char** argv);
void free_argv(char** argv);

/* builtin */
int execute_builtin(char** argv);

/* exec */
int tursh_exec(char** argv);

/* process */
void wait_child(int pid);

/* job */
enum job_state { JOB_RUNNING, JOB_STOPPED };
struct job {
  int jobid;
  int pid;
  int pgid;
  enum job_state state;
  char* command;
  struct job* next;
  struct job* prev;
};
extern struct job* jobs;

void init_jobs();
void add_job(int pid, int pgid, char** command);
int delete_job(int pid);
int stop_job(int pid);
void show_jobs();
int find_pid_from_jobid(int jobid);
struct job* find_job_from_jobid(int jobid);
void set_fg(int pgid);

/* signal */
void ignore_signal(int signal);
void default_signal(int signal);
void set_handler(int signal, void *handler);
void sigchld_handler();

/* redirect */
void do_redirect(char* argv[]);
