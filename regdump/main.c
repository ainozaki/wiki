#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <ucontext.h>
#include <setjmp.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <ucontext.h>

typedef void entrypoint_fn(void);
entrypoint_fn *image_start;

static __attribute__((always_inline)) inline void dump_regs() {
	  unsigned long long regs[31] = {0};
		__asm__("mov %0, x1\n\t"
						"mov %1, X2\n\t"
						"mov %2, X3\n\t"
						:"=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]));
		for (int i = 0; i < 32; i++){
			printf("X%2d:0x%016llx\t", i, regs[i]);
			if (i % 4 == 3){
				printf("\n");
			}
		}
		return;
}

void load_image(const char *imgfile) {
  struct stat st;
  fprintf(stderr, "loading test image %s...\n", imgfile);
  int fd = open(imgfile, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "failed to open image file %s\n", imgfile);
    exit(1);
  }
  if (fstat(fd, &st) != 0) {
    perror("fstat");
    exit(1);
  }
  size_t len = st.st_size;
  void *addr;

  addr = mmap(0, len, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE, fd, 0);
  if (!addr) {
    perror("mmap");
    exit(1);
  }
  close(fd);
  image_start = addr;
}

int main(int argc, char *argv[]) {
	char *filename;

	if (argc < 2){
		printf("usage: %s filename\n", argv[0]);
		return -1;
	}
	filename = argv[1];

	printf("load_image..\n");
	load_image(filename);
	printf("exec..\n");
	image_start();
	printf("finish\n");
	dump_regs();
}
