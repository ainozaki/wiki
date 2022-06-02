#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char *argv[]) {
  int fd, i;
  struct stat sb;
  char *head;
  char type;
  Elf64_Ehdr *ehdr;
  Elf64_Phdr *phdr;
  Elf64_Shdr *shdr, *shstr;

  if (argc < 2) {
    fprintf(stderr, "usage: %s <filename>\n", argv[0]);
    return -1;
  }

  /// open and mmap ELF file
  fd = open(argv[1], O_RDONLY);
  fstat(fd, &sb);
  head = (char *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);

  /// ELF header
  ehdr = (Elf64_Ehdr *)head;
  if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
      ehdr->e_ident[2] != 'L' || ehdr->e_ident[3] != 'F') {
    fprintf(stderr, "%s is not ELF file.\n", argv[1]);
    return -1;
  }

  /// program header
  printf("Segments:\n");
  for (i = 0; i < ehdr->e_phnum; i++) {
    phdr = (Elf64_Phdr *)(head + ehdr->e_phoff + ehdr->e_phentsize * i);
    switch (phdr->p_type) {
    case PT_LOAD:
      type = 'L';
      break;
    case PT_DYNAMIC:
      type = 'D';
      break;
    case PT_PHDR:
      type = 'P';
      break;
    }
    printf("[%2d]", i);
    printf("\ttype:%c", type);
    printf("\tvaddr:%lu", phdr->p_vaddr);
    printf("\toffset:%lu", phdr->p_offset);
    printf("\n");
  }

  /// section header
  printf("Sections:\n");
  shstr = (Elf64_Shdr *)(head + ehdr->e_shoff +
                         ehdr->e_shentsize * ehdr->e_shstrndx);
  for (i = 0; i < ehdr->e_shnum; i++) {
    shdr = (Elf64_Shdr *)(head + ehdr->e_shoff + ehdr->e_shentsize * i);
    printf("[%2d] %20s", i, (char *)(head + shstr->sh_offset + shdr->sh_name));
    printf("\toffset: %lu", shdr->sh_offset);
    printf("\n");
  }

  munmap(head, sb.st_size);
  return 0;
}
