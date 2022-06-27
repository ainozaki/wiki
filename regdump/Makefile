BINARY=sample.bin

all:
	aarch64-linux-gnu-gcc --static main.c
run:
	qemu-aarch64 -L aarch64-linux-gnu- ./a.out ${BINARY}

binary:
	./extract_text.sh sample.s sample.bin
