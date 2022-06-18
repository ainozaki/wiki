TARGET = mysh
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = 

SRC = argv.c \
			 builtin.c \
			 exec.c \
			 job.c \
			 main.c \
			 redirect.c \
			 signal.c \
			 wait.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run:
	./$(TARGET)

format:
	clang-format -i $(MAIN) $(SRCS) $(HEADERS)

clean:
	rm -f *.o $(TARGET)

.PHONY: format clean
