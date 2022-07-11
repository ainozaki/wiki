CFLAGS=-O2 -Wall -Wextra -I./include -I../common
LDFLAGS=

SRC = main.c \
			../common/dhcph.c
OBJ=$(SRC:.c=.o)
DEP=$(SRC:.c=.d)

TARGET = mydhcpc

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ -MMD -MP -MF $(@:.o=.d)

-include $(DEP)

run:
	./$(TARGET) 192.168.64.2

format: $(SRC)
	find ./ -type f -name "*.c" -or -name "*.h" | xargs clang-format -i

clean:
	find ../ -type f -name "*.o" -or -name "*.d" -or -name "*.out" | xargs rm -rf
	rm $(TARGET)

.PHONY: all test run run-test format clean

.SECONDARY: $(OBJ)
