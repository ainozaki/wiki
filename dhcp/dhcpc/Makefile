CFLAGS=-O2 -Wall -Wextra
LDFLAGS=

SRC = main.c
OBJ=$(SRC:.c=.o)
DEP=$(SRC:.c=.d)

TARGET = mydhcpc

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CXXFLAGS) -c $< -o $@ -MMD -MP -MF $(@:.o=.d)

-include $(DEP)

run:
	./$(TARGET)

format: $(SRC)
	find ./ -type f -name "*.c" -or -name "*.h" | xargs clang-format -i

clean:
	find ./ -type f -name "*.o" -or -name "*.d" -or -name "*.out" | xargs rm -rf
	rm $(TARGET)

.PHONY: all test run run-test format clean

.SECONDARY: $(OBJ)
