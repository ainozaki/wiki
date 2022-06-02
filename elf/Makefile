CC=clang++
CXXFLAGS=-O2 -Wall -I./include
LFLAGS=

MAIN=
SRC=elf.cc
HEADER=

TARGET = parse-elf

all: $(SRC) $(HEADER)
	$(CC) $(CXXFLAGS) $(SRC) $(MAIN) -o $(TARGET)  $(LFLAGS)

run:
	./$(TARGET)

sample: sample.cc
	$(CC) $(CXXFLAGS) -o $@ $^

run-test:
	./$(TEST_TARGET)

format: $(SRC) $(HEADER) $(MAIN)
	clang-format -i $^

clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o

.PHONY: all test run format clean
