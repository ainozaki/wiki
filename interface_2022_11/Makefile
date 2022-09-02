OUTDIR	= ./build
TARGET	= $(OUTDIR)/curo
SOURCES	= $(wildcard *.cpp)
OBJECTS	= $(addprefix $(OUTDIR)/, $(SOURCES:.cpp=.o))

.PHONY: all

all: $(TARGET)
run: $(TARGET)
	./$<
format:
	find ./ -type f -name "*.cpp" | xargs clang-format -i $<

$(TARGET): $(OBJECTS) Makefile
	$(CXX) -o $(TARGET) $(OBJECTS)

$(OUTDIR)/%.o: %.cpp Makefile
	mkdir -p build
	$(CXX) -o $@ -c $<
