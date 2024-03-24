# Define compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -Wall -std=c++11

# Define the name of the executable
TARGET = corpus_stats_c

# Define source files
SOURCES = corpus_stats_c.cc

# Define object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link the target with object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files into object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
