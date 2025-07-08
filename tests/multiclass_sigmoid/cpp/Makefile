CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lonnxruntime -pthread
TARGET = test_onnx_model
SOURCES = test_onnx_model.cpp

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean target
clean:
	rm -f $(TARGET)

# Test with custom text
test: $(TARGET)
	./$(TARGET) "I feel so happy today!"

.PHONY: all run clean test 