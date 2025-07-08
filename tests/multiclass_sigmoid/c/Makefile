CC = gcc
CFLAGS = -std=c99 -O2 -Wall -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lonnxruntime -lm
TARGET = test_onnx_model
SOURCES = test_onnx_model.c

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

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