FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ONNX model tester from model_test folder
COPY model_test/onnx_model_tester.py .

# Copy the entrypoint script
COPY docker-entrypoint.sh .

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Make scripts executable
RUN chmod +x onnx_model_tester.py docker-entrypoint.sh

# Set the entrypoint to our wrapper script
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command shows help
CMD ["--help"] 