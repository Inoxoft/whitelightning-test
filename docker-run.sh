#!/bin/bash

# ONNX Model Tester Docker Helper Script
# =====================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show usage
show_usage() {
    echo "ONNX Model Tester Docker Helper"
    echo "==============================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image"
    echo "  run            Run the container interactively"
    echo "  test MODEL     Test a specific model folder"
    echo "  shell          Open a shell in the container"
    echo "  stop           Stop and remove the container"
    echo "  logs           Show container logs"
    echo "  clean          Remove container and image"
    echo ""
    echo "Examples:"
    echo "  $0 build                           # Build the image"
    echo "  $0 run                            # Start container"
    echo "  $0 test /app/models/sentiment     # Test a model"
    echo "  $0 shell                          # Open shell"
    echo ""
}

# Function to build Docker image
build_image() {
    print_info "Building ONNX Model Tester Docker image..."
    docker build -t onnx-model-tester .
    print_success "Docker image built successfully!"
}

# Function to run container
run_container() {
    print_info "Starting ONNX Model Tester container..."
    docker-compose up -d onnx-tester
    print_success "Container started! Use '$0 shell' to access it."
}

# Function to test a model
test_model() {
    local model_path="$1"
    if [ -z "$model_path" ]; then
        print_error "Please specify a model path"
        echo "Usage: $0 test /app/models/your-model-folder"
        exit 1
    fi
    
    print_info "Testing model: $model_path"
    docker exec -it onnx-model-tester python -c "
from onnx_model_tester import ONNXModelTester
import os

model_folder = '$model_path'
if not os.path.exists(model_folder):
    print('❌ Model folder not found:', model_folder)
    exit(1)

tester = ONNXModelTester(
    model_path=os.path.join(model_folder, 'model.onnx'),
    config_dir=model_folder
)

print('🧪 Running comprehensive test...')
results = tester.run_comprehensive_test(use_llm_text=True)
report = tester.generate_deployment_report()

print(f'📊 Score: {report[\"score\"]}/{report[\"max_score\"]} ({report[\"percentage\"]:.1f}%)')
print(f'Status: {report[\"deployment_status\"]}')
print(f'Ready: {\"✅ Yes\" if report[\"is_ready\"] else \"❌ No\"}')
"
}

# Function to open shell
open_shell() {
    print_info "Opening shell in ONNX Model Tester container..."
    docker exec -it onnx-model-tester /bin/bash
}

# Function to stop container
stop_container() {
    print_info "Stopping ONNX Model Tester container..."
    docker-compose down
    print_success "Container stopped!"
}

# Function to show logs
show_logs() {
    print_info "Showing container logs..."
    docker logs onnx-model-tester
}

# Function to clean up
clean_up() {
    print_warning "This will remove the container and image. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        docker-compose down
        docker rmi onnx-model-tester 2>/dev/null || true
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Main script logic
case "${1:-}" in
    "build")
        build_image
        ;;
    "run")
        run_container
        ;;
    "test")
        test_model "$2"
        ;;
    "shell")
        open_shell
        ;;
    "stop")
        stop_container
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        clean_up
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 