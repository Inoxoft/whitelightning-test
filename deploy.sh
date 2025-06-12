#!/bin/bash

# ONNX Model Tester Deployment Script
# ===================================
# Supports both local and remote Docker deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="onnx-model-tester"
REGISTRY_URL="${DOCKER_REGISTRY:-}"  # Set this env var for remote registry
VERSION="${VERSION:-latest}"

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

print_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

# Function to show usage
show_usage() {
    echo "ONNX Model Tester Deployment Script"
    echo "===================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Local Commands:"
    echo "  local-build        Build image locally"
    echo "  local-run          Run container locally"
    echo "  local-stop         Stop local container"
    echo "  local-shell        Open shell in local container"
    echo ""
    echo "Remote Commands:"
    echo "  remote-build       Build image for remote deployment"
    echo "  remote-push        Push image to registry"
    echo "  remote-deploy      Deploy to remote server"
    echo "  remote-stop        Stop remote container"
    echo "  remote-logs        Show remote container logs"
    echo ""
    echo "Utility Commands:"
    echo "  build-all          Build both local and remote images"
    echo "  clean              Clean up all containers and images"
    echo "  status             Show status of all containers"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_REGISTRY    Registry URL for remote deployment"
    echo "  VERSION           Image version tag (default: latest)"
    echo "  REMOTE_HOST       Remote server hostname/IP"
    echo "  REMOTE_USER       Remote server username"
    echo ""
    echo "Examples:"
    echo "  $0 local-build                    # Build for local use"
    echo "  $0 remote-build                   # Build for remote deployment"
    echo "  DOCKER_REGISTRY=myregistry.com $0 remote-push"
    echo "  REMOTE_HOST=server.com $0 remote-deploy"
    echo ""
}

# Local deployment functions
local_build() {
    print_header "Building Local Docker Image"
    docker build -t ${IMAGE_NAME}:${VERSION} .
    print_success "Local image built successfully!"
}

local_run() {
    print_header "Starting Local Container"
    docker-compose up -d onnx-tester
    print_success "Local container started!"
    print_info "Use './deploy.sh local-shell' to access the container"
}

local_stop() {
    print_header "Stopping Local Container"
    docker-compose down
    print_success "Local container stopped!"
}

local_shell() {
    print_header "Opening Shell in Local Container"
    docker exec -it onnx-model-tester /bin/bash
}

# Remote deployment functions
remote_build() {
    print_header "Building Remote Docker Image"
    docker build -f Dockerfile.remote -t ${IMAGE_NAME}:${VERSION} .
    
    if [ -n "$REGISTRY_URL" ]; then
        docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY_URL}/${IMAGE_NAME}:${VERSION}
        print_success "Remote image built and tagged for registry!"
    else
        print_success "Remote image built!"
        print_warning "Set DOCKER_REGISTRY env var to tag for registry"
    fi
}

remote_push() {
    if [ -z "$REGISTRY_URL" ]; then
        print_error "DOCKER_REGISTRY environment variable is required for push"
        exit 1
    fi
    
    print_header "Pushing Image to Registry"
    docker push ${REGISTRY_URL}/${IMAGE_NAME}:${VERSION}
    print_success "Image pushed to registry!"
}

remote_deploy() {
    if [ -z "$REMOTE_HOST" ]; then
        print_error "REMOTE_HOST environment variable is required"
        exit 1
    fi
    
    REMOTE_USER=${REMOTE_USER:-root}
    
    print_header "Deploying to Remote Server: $REMOTE_HOST"
    
    # Copy docker-compose file to remote server
    scp docker-compose.remote.yml ${REMOTE_USER}@${REMOTE_HOST}:/tmp/
    scp .env ${REMOTE_USER}@${REMOTE_HOST}:/tmp/ 2>/dev/null || print_warning ".env file not found, skipping"
    
    # Deploy on remote server
    ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
        cd /tmp
        
        # Pull latest image if using registry
        if [ -n "$REGISTRY_URL" ]; then
            docker pull ${REGISTRY_URL}/${IMAGE_NAME}:${VERSION}
        fi
        
        # Stop existing container
        docker-compose -f docker-compose.remote.yml down 2>/dev/null || true
        
        # Start new container
        docker-compose -f docker-compose.remote.yml up -d
        
        echo "Remote deployment completed!"
EOF
    
    print_success "Remote deployment completed!"
}

remote_stop() {
    if [ -z "$REMOTE_HOST" ]; then
        print_error "REMOTE_HOST environment variable is required"
        exit 1
    fi
    
    REMOTE_USER=${REMOTE_USER:-root}
    
    print_header "Stopping Remote Container"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "docker-compose -f /tmp/docker-compose.remote.yml down"
    print_success "Remote container stopped!"
}

remote_logs() {
    if [ -z "$REMOTE_HOST" ]; then
        print_error "REMOTE_HOST environment variable is required"
        exit 1
    fi
    
    REMOTE_USER=${REMOTE_USER:-root}
    
    print_header "Showing Remote Container Logs"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "docker logs onnx-model-tester-remote"
}

# Utility functions
build_all() {
    print_header "Building All Images"
    local_build
    remote_build
    print_success "All images built successfully!"
}

clean_all() {
    print_warning "This will remove all containers and images. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_header "Cleaning Up"
        
        # Stop and remove containers
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.remote.yml down 2>/dev/null || true
        
        # Remove images
        docker rmi ${IMAGE_NAME}:${VERSION} 2>/dev/null || true
        if [ -n "$REGISTRY_URL" ]; then
            docker rmi ${REGISTRY_URL}/${IMAGE_NAME}:${VERSION} 2>/dev/null || true
        fi
        
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

show_status() {
    print_header "Container Status"
    
    echo "Local Containers:"
    docker ps -a --filter "name=onnx-model-tester" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true
    
    echo ""
    echo "Images:"
    docker images --filter "reference=${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" || true
    
    if [ -n "$REMOTE_HOST" ]; then
        echo ""
        echo "Remote Status (${REMOTE_HOST}):"
        REMOTE_USER=${REMOTE_USER:-root}
        ssh ${REMOTE_USER}@${REMOTE_HOST} "docker ps --filter 'name=onnx-model-tester-remote' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || print_warning "Cannot connect to remote host"
    fi
}

# Main script logic
case "${1:-}" in
    "local-build")
        local_build
        ;;
    "local-run")
        local_run
        ;;
    "local-stop")
        local_stop
        ;;
    "local-shell")
        local_shell
        ;;
    "remote-build")
        remote_build
        ;;
    "remote-push")
        remote_push
        ;;
    "remote-deploy")
        remote_deploy
        ;;
    "remote-stop")
        remote_stop
        ;;
    "remote-logs")
        remote_logs
        ;;
    "build-all")
        build_all
        ;;
    "clean")
        clean_all
        ;;
    "status")
        show_status
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