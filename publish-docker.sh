#!/bin/bash

# Publish Docker Image to GitHub Container Registry
# =================================================

set -e

# Configuration
ORGANIZATION="whitelightning-ai"  # Replace with your GitHub organization
IMAGE_NAME="whitelightning"
VERSION="${VERSION:-latest}"
REGISTRY="ghcr.io"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if logged in to GitHub Container Registry
check_login() {
    print_info "Checking GitHub Container Registry login..."
    if ! docker info | grep -q "ghcr.io"; then
        print_warning "Please login to GitHub Container Registry first:"
        echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
        echo ""
        echo "Or create a Personal Access Token with 'write:packages' permission:"
        echo "  https://github.com/settings/tokens"
        exit 1
    fi
    print_success "Already logged in to GitHub Container Registry"
}

# Build the image
build_image() {
    print_info "Building Docker image..."
    docker build -t ${IMAGE_NAME}:${VERSION} .
    print_success "Image built successfully"
}

# Tag for registry
tag_image() {
    print_info "Tagging image for registry..."
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:${VERSION}
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:latest
    print_success "Image tagged for registry"
}

# Push to registry
push_image() {
    print_info "Pushing to GitHub Container Registry..."
    docker push ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:${VERSION}
    docker push ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:latest
    print_success "Image pushed successfully"
}

# Show usage instructions
show_usage() {
    echo ""
    echo "🎉 Image published successfully!"
    echo "================================"
    echo ""
    echo "Your image is now available at:"
    echo "  ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:${VERSION}"
    echo "  ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:latest"
    echo ""
    echo "Usage example:"
    echo "  docker run --rm \\"
    echo "    -v \$(pwd):/app/models \\"
    echo "    -e OPENROUTER_API_KEY=\"YOUR_API_KEY\" \\"
    echo "    ${REGISTRY}/${ORGANIZATION}/${IMAGE_NAME}:latest \\"
    echo "    -p=\"Classify customer feedback as positive or negative sentiment\" \\"
    echo "    --refinement-cycles=1 \\"
    echo "    --generate-edge-cases=true \\"
    echo "    --lang=english"
    echo ""
    echo "To make it public, go to:"
    echo "  https://github.com/orgs/${ORGANIZATION}/packages/container/${IMAGE_NAME}/settings"
    echo "  And change visibility to 'Public'"
    echo ""
}

# Main execution
main() {
    echo "🐳 Publishing ONNX Model Tester to GitHub Container Registry"
    echo "============================================================"
    echo "Organization: ${ORGANIZATION}"
    echo "Image: ${IMAGE_NAME}"
    echo "Version: ${VERSION}"
    echo "Registry: ${REGISTRY}"
    echo ""
    
    check_login
    build_image
    tag_image
    push_image
    show_usage
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 