// swift-tools-version: 5.7

import PackageDescription

let package = Package(
    name: "SwiftONNXRunner",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(
            name: "SwiftONNXRunner",
            targets: ["SwiftONNXRunner"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.16.0")
    ],
    targets: [
        .executableTarget(
            name: "SwiftONNXRunner",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            path: "SwiftONNXRunner"
        )
    ]
) 