// swift-tools-version: 5.7

import PackageDescription

let package = Package(
    name: "SwiftClassifier",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(
            name: "SwiftClassifier",
            targets: ["SwiftClassifier"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.16.0")
    ],
    targets: [
        .executableTarget(
            name: "SwiftClassifier",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            path: "SwiftClassifier"
        )
    ]
) 