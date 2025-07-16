// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "SwiftClassifier",
    platforms: [
        .macOS(.v10_15),
        .iOS(.v13)
    ],
    products: [
        .executable(
            name: "SwiftClassifier",
            targets: ["SwiftClassifier"]
        )
    ],
    dependencies: [
        // Add ONNX Runtime dependencies here when available
        // .package(url: "https://github.com/microsoft/onnxruntime-swift", from: "1.16.0")
    ],
    targets: [
        .executableTarget(
            name: "SwiftClassifier",
            dependencies: [],
            path: "Sources/SwiftClassifier"
        )
    ]
) 