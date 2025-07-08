// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "SwiftONNXRunner",
    platforms: [
        .macOS(.v10_15),
        .iOS(.v13)
    ],
    products: [
        .executable(
            name: "SwiftONNXRunner",
            targets: ["SwiftONNXRunner"]
        )
    ],
    dependencies: [
        // Add ONNX Runtime dependencies here when available
        // .package(url: "https://github.com/microsoft/onnxruntime-swift", from: "1.16.0")
    ],
    targets: [
        .executableTarget(
            name: "SwiftONNXRunner",
            dependencies: []
        )
    ]
) 