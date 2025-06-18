// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BinaryClassifierSwift",
    platforms: [
        .macOS(.v12),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .executable(name: "binary-classifier", targets: ["BinaryClassifierSwift"]),
        .library(name: "BinaryClassifierLib", targets: ["BinaryClassifierSwift"])
    ],
    dependencies: [
        // For production use with real ONNX Runtime:
        // .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.16.0")
        
        // For Linux compatibility testing, we'll use system linking
    ],
    targets: [
        .executableTarget(
            name: "BinaryClassifierSwift",
            dependencies: [
                // For production: .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            path: "Sources",
            linkerSettings: [
                // Linux system linking (comment out for production)
                .linkedLibrary("onnxruntime", .when(platforms: [.linux])),
                .unsafeFlags(["-L/usr/local/lib"], .when(platforms: [.linux]))
            ]
        ),
        .testTarget(
            name: "BinaryClassifierSwiftTests",
            dependencies: ["BinaryClassifierSwift"]
        )
    ]
) 