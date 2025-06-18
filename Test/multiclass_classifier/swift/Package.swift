// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MulticlassClassifierSwift",
    platforms: [
        .macOS(.v12),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .executable(name: "multiclass-classifier", targets: ["MulticlassClassifierSwift"]),
        .library(name: "MulticlassClassifierLib", targets: ["MulticlassClassifierSwift"])
    ],
    dependencies: [
        // For production use with real ONNX Runtime:
        // .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.16.0")
        
        // For Linux compatibility testing, we'll use system linking
    ],
    targets: [
        .executableTarget(
            name: "MulticlassClassifierSwift",
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
            name: "MulticlassClassifierSwiftTests",
            dependencies: ["MulticlassClassifierSwift"]
        )
    ]
) 