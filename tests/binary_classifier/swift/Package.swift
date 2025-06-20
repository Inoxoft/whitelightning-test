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
        .executable(name: "binary-classifier", targets: ["BinaryClassifierApp"]),
        .library(name: "BinaryClassifierLib", targets: ["BinaryClassifierLib"])
    ],
    dependencies: [
        // Using system installed onnxruntime-objc via CocoaPods for GitHub Actions
    ],
    targets: [
        .target(
            name: "BinaryClassifierLib",
            dependencies: [
                // Dependencies will be linked via CocoaPods in GitHub Actions
            ],
            path: "Sources/BinaryClassifierLib",
            linkerSettings: [
                // For GitHub Actions with CocoaPods
                .linkedFramework("onnxruntime_objc", .when(platforms: [.macOS])),
                .unsafeFlags(["-F/Users/runner/work/_temp/Pods/onnxruntime-objc"], .when(platforms: [.macOS]))
            ]
        ),
        .executableTarget(
            name: "BinaryClassifierApp",
            dependencies: ["BinaryClassifierLib"],
            path: "Sources/BinaryClassifierApp"
        ),
        .testTarget(
            name: "BinaryClassifierSwiftTests",
            dependencies: ["BinaryClassifierLib"],
            path: "Tests"
        )
    ]
) 