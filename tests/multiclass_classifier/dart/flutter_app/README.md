# Multiclass News Classifier Flutter App

A Flutter application that classifies news articles into 10 different categories using ONNX Runtime.

## Features

- **📱 Cross-Platform**: Runs on mobile, web, and desktop
- **🎯 10 News Categories**: Politics, Sports, Business, Technology, Entertainment, Health, Science, World, Environment, Education
- **📊 Visual Results**: Color-coded categories with confidence scores and progress bars
- **🔄 Fallback Logic**: Mock classification for CI/testing environments
- **⚡ Real-time**: Instant classification with ONNX Runtime

## News Categories

| Category | Color | Example Keywords |
|----------|-------|------------------|
| Politics | Red | government, election, president, congress |
| Sports | Green | game, team, player, championship |
| Business | Blue | company, market, stock, economy |
| Technology | Purple | software, computer, AI, innovation |
| Entertainment | Orange | movie, music, celebrity, film |
| Health | Pink | medical, doctor, hospital, treatment |
| Science | Teal | research, study, scientist, discovery |
| World | Indigo | international, global, foreign |
| Environment | Light Green | climate, pollution, conservation |
| Education | Amber | school, university, academic |

## Getting Started

### Prerequisites

- Flutter SDK 3.16.0+
- Dart SDK 3.0.0+

### Installation

1. **Navigate to the app directory**:
   ```bash
   cd tests/multiclass_classifier/dart/flutter_app
   ```

2. **Install dependencies**:
   ```bash
   flutter pub get
   ```

3. **Run the app**:
   ```bash
   # For web
   flutter run -d chrome
   
   # For mobile (with device connected)
   flutter run
   
   # For desktop
   flutter run -d windows  # or macos/linux
   ```

### Building for Production

```bash
# Web build
flutter build web --release

# Android APK
flutter build apk --release

# iOS (on macOS)
flutter build ios --release
```

## Model Files

The app requires these model files in `assets/models/`:

- **`model.onnx`**: Trained multiclass classification model
- **`vocab.json`**: Vocabulary mapping and IDF values
- **`scaler.json`**: Feature scaling parameters

## Usage

1. **Enter News Text**: Type or paste a news article in the text field
2. **Classify**: Tap the "Classify News" button
3. **View Results**: See all 10 categories ranked by confidence with visual indicators

### Example Inputs

- **Politics**: "The president announced new economic policies during today's press conference."
- **Sports**: "The basketball team secured their championship victory with a final score of 98-87."
- **Technology**: "New AI software breakthrough promises to revolutionize computer vision applications."

## Architecture

```
lib/
├── main.dart              # Main app with classification logic
assets/models/
├── model.onnx            # ONNX model file
├── vocab.json            # Vocabulary and IDF values
└── scaler.json           # Feature scaling parameters
web/
├── index.html            # Web app entry point
├── manifest.json         # PWA configuration
└── icons/                # App icons
test/
└── widget_test.dart      # Comprehensive widget tests
```

## Error Handling

The app includes robust error handling:

- **ONNX Runtime Errors**: Falls back to mock classification
- **Missing Models**: Uses keyword-based classification
- **Network Issues**: Graceful degradation
- **Invalid Input**: User-friendly error messages

## Testing

Run the comprehensive test suite:

```bash
flutter test
```

Tests cover:
- ✅ UI components and layout
- ✅ Classification functionality
- ✅ Mock implementation logic
- ✅ Error handling
- ✅ Loading states

## Development

### Mock Implementation

For CI/testing environments, the app uses intelligent keyword-based classification:

```dart
final keywordMap = {
  'Politics': ['government', 'election', 'president'],
  'Sports': ['game', 'team', 'player'],
  'Technology': ['software', 'AI', 'innovation'],
  // ... more categories
};
```

### Adding New Categories

1. Update the `newsCategories` list
2. Add keywords to `keywordMap` in mock implementation
3. Add color mapping in `_getCategoryColor()`
4. Update tests and documentation

## Performance

- **Model Loading**: One-time initialization
- **Inference Speed**: ~50-100ms per classification
- **Memory Usage**: Optimized for mobile devices
- **Web Performance**: Progressive loading with service worker

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the ONNX Runtime testing suite. 