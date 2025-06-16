import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:binary_classifier_flutter/main.dart';

void main() {
  testWidgets('Binary classifier app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app starts with the correct title
    expect(find.text('Binary Classification'), findsOneWidget);
    expect(find.text('Enter text to classify'), findsOneWidget);
    expect(find.text('Classify'), findsOneWidget);

    // Test text input
    await tester.enterText(find.byType(TextField), 'This is a great product!');
    await tester.pump();

    // Tap the classify button
    await tester.tap(find.text('Classify'));
    await tester.pump();

    // Wait for the classification to complete
    await tester.pumpAndSettle();

    // Check if result is displayed (should use mock implementation)
    expect(find.textContaining('Classification Result:'), findsOneWidget);
    expect(find.textContaining('Probability:'), findsOneWidget);
    expect(find.textContaining('Class:'), findsOneWidget);
  });

  testWidgets('Test mock classification logic', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Test positive sentiment
    await tester.enterText(find.byType(TextField), 'This is amazing and wonderful!');
    await tester.pump();
    await tester.tap(find.text('Classify'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Positive'), findsOneWidget);

    // Test negative sentiment
    await tester.enterText(find.byType(TextField), 'This is terrible and awful!');
    await tester.pump();
    await tester.tap(find.text('Classify'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Negative'), findsOneWidget);
  });
} 