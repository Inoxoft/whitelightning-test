import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import '../lib/main.dart';

void main() {
  testWidgets('Multiclass news classifier app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app starts with the correct title
    expect(find.text('News Classification'), findsOneWidget);
    expect(find.text('Enter news text to classify'), findsOneWidget);
    expect(find.text('Classify News'), findsOneWidget);

    // Test text input
    await tester.enterText(find.byType(TextField), 'The president announced new economic policies today.');
    await tester.pump();

    // Tap the classify button
    await tester.tap(find.text('Classify News'));
    await tester.pump();

    // Wait for the classification to complete
    await tester.pumpAndSettle();

    // Check if result is displayed (should use mock implementation)
    expect(find.text('Classification Results:'), findsOneWidget);
  });

  testWidgets('Test mock classification logic for different categories', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Test Politics classification
    await tester.enterText(find.byType(TextField), 'The government announced new election policies.');
    await tester.pump();
    await tester.tap(find.text('Classify News'));
    await tester.pumpAndSettle();

    expect(find.text('Politics'), findsOneWidget);

    // Test Sports classification
    await tester.enterText(find.byType(TextField), 'The basketball team won the championship game.');
    await tester.pump();
    await tester.tap(find.text('Classify News'));
    await tester.pumpAndSettle();

    expect(find.text('Sports'), findsOneWidget);

    // Test Technology classification
    await tester.enterText(find.byType(TextField), 'New AI software revolutionizes computer technology.');
    await tester.pump();
    await tester.tap(find.text('Classify News'));
    await tester.pumpAndSettle();

    expect(find.text('Technology'), findsOneWidget);
  });

  testWidgets('Test UI components and layout', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Check for text field
    expect(find.byType(TextField), findsOneWidget);
    
    // Check for classify button
    expect(find.byType(ElevatedButton), findsOneWidget);
    
    // Check app bar
    expect(find.byType(AppBar), findsOneWidget);
    
    // Test that text field accepts input
    final textField = find.byType(TextField);
    await tester.enterText(textField, 'Test news article');
    await tester.pump();
    
    expect(find.text('Test news article'), findsOneWidget);
  });

  testWidgets('Test empty input handling', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Try to classify with empty text
    await tester.tap(find.text('Classify News'));
    await tester.pump();

    // Should not show results for empty input
    expect(find.text('Classification Results:'), findsNothing);
  });

  testWidgets('Test loading state', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Enter text and start classification
    await tester.enterText(find.byType(TextField), 'Test news for loading state');
    await tester.pump();
    
    // Tap classify button
    await tester.tap(find.text('Classify News'));
    await tester.pump();

    // Should show loading indicator briefly
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });
} 