package com.whitelightning;

import ai.onnxruntime.*;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class SpamDetectorTest {
    private static final Logger logger = LoggerFactory.getLogger(SpamDetectorTest.class);
    private static OrtSession session;
    private static OrtEnvironment env;
    private static final String MODEL_PATH = "../../../models/spam_detector/model.onnx";
    
    @BeforeAll
    public static void setup() throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(MODEL_PATH, new OrtSession.SessionOptions());
        logger.info("ONNX Runtime version: {}", env.getVersion());
    }

    @Test
    @DisplayName("Test model loading and basic inference")
    public void testModelLoading() throws OrtException {
        assertNotNull(session, "Session should not be null");
        assertNotNull(env, "Environment should not be null");
        
        // Get model metadata
        NodeInfo inputInfo = session.getInputInfo().values().iterator().next();
        NodeInfo outputInfo = session.getOutputInfo().values().iterator().next();
        
        logger.info("Input name: {}", inputInfo.getName());
        logger.info("Input type: {}", inputInfo.getType());
        logger.info("Output name: {}", outputInfo.getName());
        logger.info("Output type: {}", outputInfo.getType());
    }

    @Test
    @DisplayName("Test spam detection on sample texts")
    public void testSpamDetection() throws OrtException {
        List<String> testTexts = Arrays.asList(
            "Buy now! Limited time offer!",
            "Hello, how are you doing today?",
            "URGENT: Your account needs verification",
            "Meeting at 2 PM in the conference room"
        );

        for (String text : testTexts) {
            long startTime = System.nanoTime();
            
            // Prepare input tensor
            OnnxTensor inputTensor = createInputTensor(text);
            
            // Run inference
            OrtSession.Result result = session.run(Map.of("input", inputTensor));
            
            // Get prediction
            float[] output = ((OnnxTensor) result.get(0)).getFloatBuffer().array();
            boolean isSpam = output[0] > 0.5;
            
            long endTime = System.nanoTime();
            double inferenceTime = (endTime - startTime) / 1_000_000.0; // Convert to milliseconds
            
            logger.info("Text: {}", text);
            logger.info("Spam probability: {}", output[0]);
            logger.info("Is spam: {}", isSpam);
            logger.info("Inference time: {} ms", inferenceTime);
            
            // Basic assertions
            assertTrue(output[0] >= 0 && output[0] <= 1, "Probability should be between 0 and 1");
        }
    }

    @Test
    @DisplayName("Test performance metrics")
    public void testPerformance() throws OrtException {
        int numIterations = 100;
        long totalTime = 0;
        long maxMemory = 0;
        
        Runtime runtime = Runtime.getRuntime();
        
        for (int i = 0; i < numIterations; i++) {
            String text = "Sample text for performance testing " + i;
            
            // Measure memory before
            long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
            
            // Run inference
            long startTime = System.nanoTime();
            OnnxTensor inputTensor = createInputTensor(text);
            session.run(Map.of("input", inputTensor));
            long endTime = System.nanoTime();
            
            // Measure memory after
            long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
            maxMemory = Math.max(maxMemory, memoryAfter - memoryBefore);
            
            totalTime += (endTime - startTime);
        }
        
        double avgInferenceTime = totalTime / (numIterations * 1_000_000.0); // Convert to milliseconds
        double maxMemoryMB = maxMemory / (1024.0 * 1024.0); // Convert to MB
        
        logger.info("Average inference time: {} ms", avgInferenceTime);
        logger.info("Maximum memory usage: {} MB", maxMemoryMB);
        
        // Performance assertions
        assertTrue(avgInferenceTime < 100, "Average inference time should be less than 100ms");
        assertTrue(maxMemoryMB < 500, "Maximum memory usage should be less than 500MB");
    }

    private OnnxTensor createInputTensor(String text) throws OrtException {
        // Convert text to input tensor format
        // This is a placeholder - you'll need to implement the actual text preprocessing
        // based on your model's requirements (tokenization, padding, etc.)
        float[] inputData = new float[512]; // Adjust size based on your model
        Arrays.fill(inputData, 0.0f);
        
        long[] shape = new long[]{1, 512}; // Adjust shape based on your model
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
    }
} 