#!/usr/bin/env node

// Load environment variables from .env file
import "dotenv/config";

/**
 * Test script to diagnose transcription issues
 * This script will test the transcription service directly
 */

import { TranscriptionService } from "./services/transcription-service.js";
import { existsSync } from "fs";
import logger from "./utils/logger.js";

async function testTranscription() {
  logger.info("=== Starting Transcription Test ===");

  const transcriptionService = new TranscriptionService();

  try {
    // Check if a test audio file was provided
    const testAudioPath = process.argv[2];
    if (!testAudioPath) {
      logger.error("Please provide a path to an audio file");
      logger.info("Usage: npm run test <path-to-audio-file>");
      logger.info("Example: npm run test ./test-audio.mp3");
      process.exit(1);
    }

    // Check if file exists
    if (!existsSync(testAudioPath)) {
      logger.error({ testAudioPath }, "File not found");
      process.exit(1);
    }

    logger.info({ testAudioPath }, "Testing file");

    // Initialize the transcription service
    logger.info("Initializing OpenAI Whisper API service...");
    await transcriptionService.initialize();
    logger.info("Service initialized successfully");

    // Transcribe
    logger.info("Transcribing audio...");
    const startTime = Date.now();
    const text = await transcriptionService.transcribe(testAudioPath);
    const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

    logger.info({ processingTime: parseFloat(processingTime) }, "Transcription completed");

    // Display results
    logger.info("=== RESULTS ===");
    logger.info({ textLength: text.length }, "Text length");
    logger.info({ transcription: text }, "Transcription");

    if (!text || text.length === 0) {
      logger.warn("Transcription returned empty text!");
      logger.info("Possible causes:");
      logger.info("  1. Audio file is silent or contains no speech");
      logger.info("  2. Audio quality is too poor");
      logger.info("  3. Audio format not supported");
      logger.info("  4. File is corrupted");
      logger.info("  5. OpenAI API key is invalid or missing");
      logger.info("  6. Network/API connection issues");
    } else {
      logger.info("Transcription successful!");
    }

  } catch (error) {
    logger.error(
      {
        error: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : undefined,
      },
      "Test failed with error"
    );
    process.exit(1);
  }
}

testTranscription();
