#!/usr/bin/env node

// Load environment variables from .env file
import "dotenv/config";

/**
 * Test script to diagnose transcription issues
 * This script will test the transcription service directly
 */

import { TranscriptionService } from "./services/transcription-service.js";
import { existsSync } from "fs";

async function testTranscription() {
  console.log("=== Starting Transcription Test ===\n");

  const transcriptionService = new TranscriptionService();

  try {
    // Check if a test audio file was provided
    const testAudioPath = process.argv[2];
    if (!testAudioPath) {
      console.error("❌ Error: Please provide a path to an audio file");
      console.log("\nUsage: npm run test <path-to-audio-file>");
      console.log("Example: npm run test ./test-audio.mp3\n");
      process.exit(1);
    }

    // Check if file exists
    if (!existsSync(testAudioPath)) {
      console.error(`❌ Error: File not found: ${testAudioPath}`);
      process.exit(1);
    }

    console.log(`Testing file: ${testAudioPath}\n`);

    // Initialize the transcription service
    console.log("1. Initializing OpenAI Whisper API service...");
    await transcriptionService.initialize();
    console.log("✓ Service initialized successfully\n");

    // Transcribe
    console.log("2. Transcribing audio...");
    const startTime = Date.now();
    const text = await transcriptionService.transcribe(testAudioPath);
    const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

    console.log(`✓ Transcription completed in ${processingTime}s\n`);

    // Display results
    console.log("=== RESULTS ===");
    console.log(`Text length: ${text.length} characters`);
    console.log(`Transcription: "${text}"\n`);

    if (!text || text.length === 0) {
      console.log("⚠️  WARNING: Transcription returned empty text!");
      console.log("\nPossible causes:");
      console.log("  1. Audio file is silent or contains no speech");
      console.log("  2. Audio quality is too poor");
      console.log("  3. Audio format not supported");
      console.log("  4. File is corrupted");
      console.log("  5. OpenAI API key is invalid or missing");
      console.log("  6. Network/API connection issues");
    } else {
      console.log("✓ Transcription successful!");
    }

  } catch (error) {
    console.error("\n❌ Test failed with error:");
    console.error(error);
    process.exit(1);
  }
}

testTranscription();
