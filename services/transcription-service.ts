import { nodewhisper } from "nodejs-whisper";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { unlink } from "fs/promises";

/**
 * Service for managing Whisper transcription using nodejs-whisper
 * This uses the official OpenAI Whisper C++ implementation
 */
export class TranscriptionService {
  private modelName: string = "base"; // base model for good speed/accuracy balance
  private isInitialized: boolean = false;

  /**
   * Initializes the Whisper model (downloads if needed)
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    console.log(`Initializing Whisper ${this.modelName} model...`);
    console.log("Note: First run will download the model (~150MB for base)");
    
    // nodejs-whisper will auto-download the model on first use
    this.isInitialized = true;
    console.log("Whisper service initialized");
  }

  /**
   * Transcribes audio file to text
   * @param filePath - Absolute path to the audio file
   * @returns Transcribed text
   */
  async transcribe(filePath: string): Promise<string> {
    await this.initialize();

    try {
      console.log(`Transcribing file: ${filePath}`);
      
      // Check audio duration first
      const duration = await this.getAudioDuration(filePath);
      console.log(`Audio duration: ${duration} seconds`);
      
      // If audio is longer than 10 minutes, split into chunks and process in parallel
      if (duration > 600) {
        console.log("Large file detected, processing in parallel chunks...");
        return await this.transcribeInChunks(filePath, duration);
      }
      
      // For smaller files, process normally
      return await this.transcribeSingle(filePath);
    } catch (error) {
      console.error("Transcription error:", error);
      throw new Error(
        `Failed to transcribe: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  /**
   * Get audio duration in seconds
   */
  private async getAudioDuration(filePath: string): Promise<number> {
    const execAsync = promisify(exec);
    
    try {
      const { stdout } = await execAsync(
        `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`
      );
      return parseFloat(stdout.trim());
    } catch (error) {
      console.warn("Could not determine audio duration, assuming short file");
      return 0;
    }
  }

  /**
   * Transcribe large file by splitting into chunks and processing in parallel
   */
  private async transcribeInChunks(filePath: string, duration: number): Promise<string> {
    const execAsync = promisify(exec);
    
    // Split into 5-minute chunks
    const chunkDuration = 300; // 5 minutes
    const numChunks = Math.ceil(duration / chunkDuration);
    
    console.log(`Splitting into ${numChunks} chunks of ${chunkDuration}s each`);
    
    // Create chunk files
    const chunkFiles: string[] = [];
    for (let i = 0; i < numChunks; i++) {
      const startTime = i * chunkDuration;
      const chunkFile = filePath.replace(/(\.[^.]+)$/, `_chunk${i}$1`);
      chunkFiles.push(chunkFile);
      
      // Extract chunk using ffmpeg
      await execAsync(
        `ffmpeg -i "${filePath}" -ss ${startTime} -t ${chunkDuration} -c copy "${chunkFile}"`
      );
    }
    
    console.log(`Created ${chunkFiles.length} chunk files, processing in parallel...`);
    
    // Process chunks in parallel (max 4 at a time to avoid overwhelming the system)
    const maxConcurrent = 4;
    const results: string[] = [];
    
    for (let i = 0; i < chunkFiles.length; i += maxConcurrent) {
      const batch = chunkFiles.slice(i, i + maxConcurrent);
      const batchResults = await Promise.all(
        batch.map(chunkFile => this.transcribeSingle(chunkFile))
      );
      results.push(...batchResults);
      
      console.log(`Processed chunks ${i + 1}-${Math.min(i + maxConcurrent, chunkFiles.length)} of ${chunkFiles.length}`);
    }
    
    // Clean up chunk files
    await Promise.all(chunkFiles.map(file => unlink(file).catch(() => {})));
    
    // Combine results
    const combinedText = results.filter(r => r.length > 0).join(" ");
    console.log(`Combined transcription: ${combinedText.length} characters`);
    
    return combinedText;
  }

  /**
   * Transcribe a single file (or chunk)
   */
  private async transcribeSingle(filePath: string): Promise<string> {
    // Configure transcription options
    const options: any = {
      modelName: this.modelName,
      autoDownloadModelName: this.modelName,
      whisperOptions: {
        language: "hi",  // Force Hindi - prevents auto-translation to English
        translateToEnglish: false,  // Double insurance
      },
    };

    console.log("Whisper options:", JSON.stringify(options, null, 2));

    // Transcribe the audio file
    const output = await nodewhisper(filePath, options);
    
    // nodejs-whisper returns the text directly
    let text = typeof output === "string" ? output.trim() : "";
    
    // Remove any timestamps that might still appear
    text = this.removeTimestamps(text);
    
    // Basic validation
    if (!text || text.length === 0) {
      return "";
    }

    // Check for obvious hallucinations
    if (this.isLikelyHallucination(text)) {
      console.warn("Detected likely hallucination in chunk, skipping");
      return "";
    }

    return text;
  }

  /**
   * Remove timestamp patterns from text
   */
  private removeTimestamps(text: string): string {
    // Remove patterns like: [00:00.000 --> 00:05.000]
    let cleaned = text.replace(/\[\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}\]/g, "");
    
    // Remove patterns like: [00:00:00.000 --> 00:00:05.000]
    cleaned = cleaned.replace(/\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]/g, "");
    
    // Remove standalone timestamps like: [00:00.000]
    cleaned = cleaned.replace(/\[\d{2}:\d{2}\.\d{3}\]/g, "");
    
    // Remove leading/trailing whitespace and multiple spaces
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    
    return cleaned;
  }

  /**
   * Simple hallucination detection
   */
  private isLikelyHallucination(text: string): boolean {
    // Check for excessive character repetition
    const charRepetition = /(.)\1{30,}/;
    if (charRepetition.test(text)) {
      console.warn("Detected excessive character repetition");
      return true;
    }

    // Check for very short repetitive patterns
    const patternRepetition = /(.{1,5})\1{15,}/;
    if (patternRepetition.test(text)) {
      console.warn("Detected excessive pattern repetition");
      return true;
    }

    return false;
  }

  /**
   * Change the model (tiny, base, small, medium, large)
   */
  setModel(modelName: "tiny" | "base" | "small" | "medium" | "large"): void {
    this.modelName = modelName;
    this.isInitialized = false;
    console.log(`Model changed to: ${modelName}`);
  }
}
