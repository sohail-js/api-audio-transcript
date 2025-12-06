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
  private modelName: string = "tiny"; // Use 'tiny' model for faster processing (was 'base')
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
      console.log(`Audio duration detected: ${duration} seconds`);
      
      // If audio is longer than 60 seconds (lowered for testing), split into chunks
      // Original was 600 (10 minutes)
      const PARALLEL_THRESHOLD = 60; 
      
      if (duration > PARALLEL_THRESHOLD) {
        console.log(`Large file detected (> ${PARALLEL_THRESHOLD}s), processing in parallel chunks...`);
        return await this.transcribeInChunks(filePath, duration);
      } else {
        console.log(`Short file detected (<= ${PARALLEL_THRESHOLD}s), processing normally...`);
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
      console.log(`Getting duration for: ${filePath}`);
      const { stdout, stderr } = await execAsync(
        `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`
      );
      console.log(`ffprobe output: "${stdout.trim()}"`);
      const duration = parseFloat(stdout.trim());
      if (isNaN(duration)) {
        console.warn("Parsed duration is NaN");
        return 0;
      }
      return duration;
    } catch (error) {
      console.warn("Could not determine audio duration:", error);
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
    
    // Process chunks in parallel (max 2 at a time to avoid overwhelming the CPU)
    // 4 instances * 4 threads = 16 threads > 12 available threads -> Slowness
    const maxConcurrent = 2;
    const results: string[] = [];
    
    for (let i = 0; i < chunkFiles.length; i += maxConcurrent) {
      const batch = chunkFiles.slice(i, i + maxConcurrent);
      console.log(`Processing batch ${Math.ceil((i + 1) / maxConcurrent)}/${Math.ceil(chunkFiles.length / maxConcurrent)} (Chunks ${i}-${i + batch.length - 1})`);
      
      const batchResults = await Promise.all(
        batch.map(chunkFile => this.transcribeSingle(chunkFile))
      );
      results.push(...batchResults);
    }
    
    // Clean up chunk files - DISABLED for debugging
    // await Promise.all(chunkFiles.map(file => unlink(file).catch(() => {})));
    
    // Combine results
    const combinedText = results.filter(r => r.length > 0).join(" ");
    console.log(`Combined transcription: ${combinedText.length} characters`);
    
    return combinedText;
  }

  /**
   * Transcribe a single file (or chunk)
   */
  private async transcribeSingle(filePath: string): Promise<string> {
    // await this.initialize(); // Already called in transcribe()

    try {
      console.log(`Transcribing file: ${filePath}`);
      
      // Convert to WAV if needed (whisper-cli requires WAV)
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      
      let wavFile = filePath;
      if (!filePath.endsWith('.wav')) {
        wavFile = filePath.replace(/\.[^.]+$/, '.wav');
        console.log(`Converting to WAV: ${wavFile}`);
        await execAsync(`ffmpeg -i "${filePath}" -ar 16000 -ac 1 -c:a pcm_s16le "${wavFile}"`);
      }
      
      const whisperPath = "node_modules/nodejs-whisper/cpp/whisper.cpp/build/bin/whisper-cli";
      const modelPath = `node_modules/nodejs-whisper/cpp/whisper.cpp/models/ggml-${this.modelName}.bin`;
      
      // CRITICAL: Use -l hi (not auto) and do NOT use -tr flag
      // This forces transcription in Hindi, not translation to English
      const command = `"${whisperPath}" -l hi -m "${modelPath}" -f "${wavFile}" --no-timestamps -otxt`;
      
      console.log(`Executing: ${command}`);
      
      try {
        const { stdout, stderr } = await execAsync(command, {
          cwd: process.cwd(),
          maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        });
        
        if (stderr) {
            // whisper-cli writes progress to stderr, which is normal
            // But we should check if it contains errors
            // console.log(`Whisper stderr: ${stderr.substring(0, 200)}...`);
        }
      } catch (execError: any) {
        console.error(`Whisper execution failed for ${wavFile}`);
        console.error(`Command: ${command}`);
        console.error(`Error: ${execError.message}`);
        if (execError.stderr) console.error(`Stderr: ${execError.stderr}`);
        throw execError;
      }
      
      // Read the generated .txt file (whisper-cli creates filename.wav.txt)
      const { readFile } = await import("fs/promises");
      const txtFile = `${wavFile}.txt`;
      
      let text = "";
      try {
        console.log(`Reading output file: ${txtFile}`);
        text = await readFile(txtFile, "utf-8");
        text = text.trim();
        console.log(`Read ${text.length} chars from ${txtFile}`);
        
        // Clean up the txt file - DISABLED for debugging
        // await unlink(txtFile).catch(() => {});
        
        // Clean up WAV file if we created it - DISABLED for debugging
        // if (wavFile !== filePath) {
        //   await unlink(wavFile).catch(() => {});
        // }
      } catch (error) {
        console.error(`Error reading transcription file ${txtFile}:`, error);
        // Fallback to stdout if file reading fails, though whisper-cli usually writes to file with -otxt
        // text = stdout.trim(); 
        return "";
      }
      
      // Remove any timestamps that might still appear (though --no-timestamps should prevent this)
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
    } catch (error) {
      console.error("Transcription error in transcribeSingle:", error);
      throw new Error(
        `Failed to transcribe single file: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
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
