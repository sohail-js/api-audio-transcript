import { nodewhisper } from "nodejs-whisper";
import path from "path";

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
      
      // Configure transcription options
      // CRITICAL: Set language explicitly to 'hi' to prevent translation
      // With auto-detect, whisper.cpp translates mixed content to English
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
      
      // Remove any timestamps that might still appear (format: [00:00.000 --> 00:05.000])
      text = this.removeTimestamps(text);
      
      console.log(`Transcription completed: ${text.length} characters`);
      console.log(`Preview: ${text.substring(0, 100)}...`);
      
      // Basic validation
      if (!text || text.length === 0) {
        console.warn("Transcription returned empty text");
        return "";
      }

      // Check for obvious hallucinations (repetitive patterns)
      if (this.isLikelyHallucination(text)) {
        console.warn("Detected likely hallucination, returning empty");
        return "";
      }

      return text;
    } catch (error) {
      console.error("Transcription error:", error);
      throw new Error(
        `Failed to transcribe: ${error instanceof Error ? error.message : "Unknown error"}`
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
