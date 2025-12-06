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
      const options = {
        modelName: this.modelName,           // Model to use (tiny, base, small, medium, large)
        autoDownloadModelName: this.modelName, // Auto-download if not present
        whisperOptions: {
          language: "auto",                  // Auto-detect language
          gen_file_txt: false,               // Don't generate txt file
          gen_file_subtitle: false,          // Don't generate subtitle files
          gen_file_vtt: false,
          word_timestamps: false,            // Don't need word-level timestamps
          timestamp_size: 0,
        },
      };

      // Transcribe the audio file
      const output = await nodewhisper(filePath, options);
      
      // nodejs-whisper returns the text directly
      const text = typeof output === "string" ? output.trim() : "";
      
      console.log(`Transcription completed: ${text.length} characters`);
      
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
