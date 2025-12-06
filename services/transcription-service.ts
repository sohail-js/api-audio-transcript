import OpenAI from "openai";
import { readFile } from "fs/promises";
import { basename } from "path";

/**
 * Service for managing audio transcription using OpenAI's Whisper API
 */
export class TranscriptionService {
  private openai: OpenAI;
  private isInitialized: boolean = false;

  constructor() {
    // Initialize OpenAI client with API key from environment variable
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OPENAI_API_KEY environment variable is required. Please set it in your .env file or environment."
      );
    }

    this.openai = new OpenAI({
      apiKey: apiKey,
    });
  }

  /**
   * Initializes the transcription service
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    console.log("Initializing OpenAI Whisper API service...");
    this.isInitialized = true;
    console.log("OpenAI Whisper service initialized");
  }

  /**
   * Transcribes audio file to text using OpenAI's Whisper API
   * @param filePath - Absolute path to the audio file
   * @param language - Optional language code (e.g., 'hi', 'en', 'ur'). If not provided, auto-detects.
   * @returns Transcribed text
   */
  async transcribe(filePath: string, language?: string): Promise<string> {
    await this.initialize();

    try {
      console.log(`Transcribing file: ${filePath}`);

      // Read the file and create a File object for OpenAI API
      const fileBuffer = await readFile(filePath);
      const fileName = basename(filePath);

      // Create a File object (available in Node.js 18+)
      const file = new File([fileBuffer], fileName, {
        type: this.getMimeType(fileName),
      });

      // Call OpenAI's Whisper API
      const transcription = await this.openai.audio.transcriptions.create({
        file: file,
        model: "whisper-1",
        language: language, // Optional: specify language or let it auto-detect
        response_format: "text", // Get plain text response
      });

      // The response is a string when response_format is "text"
      const text =
        typeof transcription === "string"
          ? transcription
          : (transcription as any).text || "";

      console.log(`Transcription completed: ${text.length} characters`);

      return text.trim();
    } catch (error) {
      console.error("Transcription error:", error);

      if (error instanceof Error) {
        // Handle specific OpenAI API errors
        if (error.message.includes("API key")) {
          throw new Error("Invalid or missing OpenAI API key");
        }
        if (error.message.includes("file")) {
          throw new Error(`File error: ${error.message}`);
        }
        if (error.message.includes("rate limit")) {
          throw new Error(
            "OpenAI API rate limit exceeded. Please try again later."
          );
        }
      }

      throw new Error(
        `Failed to transcribe: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Transcribe with additional options (JSON format, timestamps, etc.)
   * @param filePath - Absolute path to the audio file
   * @param options - Additional transcription options
   * @returns Transcription result
   */
  async transcribeWithOptions(
    filePath: string,
    options?: {
      language?: string;
      prompt?: string; // Context prompt to improve accuracy
      response_format?: "json" | "text" | "srt" | "verbose_json" | "vtt";
      temperature?: number; // 0-1, controls randomness
    }
  ): Promise<string | any> {
    await this.initialize();

    try {
      console.log(`Transcribing file with options: ${filePath}`);

      // Read the file and create a File object for OpenAI API
      const fileBuffer = await readFile(filePath);
      const fileName = basename(filePath);

      const file = new File([fileBuffer], fileName, {
        type: this.getMimeType(fileName),
      });

      const transcription = await this.openai.audio.transcriptions.create({
        file: file,
        model: "whisper-1",
        language: options?.language,
        prompt: options?.prompt,
        response_format: options?.response_format || "text",
        temperature: options?.temperature,
      });

      if (
        options?.response_format === "json" ||
        options?.response_format === "verbose_json"
      ) {
        return transcription;
      }

      return typeof transcription === "string"
        ? transcription
        : (transcription as any).text || "";
    } catch (error) {
      console.error("Transcription error:", error);
      throw new Error(
        `Failed to transcribe: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Get MIME type from file extension
   */
  private getMimeType(fileName: string): string {
    const ext = fileName.split(".").pop()?.toLowerCase();
    const mimeTypes: Record<string, string> = {
      mp3: "audio/mpeg",
      wav: "audio/wav",
      m4a: "audio/mp4",
      flac: "audio/flac",
      ogg: "audio/ogg",
      webm: "audio/webm",
      opus: "audio/opus",
    };
    return mimeTypes[ext || ""] || "audio/mpeg";
  }
}
