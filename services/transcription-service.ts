import OpenAI from "openai";
import { readFile, unlink, stat } from "fs/promises";
import { basename, join, dirname } from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// OpenAI Whisper API file size limit: 25MB
const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25MB

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
   * @param useDiarize - If true, uses gpt-4o-transcribe-diarize model for speaker diarization
   * @returns Transcribed text
   */
  async transcribe(
    filePath: string,
    language?: string,
    useDiarize?: boolean
  ): Promise<string> {
    await this.initialize();

    // Determine model outside try block so it's accessible in catch
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : "gpt-4o-mini-transcribe";

    // Check original file size first
    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    try {
      console.log(`Transcribing file: ${filePath} using model: ${model}`);

      // Check original file size
      const originalStats = await stat(filePath);
      const originalSizeMB = originalStats.size / (1024 * 1024);
      console.log(`Original file size: ${originalSizeMB.toFixed(2)}MB`);

      const fileExt = filePath.split(".").pop()?.toLowerCase();

      // Strategy: Try to minimize processing overhead
      // 1. If original file is under 25MB, try it first (might work without conversion)
      // 2. Only convert to WAV if original format fails
      // 3. For very large files, use chunking (required due to 25MB file size limit)

      if (originalSizeMB > 25) {
        // File is too large even in original format - must use chunking
        console.log(
          `Large file detected (${originalSizeMB.toFixed(
            2
          )}MB). Processing in chunks (required due to 25MB file size limit)`
        );
        return await this.transcribeInChunks(filePath, language, useDiarize);
      }

      // Try original format first (if it's a supported format)
      // This avoids unnecessary conversion and keeps it to 1 API call
      const supportedFormats = ["mp3", "m4a", "wav", "webm", "mpeg", "mpga"];
      if (supportedFormats.includes(fileExt || "")) {
        console.log(
          `File is under 25MB and in supported format (${fileExt}). Trying original format first...`
        );
        try {
          const fileBuffer = await readFile(filePath);
          const fileName = basename(filePath);
          const mimeType = this.getMimeType(fileName);

          const file = new File([fileBuffer], fileName, {
            type: mimeType,
            lastModified: Date.now(),
          });

          const transcription = await this.openai.audio.transcriptions.create({
            file: file,
            model: model,
            language: language,
            response_format: "text",
          });

          const text =
            typeof transcription === "string"
              ? transcription
              : (transcription as any).text || "";

          console.log(
            `Transcription completed using original format: ${text.length} characters`
          );
          return text.trim();
        } catch (error: any) {
          // If original format fails, fall back to WAV conversion
          if (
            error.message?.includes("corrupted") ||
            error.message?.includes("unsupported") ||
            error.message?.includes("file")
          ) {
            console.log(
              `Original format failed, converting to WAV: ${error.message}`
            );
            // Continue to WAV conversion below
          } else {
            throw error;
          }
        }
      }

      // Convert to WAV format if original format didn't work or isn't supported
      if (fileExt !== "wav") {
        console.log(`Converting ${fileExt} to optimized WAV format...`);
        convertedFilePath = await this.convertToWav(filePath, true); // optimized
        finalFilePath = convertedFilePath;
        console.log(`Converted to: ${convertedFilePath}`);
      }

      // Read the file and create a File object for OpenAI API
      const fileBuffer = await readFile(finalFilePath);
      const fileName = basename(finalFilePath);
      const fileSizeMB = fileBuffer.length / (1024 * 1024);

      // Double-check file size after conversion
      if (fileSizeMB > 25) {
        // If conversion made it too large, fall back to chunking
        console.log(
          `Converted file (${fileSizeMB.toFixed(
            2
          )}MB) exceeds 25MB limit. Processing in chunks...`
        );
        // Clean up converted file
        if (convertedFilePath) {
          await unlink(convertedFilePath).catch(() => {});
        }
        return await this.transcribeInChunks(filePath, language, useDiarize);
      }

      // Create File object with WAV MIME type
      const file = new File([fileBuffer], fileName, {
        type: "audio/wav",
        lastModified: Date.now(),
      });

      console.log(
        `File prepared: ${fileName}, size: ${fileSizeMB.toFixed(
          2
        )}MB, type: audio/wav`
      );

      // Call OpenAI's Whisper API
      const transcription = await this.openai.audio.transcriptions.create({
        file: file,
        model: model,
        language: language, // Optional: specify language or let it auto-detect
        response_format: "text", // Get plain text response
      });

      // The response is a string when response_format is "text"
      const text =
        typeof transcription === "string"
          ? transcription
          : (transcription as any).text || "";

      console.log(`Transcription completed: ${text.length} characters`);

      // Clean up converted file if we created one
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          console.warn(`Failed to delete converted file: ${err}`);
        });
      }

      return text.trim();
    } catch (error) {
      // Clean up converted file if we created one, even on error
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          console.warn(`Failed to delete converted file on error: ${err}`);
        });
      }

      console.error("Transcription error:", error);

      if (error instanceof Error) {
        // Handle specific OpenAI API errors
        if (error.message.includes("API key")) {
          throw new Error("Invalid or missing OpenAI API key");
        }
        if (
          error.message.includes("file") ||
          error.message.includes("corrupted") ||
          error.message.includes("unsupported")
        ) {
          // Provide more helpful error message for file format issues
          const fileExt = filePath.split(".").pop()?.toLowerCase();
          throw new Error(
            `File format error: ${error.message}. ` +
              `The file (${fileExt}) might not be compatible with ${model}. ` +
              `The service will automatically convert to WAV format if needed.`
          );
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
   * @param useDiarize - If true, uses gpt-4o-transcribe-diarize model for speaker diarization
   * @returns Transcription result
   */
  async transcribeWithOptions(
    filePath: string,
    options?: {
      language?: string;
      prompt?: string; // Context prompt to improve accuracy
      response_format?: "json" | "text" | "srt" | "verbose_json" | "vtt";
      temperature?: number; // 0-1, controls randomness
    },
    useDiarize?: boolean
  ): Promise<string | any> {
    await this.initialize();

    // Determine model outside try block so it's accessible in catch
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : "gpt-4o-mini-transcribe";

    // Convert to WAV format for better compatibility with new models
    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    try {
      console.log(
        `Transcribing file with options: ${filePath} using model: ${model}`
      );

      // Convert to WAV format for better compatibility
      const fileExt = filePath.split(".").pop()?.toLowerCase();
      if (fileExt !== "wav") {
        console.log(`Converting ${fileExt} to WAV format for compatibility...`);
        convertedFilePath = await this.convertToWav(filePath);
        finalFilePath = convertedFilePath;
        console.log(`Converted to: ${convertedFilePath}`);
      }

      // Read the converted file and create a File object for OpenAI API
      const fileBuffer = await readFile(finalFilePath);
      const fileName = basename(finalFilePath);

      // Create File object with WAV MIME type
      const file = new File([fileBuffer], fileName, {
        type: "audio/wav",
        lastModified: Date.now(),
      });

      const transcription = await this.openai.audio.transcriptions.create({
        file: file,
        model: model,
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

      const result =
        typeof transcription === "string"
          ? transcription
          : (transcription as any).text || "";

      // Clean up converted file if we created one
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          console.warn(`Failed to delete converted file: ${err}`);
        });
      }

      return result;
    } catch (error) {
      // Clean up converted file if we created one, even on error
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          console.warn(`Failed to delete converted file on error: ${err}`);
        });
      }

      console.error("Transcription error:", error);
      throw new Error(
        `Failed to transcribe: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Transcribe large file by splitting into chunks and processing in parallel
   *
   * Note: OpenAI charges by audio duration ($0.006/minute), not file size or number of API calls.
   * Chunking is only necessary due to the 25MB file size limit, not for cost reasons.
   * The total cost is the same whether processed as 1 file or multiple chunks.
   */
  private async transcribeInChunks(
    filePath: string,
    language?: string,
    useDiarize?: boolean
  ): Promise<string> {
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : "gpt-4o-mini-transcribe";

    try {
      // Get audio duration to calculate chunk size
      const duration = await this.getAudioDuration(filePath);
      const fileStats = await stat(filePath);
      const fileSizeBytes = fileStats.size;

      console.log(
        `Large file detected. Duration: ${(duration / 60).toFixed(
          2
        )} minutes. ` +
          `Estimated cost: $${((duration / 60) * 0.006).toFixed(
            4
          )} (same regardless of chunking)`
      );

      // Estimate bytes per second to calculate safe chunk duration
      // Use 80% of limit to be safe (20MB per chunk)
      const safeChunkSizeBytes = MAX_FILE_SIZE_BYTES * 0.8;
      const bytesPerSecond = fileSizeBytes / duration;
      const chunkDuration = Math.floor(safeChunkSizeBytes / bytesPerSecond);

      // Use optimized WAV conversion (8kHz) which reduces file size by ~50%
      // This allows larger chunks, reducing processing overhead (cost is the same)
      // Estimate: optimized WAV is ~6x larger than compressed (vs 12x for standard WAV)
      const optimizedWavMultiplier = 6;
      const optimizedChunkDuration = Math.floor(
        chunkDuration / optimizedWavMultiplier
      );

      // Minimum chunk duration of 60 seconds, maximum of 5 minutes
      // Optimized WAV allows larger chunks, reducing processing overhead
      const actualChunkDuration = Math.max(
        60,
        Math.min(optimizedChunkDuration, 300)
      );
      const numChunks = Math.ceil(duration / actualChunkDuration);

      console.log(
        `Splitting into ${numChunks} chunks of ~${actualChunkDuration}s each ` +
          `(required due to 25MB file size limit, cost remains the same)`
      );

      console.log(
        `Splitting into ${numChunks} chunks of ~${actualChunkDuration}s each`
      );

      // Create chunk files
      const chunkFiles: string[] = [];
      for (let i = 0; i < numChunks; i++) {
        const startTime = i * actualChunkDuration;
        const chunkFile = filePath.replace(/(\.[^.]+)$/, `_chunk${i}$1`);
        chunkFiles.push(chunkFile);

        // Extract chunk using ffmpeg (copy codec to avoid re-encoding)
        await execAsync(
          `ffmpeg -i "${filePath}" -ss ${startTime} -t ${actualChunkDuration} -c copy "${chunkFile}" -y`
        );
      }

      console.log(
        `Created ${chunkFiles.length} chunk files, processing in parallel...`
      );

      // Process chunks in parallel (max 2 at a time to avoid rate limits)
      const maxConcurrent = 2;
      const results: string[] = [];

      for (let i = 0; i < chunkFiles.length; i += maxConcurrent) {
        const batch = chunkFiles.slice(i, i + maxConcurrent);
        console.log(
          `Processing batch ${Math.ceil((i + 1) / maxConcurrent)}/${Math.ceil(
            chunkFiles.length / maxConcurrent
          )} (Chunks ${i}-${i + batch.length - 1})`
        );

        const batchResults = await Promise.all(
          batch.map((chunkFile) =>
            this.transcribeSingle(chunkFile, language, useDiarize)
          )
        );
        results.push(...batchResults);
      }

      // Clean up chunk files
      await Promise.all(chunkFiles.map((file) => unlink(file).catch(() => {})));

      // Combine results
      const combinedText = results.filter((r) => r.length > 0).join(" ");
      console.log(
        `Combined transcription: ${combinedText.length} characters from ${results.length} chunks`
      );

      return combinedText;
    } catch (error) {
      console.error("Error in transcribeInChunks:", error);
      throw new Error(
        `Failed to transcribe in chunks: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Transcribe a single file (or chunk) - handles WAV conversion
   */
  private async transcribeSingle(
    filePath: string,
    language?: string,
    useDiarize?: boolean
  ): Promise<string> {
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : "gpt-4o-mini-transcribe";

    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    try {
      // Convert to optimized WAV if needed (smaller file size)
      const fileExt = filePath.split(".").pop()?.toLowerCase();
      if (fileExt !== "wav") {
        convertedFilePath = await this.convertToWav(filePath, true); // optimized
        finalFilePath = convertedFilePath;
      }

      // Read the file and create a File object
      const fileBuffer = await readFile(finalFilePath);
      const fileName = basename(finalFilePath);

      // Check file size
      const fileSizeMB = fileBuffer.length / (1024 * 1024);
      if (fileSizeMB > 25) {
        throw new Error(
          `Chunk file size (${fileSizeMB.toFixed(
            2
          )}MB) still exceeds 25MB limit`
        );
      }

      const file = new File([fileBuffer], fileName, {
        type: "audio/wav",
        lastModified: Date.now(),
      });

      // Call OpenAI's Whisper API
      const transcription = await this.openai.audio.transcriptions.create({
        file: file,
        model: model,
        language: language,
        response_format: "text",
      });

      const text =
        typeof transcription === "string"
          ? transcription
          : (transcription as any).text || "";

      // Clean up converted file
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch(() => {});
      }

      return text.trim();
    } catch (error) {
      // Clean up on error
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch(() => {});
      }
      throw error;
    }
  }

  /**
   * Get audio duration in seconds using ffprobe
   */
  private async getAudioDuration(filePath: string): Promise<number> {
    try {
      const { stdout } = await execAsync(
        `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`
      );
      const duration = parseFloat(stdout.trim());
      if (isNaN(duration)) {
        console.warn("Could not parse audio duration, defaulting to estimate");
        return 0;
      }
      return duration;
    } catch (error) {
      console.warn("Could not determine audio duration:", error);
      return 0;
    }
  }

  /**
   * Convert audio file to WAV format using ffmpeg
   * @param filePath - Path to the input audio file
   * @param optimized - If true, uses lower sample rate to reduce file size (8kHz instead of 16kHz)
   * @returns Path to the converted WAV file
   */
  private async convertToWav(
    filePath: string,
    optimized: boolean = false
  ): Promise<string> {
    const outputPath = filePath.replace(/\.[^.]+$/, ".wav");

    try {
      // Use optimized settings to reduce file size while maintaining acceptable quality
      // -ar 8000: sample rate 8kHz (still good for speech, reduces file size by ~50%)
      // -ar 16000: sample rate 16kHz (better quality, larger file)
      // -ac 1: mono channel (reduces file size by 50% vs stereo)
      // -c:a pcm_s16le: PCM 16-bit little-endian (standard WAV format)
      const sampleRate = optimized ? "8000" : "16000";
      await execAsync(
        `ffmpeg -i "${filePath}" -ar ${sampleRate} -ac 1 -c:a pcm_s16le "${outputPath}" -y`
      );

      console.log(
        `Successfully converted to WAV (${sampleRate}Hz): ${outputPath}`
      );
      return outputPath;
    } catch (error) {
      console.error(`Failed to convert file to WAV:`, error);
      throw new Error(
        `Failed to convert audio file to WAV format: ${
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
