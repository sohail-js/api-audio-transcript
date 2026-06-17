import OpenAI from "openai";
import { readFile, unlink, stat } from "fs/promises";
import { basename, join, dirname } from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import logger, { createChildLogger } from "../utils/logger.js";

const execFileAsync = promisify(execFile);

// OpenAI Whisper API file size limit: 25MB
const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25MB
const OPTIMIZED_WAV_SAMPLE_RATE = 8000;
const OPTIMIZED_WAV_BYTES_PER_SECOND = OPTIMIZED_WAV_SAMPLE_RATE * 2; // 16-bit mono PCM

export type TranscriptionProgress = {
  completed: number;
  stage: string;
  message: string;
  currentChunk?: number;
  totalChunks?: number;
  chunkText?: string;
};

type ProgressCallback = (progress: TranscriptionProgress) => void;

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

/**
 * Service for managing audio transcription using OpenAI's Whisper API
 */
export class TranscriptionService {
  private openai: OpenAI;
  private isInitialized: boolean = false;
  private serviceLogger = createChildLogger({
    service: "TranscriptionService",
  });

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

    this.serviceLogger.info("Initializing OpenAI Whisper API service...");
    this.isInitialized = true;
    this.serviceLogger.info("OpenAI Whisper service initialized");
  }

  /**
   * Transcribes audio file to text using OpenAI's Whisper API
   * @param filePath - Absolute path to the audio file
   * @param language - Optional language code (e.g., 'hi', 'en', 'ur'). If not provided, auto-detects.
   * @param useDiarize - If true, uses gpt-4o-transcribe-diarize model for speaker diarization
   * @param requestId - Optional request ID for logging context
   * @param useHighAccuracy - If true, uses gpt-4o-transcribe model for higher accuracy (better for background noise, complex dialogues)
   * @returns Transcribed text
   */
  async transcribe(
    filePath: string,
    language?: string,
    useDiarize?: boolean,
    requestId?: string,
    useHighAccuracy?: boolean,
    onProgress?: ProgressCallback
  ): Promise<string> {
    await this.initialize();

    // Determine model outside try block so it's accessible in catch
    // Priority: diarize > high accuracy > default mini
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : useHighAccuracy
      ? "gpt-4o-transcribe"
      : "gpt-4o-mini-transcribe";

    // Create logger with context
    const logContext: Record<string, any> = { filePath, model };
    if (requestId) logContext.requestId = requestId;
    const log = createChildLogger(logContext);

    // Check original file size first
    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    try {
      log.info("Starting transcription");
      onProgress?.({
        completed: 5,
        stage: "started",
        message: "Transcription request accepted",
      });

      // Check original file size
      const originalStats = await stat(filePath);
      const originalSizeMB = originalStats.size / (1024 * 1024);
      onProgress?.({
        completed: 10,
        stage: "validating",
        message: "Audio file validated",
      });
      log.info(
        { originalSizeMB: originalSizeMB.toFixed(2) },
        "File size checked"
      );

      const fileExt = filePath.split(".").pop()?.toLowerCase();

      // Strategy: Try to minimize processing overhead
      // 1. If original file is under 25MB, try it first (might work without conversion)
      // 2. Only convert to WAV if original format fails
      // 3. For very large files, use chunking (required due to 25MB file size limit)

      if (originalSizeMB > 25) {
        // File is too large even in original format - must use chunking
        log.info(
          { originalSizeMB: originalSizeMB.toFixed(2) },
          "Large file detected, processing in chunks (required due to 25MB file size limit)"
        );
        return await this.transcribeInChunks(
          filePath,
          language,
          useDiarize,
          requestId,
          useHighAccuracy,
          onProgress
        );
      }

      // Try original format first (if it's a supported format)
      // This avoids unnecessary conversion and keeps it to 1 API call
      const supportedFormats = ["mp3", "m4a", "wav", "webm", "mpeg", "mpga"];
      if (supportedFormats.includes(fileExt || "")) {
        log.debug(
          { fileExt },
          "File is under 25MB and in supported format, trying original format first"
        );
        try {
          const fileBuffer = await readFile(filePath);
          const fileName = basename(filePath);
          const mimeType = this.getMimeType(fileName);

          const file = new File([fileBuffer], fileName, {
            type: mimeType,
            lastModified: Date.now(),
          });
          onProgress?.({
            completed: 45,
            stage: "transcribing",
            message: "Audio sent to OpenAI for transcription",
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

          log.info(
            { textLength: text.length },
            "Transcription completed using original format"
          );
          onProgress?.({
            completed: 90,
            stage: "transcribed",
            message: "Transcript generated",
          });
          return text.trim();
        } catch (error: any) {
          // If original format fails, fall back to WAV conversion
          if (
            error.message?.includes("corrupted") ||
            error.message?.includes("unsupported") ||
            error.message?.includes("file")
          ) {
            log.warn(
              { error: error.message },
              "Original format failed, converting to WAV"
            );
            // Continue to WAV conversion below
          } else {
            throw error;
          }
        }
      }

      // Convert to WAV format if original format didn't work or isn't supported
      if (fileExt !== "wav") {
        log.info({ fileExt }, "Converting to optimized WAV format");
        onProgress?.({
          completed: 25,
          stage: "converting",
          message: "Converting audio to WAV",
        });
        convertedFilePath = await this.convertToWav(filePath, true, log); // optimized
        finalFilePath = convertedFilePath;
        log.debug({ convertedFilePath }, "File converted to WAV");
        onProgress?.({
          completed: 35,
          stage: "converted",
          message: "Audio conversion completed",
        });
      }

      // Read the file and create a File object for OpenAI API
      const fileBuffer = await readFile(finalFilePath);
      const fileName = basename(finalFilePath);
      const fileSizeMB = fileBuffer.length / (1024 * 1024);

      // Double-check file size after conversion
      if (fileSizeMB > 25) {
        // If conversion made it too large, fall back to chunking
        log.info(
          { fileSizeMB: fileSizeMB.toFixed(2) },
          "Converted file exceeds 25MB limit, processing in chunks"
        );
        // Clean up converted file
        if (convertedFilePath) {
          await unlink(convertedFilePath).catch(() => {});
        }
        return await this.transcribeInChunks(
          filePath,
          language,
          useDiarize,
          requestId,
          useHighAccuracy,
          onProgress
        );
      }

      // Create File object with WAV MIME type
      const file = new File([fileBuffer], fileName, {
        type: "audio/wav",
        lastModified: Date.now(),
      });

      log.debug(
        { fileName, fileSizeMB: fileSizeMB.toFixed(2), mimeType: "audio/wav" },
        "File prepared for transcription"
      );
      onProgress?.({
        completed: 45,
        stage: "transcribing",
        message: "Audio sent to OpenAI for transcription",
      });

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

      log.info({ textLength: text.length }, "Transcription completed");
      onProgress?.({
        completed: 90,
        stage: "transcribed",
        message: "Transcript generated",
      });

      // Clean up converted file if we created one
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          log.warn(
            {
              error: err instanceof Error ? err.message : "Unknown error",
              convertedFilePath,
            },
            "Failed to delete converted file"
          );
        });
      }

      return text.trim();
    } catch (error) {
      // Clean up converted file if we created one, even on error
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          log.warn(
            {
              error: err instanceof Error ? err.message : "Unknown error",
              convertedFilePath,
            },
            "Failed to delete converted file on error"
          );
        });
      }

      log.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
        },
        "Transcription error"
      );

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
   * @param useHighAccuracy - If true, uses gpt-4o-transcribe model for higher accuracy (better for background noise, complex dialogues)
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
    useDiarize?: boolean,
    useHighAccuracy?: boolean
  ): Promise<string | any> {
    await this.initialize();

    // Determine model outside try block so it's accessible in catch
    // Priority: diarize > high accuracy > default mini
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : useHighAccuracy
      ? "gpt-4o-transcribe"
      : "gpt-4o-mini-transcribe";

    // Convert to WAV format for better compatibility with new models
    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    const logContext: Record<string, any> = { filePath, model };
    const log = createChildLogger(logContext);

    try {
      log.info("Starting transcription with options");

      // Convert to WAV format for better compatibility
      const fileExt = filePath.split(".").pop()?.toLowerCase();
      if (fileExt !== "wav") {
        log.info({ fileExt }, "Converting to WAV format for compatibility");
        convertedFilePath = await this.convertToWav(filePath, false, log);
        finalFilePath = convertedFilePath;
        log.debug({ convertedFilePath }, "File converted to WAV");
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
          log.warn(
            {
              error: err instanceof Error ? err.message : "Unknown error",
              convertedFilePath,
            },
            "Failed to delete converted file"
          );
        });
      }

      return result;
    } catch (error) {
      // Clean up converted file if we created one, even on error
      if (convertedFilePath) {
        await unlink(convertedFilePath).catch((err) => {
          log.warn(
            {
              error: err instanceof Error ? err.message : "Unknown error",
              convertedFilePath,
            },
            "Failed to delete converted file on error"
          );
        });
      }

      log.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
        },
        "Transcription error"
      );
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
    useDiarize?: boolean,
    requestId?: string,
    useHighAccuracy?: boolean,
    onProgress?: ProgressCallback
  ): Promise<string> {
    // Priority: diarize > high accuracy > default mini
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : useHighAccuracy
      ? "gpt-4o-transcribe"
      : "gpt-4o-mini-transcribe";

    const logContext: Record<string, any> = {
      filePath,
      model,
      operation: "transcribeInChunks",
    };
    if (requestId) logContext.requestId = requestId;
    const log = createChildLogger(logContext);

    const chunkFiles: string[] = [];
    let chunkSourcePath = filePath;
    let convertedChunkSourcePath: string | null = null;

    try {
      // Get audio duration to calculate chunk size
      let duration = await this.getAudioDuration(chunkSourcePath);

      if (
        (!duration || duration <= 0 || !isFinite(duration)) &&
        !filePath.toLowerCase().endsWith(".wav")
      ) {
        log.warn(
          { filePath },
          "Could not determine duration from original file, converting to WAV before chunking"
        );
        onProgress?.({
          completed: 15,
          stage: "converting",
          message: "Preparing audio for chunking",
        });
        convertedChunkSourcePath = await this.convertToWav(filePath, true, log);
        chunkSourcePath = convertedChunkSourcePath;
        duration = await this.getAudioDuration(chunkSourcePath);
      }

      onProgress?.({
        completed: 15,
        stage: "analyzing",
        message: "Audio duration analyzed",
      });

      // Validate duration - must be positive to calculate chunk sizes
      if (!duration || duration <= 0 || !isFinite(duration)) {
        log.error(
          { duration, filePath: chunkSourcePath },
          "Invalid or zero audio duration, cannot process in chunks"
        );
        throw new Error(
          "Cannot determine audio duration. The file may be corrupted or in an unsupported format."
        );
      }

      const fileStats = await stat(chunkSourcePath);
      const fileSizeBytes = fileStats.size;
      const durationMinutes = (duration / 60).toFixed(2);
      const estimatedCost = ((duration / 60) * 0.006).toFixed(4);

      log.info(
        {
          durationMinutes: parseFloat(durationMinutes),
          fileSizeBytes,
          estimatedCost: parseFloat(estimatedCost),
        },
        "Large file detected, estimated cost (same regardless of chunking)"
      );

      // Estimate chunk duration from optimized WAV output size, not compressed input size.
      // 8kHz 16-bit mono PCM is about 16KB/s, so 10-minute chunks stay around 9.6MB.
      const safeChunkSizeBytes = MAX_FILE_SIZE_BYTES * 0.8;
      const optimizedChunkDuration = Math.floor(
        safeChunkSizeBytes / OPTIMIZED_WAV_BYTES_PER_SECOND
      );

      // Minimum chunk duration of 60 seconds, maximum of 10 minutes.
      const actualChunkDuration = Math.max(
        60,
        Math.min(optimizedChunkDuration, 600)
      );
      const numChunks = Math.ceil(duration / actualChunkDuration);

      log.info(
        {
          numChunks,
          chunkDuration: actualChunkDuration,
        },
        "Splitting into chunks (required due to 25MB file size limit, cost remains the same)"
      );
      onProgress?.({
        completed: 20,
        stage: "chunking",
        message: `Splitting audio into ${numChunks} chunks`,
        totalChunks: numChunks,
      });

      // Create chunk files
      for (let i = 0; i < numChunks; i++) {
        const startTime = i * actualChunkDuration;
        const chunkFile = chunkSourcePath.replace(
          /(\.[^.]+)$/,
          `_chunk${i}.wav`
        );
        chunkFiles.push(chunkFile);

        await this.createOptimizedWavChunk(
          chunkSourcePath,
          chunkFile,
          startTime,
          actualChunkDuration
        );
      }

      log.info(
        { chunkCount: chunkFiles.length },
        "Created chunk files, processing in parallel"
      );
      onProgress?.({
        completed: 25,
        stage: "chunks-created",
        message: "Audio chunks created",
        totalChunks: chunkFiles.length,
      });

      // Process chunks in parallel (max 2 at a time to avoid rate limits)
      const maxConcurrent = 2;
      const results: string[] = [];

      for (let i = 0; i < chunkFiles.length; i += maxConcurrent) {
        const batch = chunkFiles.slice(i, i + maxConcurrent);
        const batchNum = Math.ceil((i + 1) / maxConcurrent);
        const totalBatches = Math.ceil(chunkFiles.length / maxConcurrent);
        log.info(
          {
            batchNum,
            totalBatches,
            chunkStart: i,
            chunkEnd: i + batch.length - 1,
          },
          "Processing batch"
        );

        const batchResults = await Promise.all(
          batch.map(async (chunkFile, batchIndex) => {
            const text = await this.transcribeSingle(
              chunkFile,
              language,
              useDiarize,
              requestId,
              useHighAccuracy
            );

            return {
              chunkNumber: i + batchIndex + 1,
              text,
            };
          })
        );

        for (const batchResult of batchResults) {
          results[batchResult.chunkNumber - 1] = batchResult.text;
          const completedChunks = results.filter(
            (result) => result !== undefined
          ).length;

          onProgress?.({
            completed: clampProgress(
              25 + (completedChunks / chunkFiles.length) * 65
            ),
            stage: "transcribing",
            message: `Transcribed ${completedChunks} of ${chunkFiles.length} chunks`,
            currentChunk: batchResult.chunkNumber,
            totalChunks: chunkFiles.length,
            chunkText: batchResult.text,
          });
        }
      }

      // Combine results
      const combinedText = results.filter((r) => r.length > 0).join(" ");
      log.info(
        {
          combinedTextLength: combinedText.length,
          chunkCount: results.length,
        },
        "Combined transcription from chunks"
      );
      onProgress?.({
        completed: 90,
        stage: "transcribed",
        message: "Transcript generated from all chunks",
        currentChunk: results.length,
        totalChunks: chunkFiles.length,
      });

      return combinedText;
    } catch (error) {
      log.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
        },
        "Error in transcribeInChunks"
      );
      throw new Error(
        `Failed to transcribe in chunks: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      await Promise.all(chunkFiles.map((file) => unlink(file).catch(() => {})));
      if (convertedChunkSourcePath) {
        await unlink(convertedChunkSourcePath).catch(() => {});
      }
    }
  }

  /**
   * Transcribe a single file (or chunk) - handles WAV conversion
   */
  private async transcribeSingle(
    filePath: string,
    language?: string,
    useDiarize?: boolean,
    requestId?: string,
    useHighAccuracy?: boolean
  ): Promise<string> {
    // Priority: diarize > high accuracy > default mini
    const model = useDiarize
      ? "gpt-4o-transcribe-diarize"
      : useHighAccuracy
      ? "gpt-4o-transcribe"
      : "gpt-4o-mini-transcribe";

    // Create logger with context
    const logContext: Record<string, any> = {
      filePath,
      model,
      operation: "transcribeSingle",
    };
    if (requestId) logContext.requestId = requestId;
    const log = createChildLogger(logContext);

    let convertedFilePath: string | null = null;
    let finalFilePath = filePath;

    try {
      // Convert to optimized WAV if needed (smaller file size)
      const fileExt = filePath.split(".").pop()?.toLowerCase();
      if (fileExt !== "wav") {
        convertedFilePath = await this.convertToWav(filePath, true, log); // optimized
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

      log.debug(
        { textLength: text.length },
        "Transcription completed for chunk"
      );

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
      log.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
        },
        "Error transcribing chunk"
      );
      throw error;
    }
  }

  /**
   * Get audio duration in seconds using ffprobe
   */
  private async getAudioDuration(filePath: string): Promise<number> {
    try {
      const { stdout } = await execFileAsync(
        "ffprobe",
        [
          "-v",
          "error",
          "-print_format",
          "json",
          "-show_format",
          "-show_streams",
          filePath,
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );
      const metadata = JSON.parse(stdout);
      const duration = this.findDurationFromFfprobe(metadata);

      if (duration) {
        return duration;
      }

      const estimatedDuration = await this.estimateDurationFromBitrate(
        filePath,
        metadata
      );

      if (!estimatedDuration) {
        this.serviceLogger.warn(
          { filePath },
          "Could not parse audio duration"
        );
        return 0;
      }

      this.serviceLogger.warn(
        { filePath, estimatedDuration },
        "Estimated audio duration from bitrate"
      );
      return estimatedDuration;
    } catch (error) {
      if (this.isExecutableNotFoundError(error)) {
        throw new Error(
          "ffprobe executable not found. Install FFmpeg and ensure ffprobe is available on PATH."
        );
      }

      this.serviceLogger.warn(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          filePath,
        },
        "Could not determine audio duration"
      );
      return 0;
    }
  }

  private findDurationFromFfprobe(metadata: any): number {
    const candidates = [
      metadata?.format?.duration,
      ...(Array.isArray(metadata?.streams)
        ? metadata.streams.map((stream: any) => stream?.duration)
        : []),
    ];

    for (const candidate of candidates) {
      const duration = Number.parseFloat(candidate);
      if (duration > 0 && Number.isFinite(duration)) {
        return duration;
      }
    }

    return 0;
  }

  private async estimateDurationFromBitrate(
    filePath: string,
    metadata: any
  ): Promise<number> {
    const candidates = [
      metadata?.format?.bit_rate,
      ...(Array.isArray(metadata?.streams)
        ? metadata.streams.map((stream: any) => stream?.bit_rate)
        : []),
    ];

    for (const candidate of candidates) {
      const bitRate = Number.parseFloat(candidate);
      if (bitRate > 0 && Number.isFinite(bitRate)) {
        const fileStats = await stat(filePath);
        return (fileStats.size * 8) / bitRate;
      }
    }

    return 0;
  }

  private async createOptimizedWavChunk(
    inputPath: string,
    outputPath: string,
    startTime: number,
    duration: number
  ): Promise<void> {
    try {
      await execFileAsync(
        "ffmpeg",
        [
          "-v",
          "error",
          "-y",
          "-ss",
          String(startTime),
          "-i",
          inputPath,
          "-t",
          String(duration),
          "-vn",
          "-ar",
          String(OPTIMIZED_WAV_SAMPLE_RATE),
          "-ac",
          "1",
          "-c:a",
          "pcm_s16le",
          outputPath,
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );
    } catch (error) {
      if (this.isExecutableNotFoundError(error)) {
        throw new Error(
          "ffmpeg executable not found. Install FFmpeg and ensure ffmpeg is available on PATH."
        );
      }

      throw error;
    }
  }

  private isExecutableNotFoundError(error: unknown): boolean {
    return (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: unknown }).code === "ENOENT"
    );
  }

  /**
   * Convert audio file to WAV format using ffmpeg
   * @param filePath - Path to the input audio file
   * @param optimized - If true, uses lower sample rate to reduce file size (8kHz instead of 16kHz)
   * @param log - Optional logger instance for logging
   * @returns Path to the converted WAV file
   */
  private async convertToWav(
    filePath: string,
    optimized: boolean = false,
    log?: ReturnType<typeof createChildLogger>
  ): Promise<string> {
    const outputPath = filePath.replace(/\.[^.]+$/, ".wav");

    try {
      // Use optimized settings to reduce file size while maintaining acceptable quality
      // -ar 8000: sample rate 8kHz (still good for speech, reduces file size by ~50%)
      // -ar 16000: sample rate 16kHz (better quality, larger file)
      // -ac 1: mono channel (reduces file size by 50% vs stereo)
      // -c:a pcm_s16le: PCM 16-bit little-endian (standard WAV format)
      const sampleRate = optimized ? "8000" : "16000";
      await execFileAsync(
        "ffmpeg",
        [
          "-v",
          "error",
          "-y",
          "-i",
          filePath,
          "-ar",
          sampleRate,
          "-ac",
          "1",
          "-c:a",
          "pcm_s16le",
          outputPath,
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );

      const logger = log || this.serviceLogger;
      logger.info({ sampleRate, outputPath }, "Successfully converted to WAV");
      return outputPath;
    } catch (error) {
      if (this.isExecutableNotFoundError(error)) {
        throw new Error(
          "ffmpeg executable not found. Install FFmpeg and ensure ffmpeg is available on PATH."
        );
      }

      const logger = log || this.serviceLogger;
      logger.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
          filePath,
        },
        "Failed to convert file to WAV"
      );
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

  /**
   * Generate text from transcript using OpenAI Chat Completions API
   * @param transcript - The transcribed text from audio
   * @param prompt - The prompt to use for text generation
   * @param model - Optional model to use (defaults to OPENAI_TEXT_GENERATION_MODEL env var or 'gpt-4o-mini')
   * @param requestId - Optional request ID for logging context
   * @returns Generated text
   */
  async generateTextFromTranscript(
    transcript: string,
    prompt: string,
    model?: string,
    requestId?: string
  ): Promise<string> {
    await this.initialize();

    // Get model from parameter, environment variable, or default
    const textGenerationModel =
      model || process.env.OPENAI_TEXT_GENERATION_MODEL || "gpt-4o-mini";

    // Create logger with context
    const logContext: Record<string, any> = {
      textGenerationModel,
      transcriptLength: transcript.length,
      promptLength: prompt.length,
    };
    if (requestId) logContext.requestId = requestId;
    const log = createChildLogger(logContext);

    try {
      log.info("Starting text generation from transcript");

      // Call OpenAI's Chat Completions API
      const completion = await this.openai.chat.completions.create({
        model: textGenerationModel,
        messages: [
          {
            role: "user",
            content: `${prompt}\n\nTranscript:\n${transcript}`,
          },
        ],
      });

      const generatedText =
        completion.choices[0]?.message?.content?.trim() || "";

      if (!generatedText) {
        log.warn("Empty response from text generation");
        throw new Error("Empty response from text generation API");
      }

      log.info(
        { generatedTextLength: generatedText.length },
        "Text generation completed"
      );

      return generatedText;
    } catch (error) {
      log.error(
        {
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
        },
        "Text generation error"
      );

      if (error instanceof Error) {
        // Handle specific OpenAI API errors
        if (error.message.includes("API key")) {
          throw new Error("Invalid or missing OpenAI API key");
        }
        if (error.message.includes("rate limit")) {
          throw new Error(
            "OpenAI API rate limit exceeded. Please try again later."
          );
        }
        if (error.message.includes("model")) {
          throw new Error(
            `Invalid model specified: ${textGenerationModel}. Please check OPENAI_TEXT_GENERATION_MODEL environment variable.`
          );
        }
      }

      throw new Error(
        `Failed to generate text: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }
}
