import { pipeline } from "@xenova/transformers";

const MODEL_NAME = "Xenova/whisper-tiny";
const TASK = "automatic-speech-recognition";

/**
 * Service for managing Whisper transcription model
 */
export class TranscriptionService {
  private transcriber: any = null;
  private isInitializing = false;
  private initPromise: Promise<any> | null = null;

  /**
   * Initializes the Whisper transcription model
   */
  async initialize(): Promise<void> {
    if (this.transcriber) {
      return;
    }

    if (this.isInitializing && this.initPromise) {
      return this.initPromise;
    }

    this.isInitializing = true;
    this.initPromise = (async () => {
      try {
        console.log("Loading Whisper model...");
        this.transcriber = await pipeline(TASK, MODEL_NAME);
        console.log("Whisper model loaded successfully");
      } catch (error) {
        this.isInitializing = false;
        this.initPromise = null;
        throw error;
      }
      this.isInitializing = false;
      return this.transcriber;
    })();

    return this.initPromise;
  }

  /**
   * Gets the transcriber instance, initializing if necessary
   */
  async getTranscriber(): Promise<any> {
    await this.initialize();
    if (!this.transcriber) {
      throw new Error("Transcriber not initialized");
    }
    return this.transcriber;
  }

  /**
   * Manually chunks audio and transcribes each chunk
   */
  private async transcribeChunked(
    transcriber: any,
    audioData: Float32Array,
    sampleRate: number
  ): Promise<string> {
    const chunkLengthSamples = 30 * sampleRate; // 30 seconds
    const strideSamples = 5 * sampleRate; // 5 seconds overlap
    const chunks: string[] = [];

    console.log(`Processing ${audioData.length} samples in chunks...`);

    for (
      let start = 0;
      start < audioData.length;
      start += chunkLengthSamples - strideSamples
    ) {
      const end = Math.min(start + chunkLengthSamples, audioData.length);
      const chunk = audioData.slice(start, end);

      const chunkStartTime = start / sampleRate;
      const chunkEndTime = end / sampleRate;
      console.log(
        `Processing chunk ${chunks.length + 1}: ${chunkStartTime.toFixed(
          2
        )}s - ${chunkEndTime.toFixed(2)}s (${chunk.length} samples)`
      );

      const options: any = {
        language: null,
        task: "transcribe",
        return_timestamps: false,
      };

      try {
        const result = await transcriber(chunk, options);
        let chunkText = "";

        if (typeof result === "string") {
          chunkText = result;
        } else if (result && typeof result === "object") {
          chunkText = result.text || result.transcription || "";
        }

        if (chunkText.trim()) {
          chunks.push(chunkText.trim());
          console.log(
            `Chunk ${chunks.length} transcribed: "${chunkText.substring(
              0,
              50
            )}..."`
          );
        } else {
          console.log(`Chunk ${chunks.length} produced empty result`);
        }
      } catch (error) {
        console.error(`Error transcribing chunk ${chunks.length + 1}:`, error);
        // Continue with other chunks even if one fails
      }
    }

    const fullText = chunks.join(" ");
    console.log(
      `Combined ${chunks.length} chunks into full transcription (${fullText.length} chars)`
    );
    return fullText;
  }

  /**
   * Transcribes audio data to text
   */
  async transcribe(audioData: Float32Array): Promise<string> {
    const transcriber = await this.getTranscriber();

    // Calculate audio duration in seconds (assuming 16kHz sample rate)
    const sampleRate = 16000;
    const durationSeconds = audioData.length / sampleRate;

    // For audio longer than 30 seconds, manually chunk and process
    if (durationSeconds > 30) {
      console.log(
        `Audio is ${durationSeconds.toFixed(2)}s long, using manual chunking`
      );
      return this.transcribeChunked(transcriber, audioData, sampleRate);
    }

    // For shorter audio, process directly
    const options: any = {
      language: null,
      task: "transcribe",
      return_timestamps: false,
    };

    console.log("Processing audio directly (no chunking needed)");
    const result = await transcriber(audioData, options);

    // Log the result structure for debugging
    console.log("Transcription result type:", typeof result);
    console.log(
      "Transcription result keys:",
      result && typeof result === "object" ? Object.keys(result) : "N/A"
    );
    const resultPreview = JSON.stringify(result, null, 2);
    console.log(
      "Transcription result (first 1000 chars):",
      resultPreview.substring(0, 1000)
    );

    // Handle different possible result structures
    if (typeof result === "string") {
      return result;
    }

    if (result && typeof result === "object") {
      // Check if result is an array (might be array of chunks)
      if (Array.isArray(result)) {
        console.log(`Result is an array with ${result.length} items`);
        const fullText = result
          .map((item: any) => {
            if (typeof item === "string") return item;
            return item.text || item.transcription || "";
          })
          .filter((text: string) => text.trim().length > 0)
          .join(" ");
        if (fullText) {
          console.log(
            `Combined array items into full transcription (${fullText.length} chars)`
          );
          return fullText;
        }
      }

      // Handle chunked results - if chunks exist, combine them
      if (result.chunks && Array.isArray(result.chunks)) {
        console.log(`Found ${result.chunks.length} chunks in result.chunks`);
        const fullText = result.chunks
          .map((chunk: any) => {
            // Handle different chunk structures
            if (typeof chunk === "string") return chunk;
            return chunk.text || chunk.transcription || "";
          })
          .filter((text: string) => text.trim().length > 0)
          .join(" ");
        if (fullText) {
          console.log(
            `Combined ${result.chunks.length} chunks into full transcription (${fullText.length} chars)`
          );
          return fullText;
        }
      }

      // Check for nested chunks structure
      if (result.text && Array.isArray(result.text)) {
        console.log(
          `Found array in result.text with ${result.text.length} items`
        );
        return result.text
          .map((item: any) =>
            typeof item === "string" ? item : item.text || ""
          )
          .filter((text: string) => text.trim().length > 0)
          .join(" ");
      }

      // Try different possible property names
      const text =
        result.text || result.transcription || result.chunks?.[0]?.text || "";
      if (text) {
        console.log(`Using result.text/transcription (${text.length} chars)`);
        return text;
      }
    }

    console.warn("Could not extract text from result");
    return "";
  }
}
