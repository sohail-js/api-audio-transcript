import { readFile } from "fs/promises";
import ffmpeg from "fluent-ffmpeg";
import wavefile from "wavefile";
const { WaveFile } = wavefile;
import { normalizeToFloat32Array, isWavFile } from "../utils/audio-utils";
import { safeDeleteFile, generateTempPath } from "../utils/file-utils";

const WHISPER_SAMPLE_RATE = 16000;
const WHISPER_BIT_DEPTH = "32f";

/**
 * Service for processing audio files
 */
export class AudioService {
  /**
   * Decodes a WAV file to Float32Array
   */
  private async decodeWavFile(filePath: string): Promise<Float32Array> {
    const fileBuffer = await readFile(filePath);
    const wav = new WaveFile(fileBuffer);
    
    // Convert to Whisper's expected format: 32-bit float, 16kHz
    wav.toBitDepth(WHISPER_BIT_DEPTH);
    wav.toSampleRate(WHISPER_SAMPLE_RATE);
    
    const samples = wav.getSamples(false, Float32Array);
    // Handle potential Float64Array return type by converting to Float32Array
    const normalizedSamples: Float32Array | Float32Array[] = Array.isArray(samples)
      ? samples
      : samples instanceof Float32Array
      ? samples
      : new Float32Array(samples);
    return normalizeToFloat32Array(normalizedSamples);
  }

  /**
   * Converts a non-WAV audio file to WAV format using ffmpeg
   */
  private async convertToWav(inputPath: string, outputPath: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      ffmpeg(inputPath)
        .audioFrequency(WHISPER_SAMPLE_RATE)
        .audioChannels(1) // Mono
        .audioCodec("pcm_f32le") // 32-bit float PCM
        .format("wav")
        .on("end", () => resolve())
        .on("error", (err: Error) => reject(err))
        .save(outputPath);
    });
  }

  /**
   * Processes an audio file and returns Float32Array samples
   * Handles both WAV and non-WAV formats
   */
  async processAudioFile(filePath: string): Promise<Float32Array> {
    if (isWavFile(filePath)) {
      return this.decodeWavFile(filePath);
    }

    // For non-WAV files, convert to WAV first
    const convertedWavPath = generateTempPath("wav", "converted");

    try {
      await this.convertToWav(filePath, convertedWavPath);
      const audioData = await this.decodeWavFile(convertedWavPath);
      return audioData;
    } catch (error) {
      throw new Error(
        `Failed to convert audio file: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      // Clean up converted file
      await safeDeleteFile(convertedWavPath);
    }
  }
}

