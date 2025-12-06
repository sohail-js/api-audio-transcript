import { WaveFile } from "wavefile";

/**
 * Converts stereo audio samples to mono by averaging channels
 */
export function convertStereoToMono(samples: Float32Array[]): Float32Array {
  if (samples.length === 1) {
    return samples[0];
  }

  const mono = new Float32Array(samples[0].length);
  for (let i = 0; i < samples[0].length; i++) {
    mono[i] = (samples[0][i] + samples[1][i]) / 2;
  }
  return mono;
}

/**
 * Normalizes audio samples to Float32Array
 */
export function normalizeToFloat32Array(samples: Float32Array | Float32Array[]): Float32Array {
  if (Array.isArray(samples)) {
    return convertStereoToMono(samples);
  }
  return samples instanceof Float32Array ? samples : new Float32Array(samples);
}

/**
 * Extracts file extension from file path
 */
export function getFileExtension(filePath: string): string {
  return filePath.split(".").pop()?.toLowerCase() || "";
}

/**
 * Checks if a file format is WAV
 */
export function isWavFile(filePath: string): boolean {
  return getFileExtension(filePath) === "wav";
}

