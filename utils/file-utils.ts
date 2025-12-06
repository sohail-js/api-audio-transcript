import { writeFile, unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

/**
 * Creates a temporary file from a File object
 */
export async function createTempFile(
  file: File,
  prefix: string = "audio"
): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const fileExtension = file.name.split(".").pop() || "mp3";
  const tempFilePath = join(
    tmpdir(),
    `${prefix}-${Date.now()}-${Math.random().toString(36).substring(7)}.${fileExtension}`
  );

  await writeFile(tempFilePath, buffer);
  return tempFilePath;
}

/**
 * Safely deletes a file, ignoring errors
 */
export async function safeDeleteFile(filePath: string): Promise<void> {
  try {
    await unlink(filePath);
  } catch (error) {
    // Ignore errors during cleanup
    console.error(`Failed to delete file ${filePath}:`, error);
  }
}

/**
 * Generates a temporary file path with the given extension
 */
export function generateTempPath(extension: string, prefix: string = "temp"): string {
  return join(
    tmpdir(),
    `${prefix}-${Date.now()}-${Math.random().toString(36).substring(7)}.${extension}`
  );
}

