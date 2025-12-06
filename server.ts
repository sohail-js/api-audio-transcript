import { serve } from "@hono/node-server";
import { Hono, Context } from "hono";
import { TranscriptionService } from "./services/transcription-service.js";
import { writeFile, unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// Initialize service
const transcriptionService = new TranscriptionService();

// Initialize at startup
transcriptionService.initialize().catch((error) => {
  console.error("Failed to initialize Whisper:", error);
  process.exit(1);
});

const app = new Hono();

// POST /transcribe endpoint
app.post("/transcribe", async (c: Context) => {
  let tempFilePath: string | null = null;
  const startTime = Date.now();

  try {
    // Parse and validate request
    const formData = await c.req.formData();
    const audioFile = formData.get("audio");

    if (!audioFile) {
      return c.json({ error: "No audio file provided" }, 400);
    }

    if (typeof audioFile === "string") {
      return c.json({ error: "Invalid audio file format" }, 400);
    }

    // Save uploaded file to temporary location
    const buffer = await audioFile.arrayBuffer();
    const fileName = `whisper-${Date.now()}.${getFileExtension(audioFile.name) || "mp3"}`;
    tempFilePath = join(tmpdir(), fileName);
    
    await writeFile(tempFilePath, Buffer.from(buffer));
    console.log(`Saved temp file: ${tempFilePath}`);

    // Transcribe audio
    const text = await transcriptionService.transcribe(tempFilePath);

    // Calculate processing time
    const processingTimeMs = Date.now() - startTime;
    const processingTimeSeconds = (processingTimeMs / 1000).toFixed(2);

    return c.json({
      text,
      processingTimeSeconds: parseFloat(processingTimeSeconds),
      processingTimeMs,
    });
  } catch (error) {
    console.error("Transcription error:", error);
    return c.json(
      {
        error: "Failed to transcribe audio",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      500
    );
  } finally {
    // Clean up temporary file
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
        console.log(`Cleaned up temp file: ${tempFilePath}`);
      } catch (err) {
        console.warn(`Failed to delete temp file: ${tempFilePath}`);
      }
    }
  }
});

// Health check endpoint
app.get("/", (c: Context) => {
  return c.json({ 
    message: "Whisper Transcription API", 
    status: "ok",
    model: "base",
    languages: "Auto-detect (Hindi, Urdu, English, 99+ more)"
  });
});

/**
 * Extract file extension from filename
 */
function getFileExtension(filename: string): string | null {
  const parts = filename.split(".");
  return parts.length > 1 ? parts[parts.length - 1] : null;
}

const port = 3001;
console.log(`Server is running on port ${port}`);
console.log(`Whisper model: base (multilingual, auto-detect)`);

serve({
  fetch: app.fetch,
  port,
});
