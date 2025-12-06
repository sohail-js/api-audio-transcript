import { serve } from "@hono/node-server";
import { Hono, Context } from "hono";
import { AudioService } from "./services/audio-service";
import { TranscriptionService } from "./services/transcription-service";
import { createTempFile, safeDeleteFile } from "./utils/file-utils";

// Initialize services
const audioService = new AudioService();
const transcriptionService = new TranscriptionService();

// Initialize transcriber at startup
transcriptionService.initialize().catch((error) => {
  console.error("Failed to initialize Whisper model:", error);
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
    tempFilePath = await createTempFile(audioFile);

    // Process audio file (decode/convert to Float32Array)
    const audioData = await audioService.processAudioFile(tempFilePath);
    console.log(`Audio data length: ${audioData.length} samples`);

    // Validate audio data
    if (!audioData || audioData.length === 0) {
      return c.json(
        {
          error:
            "Invalid audio data: audio file appears to be empty or corrupted",
        },
        400
      );
    }

    // Transcribe audio
    const text = await transcriptionService.transcribe(audioData);
    console.log(
      `Transcription result: "${text}" (length: ${text.length} characters)`
    );

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
      await safeDeleteFile(tempFilePath);
    }
  }
});

// Health check endpoint
app.get("/", (c: Context) => {
  return c.json({ message: "Whisper Transcription API", status: "ok" });
});

const port = 3001;
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});
