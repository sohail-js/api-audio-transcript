// Load environment variables from .env file
import "dotenv/config";

import { serve } from "@hono/node-server";
import { Hono, Context } from "hono";
import { TranscriptionService } from "./services/transcription-service.js";
import { mkdir } from "fs/promises";
import { join } from "path";

// Initialize service
let transcriptionService: TranscriptionService;

try {
  transcriptionService = new TranscriptionService();
  // Initialize at startup
  transcriptionService.initialize().catch((error) => {
    console.error("Failed to initialize OpenAI Whisper service:", error);
    process.exit(1);
  });
} catch (error) {
  console.error("Failed to create TranscriptionService:", error);
  console.error("Please ensure OPENAI_API_KEY environment variable is set");
  process.exit(1);
}

const app = new Hono();

// POST /transcribe endpoint
app.post("/transcribe", async (c: Context) => {
  let requestDir: string | null = null;
  const startTime = Date.now();

  try {
    // Check Content-Type
    const contentType = c.req.header("content-type");
    if (!contentType || !contentType.includes("multipart/form-data")) {
      return c.json({ error: "Content-Type must be multipart/form-data" }, 400);
    }

    // Prepare requests directory
    const projectRoot = process.cwd();
    const requestsBaseDir = join(projectRoot, "requests");
    const requestId = `req-${Date.now()}-${Math.random()
      .toString(36)
      .substring(2, 7)}`;
    requestDir = join(requestsBaseDir, requestId);

    await mkdir(requestDir, { recursive: true });
    console.log(`Created request directory: ${requestDir}`);

    // Use Busboy for streaming file upload
    const { default: Busboy } = await import("busboy");
    const { Readable } = await import("stream");
    const { createWriteStream } = await import("fs");

    // Create a promise to handle the file upload
    const uploadPromise = new Promise<{ filePath: string; filename: string }>(
      (resolve, reject) => {
        try {
          const bb = Busboy({ headers: { "content-type": contentType } });
          let fileFound = false;

          bb.on("file", (name, file, info) => {
            const { filename } = info;
            console.log(`Received file field: ${name}, filename: ${filename}`);

            if (name !== "audio") {
              console.log(`Skipping field: ${name}`);
              file.resume(); // Skip non-audio fields
              return;
            }

            fileFound = true;
            const ext = getFileExtension(filename) || "mp3";
            const savePath = join(requestDir!, `original.${ext}`);

            console.log(`Streaming upload to: ${savePath}`);
            const writeStream = createWriteStream(savePath);

            file.pipe(writeStream);

            writeStream.on("finish", () => {
              console.log("File write completed");
              resolve({ filePath: savePath, filename });
            });

            writeStream.on("error", (err) => {
              console.error("File write error:", err);
              reject(err);
            });
          });

          bb.on("error", (err) => {
            console.error("Busboy error:", err);
            reject(err);
          });

          bb.on("finish", () => {
            console.log("Busboy parsing finished");
            if (!fileFound) {
              reject(new Error("No audio file found in request"));
            }
          });

          if (c.req.raw.body) {
            // @ts-ignore
            const nodeStream = Readable.fromWeb(c.req.raw.body);
            nodeStream.pipe(bb);
          } else {
            reject(new Error("Request body is empty"));
          }
        } catch (err) {
          reject(err);
        }
      }
    );

    // Wait for upload to complete
    const { filePath } = await uploadPromise;
    console.log(`Upload complete: ${filePath}`);

    // Check for diarize query parameter
    const diarizeParam = c.req.query("diarize");
    const useDiarize =
      diarizeParam === "true" || diarizeParam === "1" || diarizeParam === "yes";

    if (useDiarize) {
      console.log("Speaker diarization enabled via query parameter");
    }

    // Transcribe audio
    const text = await transcriptionService.transcribe(
      filePath,
      undefined,
      useDiarize
    );

    // Calculate processing time
    const processingTimeMs = Date.now() - startTime;
    const processingTimeSeconds = (processingTimeMs / 1000).toFixed(2);

    return c.json({
      text,
      processingTimeSeconds: parseFloat(processingTimeSeconds),
      processingTimeMs,
      requestId,
      debugPath: requestDir,
      model: useDiarize
        ? "gpt-4o-transcribe-diarize"
        : "gpt-4o-mini-transcribe",
      diarize: useDiarize,
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
  }
  // NOTE: Cleanup removed for debugging purposes as requested
});

// Health check endpoint
app.get("/", (c: Context) => {
  return c.json({
    message: "OpenAI Whisper Transcription API",
    status: "ok",
    model: "gpt-4o-mini-transcribe",
    provider: "OpenAI",
    languages: "Auto-detect (Hindi, Urdu, English, 99+ more)",
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
console.log(`Using OpenAI Whisper API (gpt-4o-mini-transcribe model)`);
console.log(`Make sure OPENAI_API_KEY environment variable is set`);

serve({
  fetch: app.fetch,
  port,
});
