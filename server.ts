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
// POST /transcribe endpoint
app.post("/transcribe", async (c: Context) => {
  let tempFilePath: string | null = null;
  const startTime = Date.now();

  try {
    // Check Content-Type
    const contentType = c.req.header("content-type");
    if (!contentType || !contentType.includes("multipart/form-data")) {
      return c.json({ error: "Content-Type must be multipart/form-data" }, 400);
    }

    // Use Busboy for streaming file upload (handles large files better than formData())
    const { default: Busboy } = await import("busboy");
    const { Readable } = await import("stream");
    const { createWriteStream } = await import("fs");
    
    // Create a promise to handle the file upload
    const uploadPromise = new Promise<{ filePath: string; filename: string }>((resolve, reject) => {
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
          const tempName = `whisper-${Date.now()}.${ext}`;
          const savePath = join(tmpdir(), tempName);
          tempFilePath = savePath; // Save ref for cleanup

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
          // If fileFound is true, we wait for writeStream.on("finish") to resolve
        });

        // Convert Web Stream to Node Stream and pipe to Busboy
        if (c.req.raw.body) {
          // @ts-ignore - Readable.fromWeb is available in Node 18+
          const nodeStream = Readable.fromWeb(c.req.raw.body);
          nodeStream.pipe(bb);
        } else {
          reject(new Error("Request body is empty"));
        }
      } catch (err) {
        reject(err);
      }
    });

    // Wait for upload to complete
    const { filePath } = await uploadPromise;
    console.log(`Upload complete: ${filePath}`);

    // Transcribe audio
    const text = await transcriptionService.transcribe(filePath);

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
