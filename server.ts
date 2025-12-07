// Load environment variables from .env file
import "dotenv/config";

import { serve } from "@hono/node-server";
import { Hono, Context } from "hono";
import { TranscriptionService } from "./services/transcription-service.js";
import { unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import logger, { createChildLogger } from "./utils/logger.js";

// Initialize service
let transcriptionService: TranscriptionService;

try {
  transcriptionService = new TranscriptionService();
  // Initialize at startup
  transcriptionService.initialize().catch((error) => {
    logger.error(
      { error: error.message, stack: error.stack },
      "Failed to initialize OpenAI Whisper service"
    );
    process.exit(1);
  });
} catch (error) {
  logger.error(
    { error: error instanceof Error ? error.message : "Unknown error" },
    "Failed to create TranscriptionService"
  );
  logger.error("Please ensure OPENAI_API_KEY environment variable is set");
  process.exit(1);
}

const app = new Hono();

// POST /transcribe endpoint
app.post("/transcribe", async (c: Context) => {
  let tempFilePath: string | null = null;
  const startTime = Date.now();
  const requestId = `req-${Date.now()}-${Math.random()
    .toString(36)
    .substring(2, 7)}`;
  const requestLogger = createChildLogger({ requestId });

  try {
    requestLogger.info("Transcription request started");

    // Check Content-Type
    const contentType = c.req.header("content-type");
    if (!contentType || !contentType.includes("multipart/form-data")) {
      requestLogger.warn("Invalid Content-Type");
      return c.json({ error: "Content-Type must be multipart/form-data" }, 400);
    }

    // Use OS temporary directory for file storage
    const tempDir = tmpdir();
    const uniqueId = `${Date.now()}-${Math.random()
      .toString(36)
      .substring(2, 9)}`;

    // Use Busboy for streaming file upload
    const { default: Busboy } = await import("busboy");
    const { Readable } = await import("stream");
    const { createWriteStream } = await import("fs");

    // Create a promise to handle the file upload and form fields
    const uploadPromise = new Promise<{
      filePath: string;
      filename: string;
      prompt?: string;
    }>((resolve, reject) => {
      try {
        const bb = Busboy({ headers: { "content-type": contentType } });
        let fileFound = false;
        let prompt: string | undefined;
        let fileWriteFinished = false;
        let busboyFinished = false;
        let resolved = false;
        let filename: string | undefined;

        const tryResolve = () => {
          if (resolved) return;
          if (
            fileWriteFinished &&
            busboyFinished &&
            fileFound &&
            filename &&
            tempFilePath
          ) {
            resolved = true;
            resolve({ filePath: tempFilePath, filename, prompt });
          }
        };

        bb.on("file", (name, file, info) => {
          const fileInfo = info;
          filename = fileInfo.filename;
          requestLogger.debug({ field: name, filename }, "Received file field");

          if (name !== "audio") {
            requestLogger.debug({ field: name }, "Skipping non-audio field");
            file.resume(); // Skip non-audio fields
            return;
          }

          fileFound = true;
          const ext = getFileExtension(filename) || "mp3";
          const tempFile = join(tempDir, `transcribe-${uniqueId}.${ext}`);
          tempFilePath = tempFile;

          requestLogger.info(
            { tempFilePath: tempFile, filename },
            "Streaming upload to temporary file"
          );
          const writeStream = createWriteStream(tempFile);

          file.pipe(writeStream);

          writeStream.on("finish", () => {
            requestLogger.info(
              { tempFilePath: tempFile },
              "File write completed"
            );
            fileWriteFinished = true;
            tryResolve();
          });

          writeStream.on("error", (err: Error) => {
            requestLogger.error(
              { error: err.message, stack: err.stack },
              "File write error"
            );
            reject(err);
          });
        });

        bb.on("field", (name, value) => {
          if (name === "prompt") {
            prompt = value;
            requestLogger.debug(
              { promptLength: value.length },
              "Received prompt field"
            );
          } else {
            requestLogger.debug({ field: name }, "Received non-prompt field");
          }
        });

        bb.on("error", (err: Error) => {
          requestLogger.error({ error: err.message }, "Busboy error");
          reject(err);
        });

        bb.on("finish", () => {
          requestLogger.debug("Busboy parsing finished");
          busboyFinished = true;
          if (!fileFound) {
            reject(new Error("No audio file found in request"));
          } else {
            tryResolve();
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
        requestLogger.error(
          { error: err instanceof Error ? err.message : "Unknown error" },
          "Upload promise error"
        );
        reject(err);
      }
    });

    // Wait for upload to complete
    const { filePath, prompt } = await uploadPromise;
    requestLogger.info({ filePath }, "Upload complete");

    // Check for diarize query parameter
    const diarizeParam = c.req.query("diarize");
    const useDiarize =
      diarizeParam === "true" || diarizeParam === "1" || diarizeParam === "yes";

    if (useDiarize) {
      requestLogger.info("Speaker diarization enabled via query parameter");
    }

    // Transcribe audio (pass requestId for logging context)
    const text = await transcriptionService.transcribe(
      filePath,
      undefined,
      useDiarize,
      requestId
    );

    // Get text generation model from environment or use default
    const textGenerationModel =
      process.env.OPENAI_TEXT_GENERATION_MODEL || "gpt-4o-mini";

    // Generate text from transcript if prompt is provided
    let generatedText: string | undefined;
    if (prompt && prompt.trim()) {
      requestLogger.info(
        { promptLength: prompt.length },
        "Prompt provided, generating text from transcript"
      );
      try {
        generatedText = await transcriptionService.generateTextFromTranscript(
          text,
          prompt,
          textGenerationModel,
          requestId
        );
        requestLogger.info(
          { generatedTextLength: generatedText.length },
          "Text generation completed"
        );
      } catch (error) {
        requestLogger.error(
          {
            error: error instanceof Error ? error.message : "Unknown error",
            stack: error instanceof Error ? error.stack : undefined,
          },
          "Text generation failed, returning transcript only"
        );
        // Continue with just the transcript if text generation fails
      }
    }

    // Calculate processing time
    const processingTimeMs = Date.now() - startTime;
    const processingTimeSeconds = (processingTimeMs / 1000).toFixed(2);

    requestLogger.info(
      {
        processingTimeMs,
        processingTimeSeconds: parseFloat(processingTimeSeconds),
        textLength: text.length,
        model: useDiarize
          ? "gpt-4o-transcribe-diarize"
          : "gpt-4o-mini-transcribe",
        hasGeneratedText: !!generatedText,
      },
      "Transcription completed successfully"
    );

    // Clean up temporary file
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
        requestLogger.debug({ tempFilePath }, "Temporary file cleaned up");
      } catch (cleanupError) {
        requestLogger.warn(
          {
            error:
              cleanupError instanceof Error
                ? cleanupError.message
                : "Unknown error",
            tempFilePath,
          },
          "Failed to clean up temporary file"
        );
      }
    }

    // Build response object
    const response: any = {
      text,
      processingTimeSeconds: parseFloat(processingTimeSeconds),
      processingTimeMs,
      requestId,
      model: useDiarize
        ? "gpt-4o-transcribe-diarize"
        : "gpt-4o-mini-transcribe",
      diarize: useDiarize,
    };

    // Add generated text and model info if prompt was provided
    if (generatedText) {
      response.generatedText = generatedText;
      response.textGenerationModel = textGenerationModel;
    }

    return c.json(response);
  } catch (error) {
    requestLogger.error(
      {
        error: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : undefined,
      },
      "Transcription error"
    );

    // Clean up temporary file on error
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
        requestLogger.debug(
          { tempFilePath },
          "Temporary file cleaned up after error"
        );
      } catch (cleanupError) {
        requestLogger.warn(
          {
            error:
              cleanupError instanceof Error
                ? cleanupError.message
                : "Unknown error",
            tempFilePath,
          },
          "Failed to clean up temporary file after error"
        );
      }
    }

    return c.json(
      {
        error: "Failed to transcribe audio",
        details: error instanceof Error ? error.message : "Unknown error",
        requestId,
      },
      500
    );
  }
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
logger.info({ port }, "Server starting");
logger.info("Using OpenAI Whisper API (gpt-4o-mini-transcribe model)");
logger.info("Make sure OPENAI_API_KEY environment variable is set");

serve({
  fetch: app.fetch,
  port,
});
