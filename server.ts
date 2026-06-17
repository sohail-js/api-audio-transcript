// Load environment variables from .env file
import "dotenv/config";

import { serve } from "@hono/node-server";
import { Hono, Context } from "hono";
import {
  TranscriptionService,
  type TranscriptionProgress,
} from "./services/transcription-service.js";
import { unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import logger, { createChildLogger } from "./utils/logger.js";

type UploadedAudio = {
  filePath: string;
  filename: string;
  prompt?: string;
};

type TranscriptionResult = {
  Completed: string;
  completed: number;
  text: string;
  generatedText?: string;
  processingTimeMs: number;
  processingTimeSeconds: number;
  requestId: string;
  model: string;
  diarize: boolean;
  accurate: boolean;
  textGenerationModel?: string;
};

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
  if (isTruthy(c.req.query("stream"))) {
    return streamTranscription(c);
  }

  let tempFilePath: string | null = null;
  const startTime = Date.now();
  const requestId = createRequestId();
  const requestLogger = createChildLogger({ requestId });

  try {
    requestLogger.info("Transcription request started");

    if (!isMultipartRequest(c)) {
      requestLogger.warn("Invalid Content-Type");
      return c.json({ error: "Content-Type must be multipart/form-data" }, 400);
    }

    const upload = await parseMultipartUpload(c, requestId, requestLogger, {
      onTempFile: (filePath) => {
        tempFilePath = filePath;
      },
    });
    requestLogger.info({ filePath: upload.filePath }, "Upload complete");

    const result = await runTranscription(
      c,
      upload,
      requestId,
      startTime,
      requestLogger
    );

    await cleanupTempFile(tempFilePath, requestLogger);
    tempFilePath = null;

    return c.json(result);
  } catch (error) {
    requestLogger.error(
      {
        error: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : undefined,
      },
      "Transcription error"
    );

    await cleanupTempFile(tempFilePath, requestLogger, true);

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

// POST /transcribe/stream endpoint
app.post("/transcribe/stream", streamTranscription);

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

function createRequestId(): string {
  return `req-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
}

function isTruthy(value: string | undefined): boolean {
  return value === "true" || value === "1" || value === "yes";
}

function isMultipartRequest(c: Context): boolean {
  const contentType = c.req.header("content-type");
  return !!contentType && contentType.includes("multipart/form-data");
}

function getSelectedModel(useDiarize: boolean, useHighAccuracy: boolean): string {
  if (useDiarize) {
    return "gpt-4o-transcribe-diarize";
  }

  if (useHighAccuracy) {
    return "gpt-4o-transcribe";
  }

  return "gpt-4o-mini-transcribe";
}

async function parseMultipartUpload(
  c: Context,
  requestId: string,
  requestLogger: ReturnType<typeof createChildLogger>,
  options?: { onTempFile?: (filePath: string) => void }
): Promise<UploadedAudio> {
  const contentType = c.req.header("content-type");
  if (!contentType || !contentType.includes("multipart/form-data")) {
    throw new Error("Content-Type must be multipart/form-data");
  }

  const tempDir = tmpdir();
  const uniqueId = `${requestId}-${Math.random().toString(36).substring(2, 9)}`;
  const { default: Busboy } = await import("busboy");
  const { Readable } = await import("stream");
  const { createWriteStream } = await import("fs");

  return new Promise<UploadedAudio>((resolve, reject) => {
    try {
      const bb = Busboy({ headers: { "content-type": contentType } });
      let fileFound = false;
      let prompt: string | undefined;
      let fileWriteFinished = false;
      let busboyFinished = false;
      let resolved = false;
      let filename: string | undefined;
      let tempFilePath: string | undefined;

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
        filename = info.filename;
        requestLogger.debug({ field: name, filename }, "Received file field");

        if (name !== "audio") {
          requestLogger.debug({ field: name }, "Skipping non-audio field");
          file.resume();
          return;
        }

        fileFound = true;
        const ext = getFileExtension(filename) || "mp3";
        tempFilePath = join(tempDir, `transcribe-${uniqueId}.${ext}`);
        options?.onTempFile?.(tempFilePath);

        requestLogger.info(
          { tempFilePath, filename },
          "Streaming upload to temporary file"
        );
        const writeStream = createWriteStream(tempFilePath);

        file.pipe(writeStream);

        writeStream.on("finish", () => {
          requestLogger.info({ tempFilePath }, "File write completed");
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
        const nodeStream = Readable.fromWeb(c.req.raw.body as any);
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
}

async function runTranscription(
  c: Context,
  upload: UploadedAudio,
  requestId: string,
  startTime: number,
  requestLogger: ReturnType<typeof createChildLogger>,
  onProgress?: (progress: TranscriptionProgress) => void
): Promise<TranscriptionResult> {
  const useDiarize = isTruthy(c.req.query("diarize"));
  const useHighAccuracy = isTruthy(c.req.query("accurate"));

  if (useDiarize) {
    requestLogger.info("Speaker diarization enabled via query parameter");
  }

  if (useHighAccuracy) {
    requestLogger.info(
      "High accuracy model (gpt-4o-transcribe) enabled via query parameter"
    );
  }

  const text = await transcriptionService.transcribe(
    upload.filePath,
    undefined,
    useDiarize,
    requestId,
    useHighAccuracy,
    onProgress
  );

  const textGenerationModel =
    process.env.OPENAI_TEXT_GENERATION_MODEL || "gpt-4o-mini";
  let generatedText: string | undefined;

  if (upload.prompt && upload.prompt.trim()) {
    requestLogger.info(
      { promptLength: upload.prompt.length },
      "Prompt provided, generating text from transcript"
    );
    onProgress?.({
      completed: 95,
      stage: "generating",
      message: "Generating text from transcript",
    });

    try {
      generatedText = await transcriptionService.generateTextFromTranscript(
        text,
        upload.prompt,
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
    }
  }

  const processingTimeMs = Date.now() - startTime;
  const processingTimeSeconds = parseFloat((processingTimeMs / 1000).toFixed(2));
  const model = getSelectedModel(useDiarize, useHighAccuracy);

  requestLogger.info(
    {
      processingTimeMs,
      processingTimeSeconds,
      textLength: text.length,
      model,
      hasGeneratedText: !!generatedText,
    },
    "Transcription completed successfully"
  );

  const result: TranscriptionResult = {
    Completed: "100%",
    completed: 100,
    text,
    processingTimeSeconds,
    processingTimeMs,
    requestId,
    model,
    diarize: useDiarize,
    accurate: useHighAccuracy,
  };

  if (generatedText) {
    result.generatedText = generatedText;
    result.textGenerationModel = textGenerationModel;
  }

  return result;
}

async function cleanupTempFile(
  tempFilePath: string | null,
  requestLogger: ReturnType<typeof createChildLogger>,
  afterError: boolean = false
): Promise<void> {
  if (!tempFilePath) {
    return;
  }

  try {
    await unlink(tempFilePath);
    requestLogger.debug(
      { tempFilePath },
      afterError
        ? "Temporary file cleaned up after error"
        : "Temporary file cleaned up"
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
      afterError
        ? "Failed to clean up temporary file after error"
        : "Failed to clean up temporary file"
    );
  }
}

function progressPayload(progress: TranscriptionProgress) {
  return {
    Completed: `${progress.completed}%`,
    ...progress,
  };
}

function streamTranscription(c: Context): Response {
  const requestId = createRequestId();
  const requestLogger = createChildLogger({ requestId });
  const startTime = Date.now();
  const encoder = new TextEncoder();
  let tempFilePath: string | null = null;
  let lastCompleted = 0;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: string, data: Record<string, unknown>) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        );
      };

      const sendProgress = (progress: TranscriptionProgress) => {
        const completed = Math.max(progress.completed, lastCompleted);
        lastCompleted = completed;
        send("progress", {
          ...progressPayload({ ...progress, completed }),
          requestId,
        });
      };

      const run = async () => {
        try {
          requestLogger.info("Streaming transcription request started");

          if (!isMultipartRequest(c)) {
            send("error", {
              error: "Content-Type must be multipart/form-data",
              requestId,
            });
            return;
          }

          sendProgress({
            completed: 1,
            stage: "uploading",
            message: "Receiving audio upload",
          });

          const upload = await parseMultipartUpload(c, requestId, requestLogger, {
            onTempFile: (filePath) => {
              tempFilePath = filePath;
            },
          });

          sendProgress({
            completed: 15,
            stage: "uploaded",
            message: "Audio upload completed",
          });

          const result = await runTranscription(
            c,
            upload,
            requestId,
            startTime,
            requestLogger,
            (progress) => {
              sendProgress(progress);
            }
          );

          await cleanupTempFile(tempFilePath, requestLogger);
          tempFilePath = null;

          send("completed", result);
        } catch (error) {
          requestLogger.error(
            {
              error: error instanceof Error ? error.message : "Unknown error",
              stack: error instanceof Error ? error.stack : undefined,
            },
            "Streaming transcription error"
          );

          await cleanupTempFile(tempFilePath, requestLogger, true);

          send("error", {
            error: "Failed to transcribe audio",
            details: error instanceof Error ? error.message : "Unknown error",
            requestId,
          });
        } finally {
          controller.close();
        }
      };

      run();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

const port = 3001;
logger.info({ port }, "Server starting");
logger.info("Using OpenAI Whisper API (gpt-4o-mini-transcribe model)");
logger.info("Make sure OPENAI_API_KEY environment variable is set");

serve({
  fetch: app.fetch,
  port,
});
