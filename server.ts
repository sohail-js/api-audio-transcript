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
  language?: string;
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

// Simple browser UI
app.get("/", (c: Context) => {
  return c.html(uiHtml);
});

// Health check endpoint
app.get("/health", (c: Context) => {
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
  const language = c.req.query("language")?.trim() || undefined;

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
    language,
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

  if (language) {
    result.language = language;
  }

  if (generatedText) {
    result.generatedText = generatedText;
    result.textGenerationModel = textGenerationModel;
  }

  return result;
}

const uiHtml = String.raw`<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Audio Transcription</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f7f8fa;
        --panel: #ffffff;
        --text: #172033;
        --muted: #627084;
        --line: #d9dee8;
        --accent: #1f6feb;
        --accent-dark: #1557b0;
        --ok: #138a43;
        --danger: #b42318;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background: var(--bg);
        color: var(--text);
        font-family: Arial, Helvetica, sans-serif;
      }

      main {
        width: min(960px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0;
      }

      header {
        margin-bottom: 20px;
      }

      h1 {
        margin: 0 0 8px;
        font-size: clamp(28px, 5vw, 42px);
        line-height: 1.05;
        letter-spacing: 0;
      }

      p {
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 360px) minmax(0, 1fr);
        gap: 18px;
        align-items: start;
      }

      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
      }

      label,
      legend {
        display: block;
        margin-bottom: 8px;
        font-weight: 700;
        font-size: 14px;
      }

      input[type="file"],
      select,
      textarea {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fff;
        color: var(--text);
        font: inherit;
      }

      input[type="file"],
      select {
        min-height: 42px;
        padding: 9px 10px;
      }

      textarea {
        min-height: 124px;
        padding: 10px;
        resize: vertical;
        line-height: 1.5;
      }

      .field {
        margin-bottom: 16px;
      }

      fieldset {
        border: 0;
        padding: 0;
        margin: 0 0 16px;
      }

      .check {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        margin: 10px 0;
        color: var(--text);
      }

      .check input {
        margin-top: 2px;
      }

      .check span {
        color: var(--muted);
        display: block;
        font-size: 13px;
        margin-top: 2px;
      }

      button {
        border: 0;
        border-radius: 6px;
        background: var(--accent);
        color: #fff;
        cursor: pointer;
        font: inherit;
        font-weight: 700;
        min-height: 42px;
        padding: 10px 14px;
      }

      button:hover:not(:disabled) {
        background: var(--accent-dark);
      }

      button:disabled {
        cursor: not-allowed;
        opacity: 0.65;
      }

      .secondary {
        background: #eef2f8;
        color: var(--text);
      }

      .secondary:hover:not(:disabled) {
        background: #dfe6f1;
      }

      .actions {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }

      .progress-wrap {
        margin-bottom: 16px;
      }

      .progress-meta {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 14px;
      }

      .progress {
        height: 12px;
        overflow: hidden;
        border-radius: 999px;
        background: #e8edf5;
      }

      .bar {
        width: 0%;
        height: 100%;
        background: var(--accent);
        transition: width 180ms ease;
      }

      .status {
        min-height: 22px;
        margin-bottom: 14px;
        color: var(--muted);
      }

      .status.error {
        color: var(--danger);
      }

      .status.done {
        color: var(--ok);
      }

      .result-heading {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
        margin-bottom: 8px;
      }

      .result-heading h2 {
        margin: 0;
        font-size: 18px;
      }

      #output {
        min-height: 360px;
      }

      .details {
        margin-top: 10px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
      }

      @media (max-width: 760px) {
        main {
          width: min(100% - 24px, 960px);
          padding: 20px 0;
        }

        .layout {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>Audio Transcription</h1>
        <p>Upload an audio file, choose the options you need, and watch the transcription progress.</p>
      </header>

      <div class="layout">
        <form id="form" class="panel">
          <div class="field">
            <label for="audio">Audio file</label>
            <input id="audio" name="audio" type="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.webm,.opus" required />
          </div>

          <div class="field">
            <label for="language">Language</label>
            <select id="language" name="language">
              <option value="">Auto detect</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="ur">Urdu</option>
              <option value="ar">Arabic</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
            </select>
          </div>

          <fieldset>
            <legend>Model options</legend>
            <label class="check">
              <input id="accurate" type="checkbox" />
              <span><strong>High accuracy</strong><br />Use gpt-4o-transcribe for noisy or complex audio.</span>
            </label>
            <label class="check">
              <input id="diarize" type="checkbox" />
              <span><strong>Speaker diarization</strong><br />Label multiple speakers. This takes priority over high accuracy.</span>
            </label>
          </fieldset>

          <div class="field">
            <label for="prompt">Optional follow-up prompt</label>
            <textarea id="prompt" name="prompt" placeholder="Example: Summarize the transcript as meeting notes."></textarea>
          </div>

          <div class="actions">
            <button id="submit" type="submit">Transcribe</button>
            <button id="reset" class="secondary" type="button">Reset</button>
          </div>
        </form>

        <section class="panel">
          <div class="progress-wrap">
            <div class="progress-meta">
              <span id="stage">Waiting for upload</span>
              <strong id="percent">0%</strong>
            </div>
            <div class="progress" aria-label="Transcription progress">
              <div id="bar" class="bar"></div>
            </div>
          </div>

          <div id="status" class="status">Choose an audio file to begin.</div>

          <div class="result-heading">
            <h2>Transcript</h2>
            <button id="copy" class="secondary" type="button" disabled>Copy</button>
          </div>
          <textarea id="output" readonly placeholder="The final transcript will appear here."></textarea>
          <div id="details" class="details"></div>
        </section>
      </div>
    </main>

    <script>
      const form = document.getElementById("form");
      const audio = document.getElementById("audio");
      const language = document.getElementById("language");
      const accurate = document.getElementById("accurate");
      const diarize = document.getElementById("diarize");
      const promptInput = document.getElementById("prompt");
      const submit = document.getElementById("submit");
      const reset = document.getElementById("reset");
      const copy = document.getElementById("copy");
      const bar = document.getElementById("bar");
      const percent = document.getElementById("percent");
      const stage = document.getElementById("stage");
      const status = document.getElementById("status");
      const output = document.getElementById("output");
      const details = document.getElementById("details");

      function setProgress(value, label, message) {
        const completed = Math.max(0, Math.min(100, Number(value) || 0));
        bar.style.width = completed + "%";
        percent.textContent = Math.round(completed) + "%";
        if (label) stage.textContent = label;
        if (message) status.textContent = message;
      }

      function setBusy(isBusy) {
        submit.disabled = isBusy;
        audio.disabled = isBusy;
        language.disabled = isBusy;
        accurate.disabled = isBusy;
        diarize.disabled = isBusy;
        promptInput.disabled = isBusy;
      }

      function resetOutput() {
        setProgress(0, "Waiting for upload", "Choose an audio file to begin.");
        status.className = "status";
        output.value = "";
        details.textContent = "";
        copy.disabled = true;
      }

      function parseSseBlock(block) {
        let event = "message";
        const dataLines = [];

        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }

        if (!dataLines.length) return null;
        return { event, data: JSON.parse(dataLines.join("\n")) };
      }

      async function handleStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";

          for (const block of blocks) {
            const message = parseSseBlock(block.trim());
            if (!message) continue;

            if (message.event === "progress") {
              const chunkInfo = message.data.totalChunks
                ? " (" + (message.data.currentChunk || 0) + "/" + message.data.totalChunks + " chunks)"
                : "";
              setProgress(
                message.data.completed,
                message.data.stage || "Working",
                (message.data.message || "Processing") + chunkInfo
              );

              if (message.data.chunkText) {
                const heading = "Chunk " + message.data.currentChunk + " of " + message.data.totalChunks;
                const separator = output.value.trim() ? "\n\n" : "";
                output.value += separator + "[" + heading + "]\n" + message.data.chunkText;
                copy.disabled = false;
              }
            }

            if (message.event === "completed") {
              setProgress(100, "Completed", "Transcription completed.");
              status.className = "status done";
              output.value = message.data.generatedText || message.data.text || "";
              copy.disabled = !output.value;

              const parts = [
                "Model: " + message.data.model,
                "Request: " + message.data.requestId,
                "Time: " + message.data.processingTimeSeconds + "s"
              ];
              if (message.data.language) parts.push("Language: " + message.data.language);
              if (message.data.generatedText) parts.push("Showing generated output from your prompt");
              details.textContent = parts.join(" | ");
            }

            if (message.event === "error") {
              throw new Error(message.data.details || message.data.error || "Transcription failed");
            }
          }
        }
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        resetOutput();

        if (!audio.files.length) {
          status.className = "status error";
          status.textContent = "Please choose an audio file first.";
          return;
        }

        const formData = new FormData();
        formData.append("audio", audio.files[0]);
        if (promptInput.value.trim()) formData.append("prompt", promptInput.value.trim());

        const params = new URLSearchParams();
        if (language.value) params.set("language", language.value);
        if (accurate.checked) params.set("accurate", "true");
        if (diarize.checked) params.set("diarize", "true");

        setBusy(true);
        setProgress(1, "Uploading", "Uploading audio file...");

        try {
          const url = "/transcribe/stream" + (params.toString() ? "?" + params.toString() : "");
          const response = await fetch(url, { method: "POST", body: formData });

          if (!response.ok || !response.body) {
            throw new Error("Server returned " + response.status + " " + response.statusText);
          }

          await handleStream(response);
        } catch (error) {
          status.className = "status error";
          status.textContent = error instanceof Error ? error.message : "Transcription failed.";
        } finally {
          setBusy(false);
        }
      });

      reset.addEventListener("click", () => {
        form.reset();
        resetOutput();
      });

      copy.addEventListener("click", async () => {
        await navigator.clipboard.writeText(output.value);
        const original = copy.textContent;
        copy.textContent = "Copied";
        setTimeout(() => {
          copy.textContent = original;
        }, 1200);
      });
    </script>
  </body>
</html>`;

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

const port = Number(process.env.PORT) || 3001;
logger.info({ port }, "Server starting");
logger.info("Using OpenAI Whisper API (gpt-4o-mini-transcribe model)");
logger.info("Make sure OPENAI_API_KEY environment variable is set");

serve({
  fetch: app.fetch,
  port,
});
