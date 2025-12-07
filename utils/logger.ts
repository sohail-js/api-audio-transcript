import pino from "pino";
import { mkdirSync } from "fs";
import { join } from "path";
import { existsSync } from "fs";

// Ensure logs directory exists
const logsDir = join(process.cwd(), "logs");
if (!existsSync(logsDir)) {
  mkdirSync(logsDir, { recursive: true });
}

// Get current date for daily log rotation
const getLogFileName = (type: "app" | "error" = "app"): string => {
  const date = new Date().toISOString().split("T")[0]; // YYYY-MM-DD
  return join(logsDir, `${type}-${date}.log`);
};

// Create file destinations
const appLogFile = getLogFileName("app");
const errorLogFile = getLogFileName("error");

// Configure Pino logger
const logLevel = process.env.LOG_LEVEL || "info";
const isDevelopment = process.env.NODE_ENV !== "production";

// Create base logger configuration
const loggerConfig: pino.LoggerOptions = {
  level: logLevel,
  formatters: {
    level: (label) => {
      return { level: label };
    },
  },
  timestamp: pino.stdTimeFunctions.isoTime,
};

// Create streams for multistream
const streams: pino.StreamEntry[] = [];

// File streams for all environments using pino.destination
// App log file - all levels
const appLogDestination = pino.destination({
  dest: appLogFile,
  append: true,
  sync: true, // Sync writes to ensure logs are written immediately
});
streams.push({
  level: logLevel as pino.Level,
  stream: appLogDestination,
});

// Error log file - errors only
const errorLogDestination = pino.destination({
  dest: errorLogFile,
  append: true,
  sync: true, // Sync writes for errors
});
streams.push({
  level: "error" as pino.Level,
  stream: errorLogDestination,
});

// Console stream - always add for visibility
streams.push({
  level: logLevel as pino.Level,
  stream: process.stdout,
});

// Create logger with multistream
const logger = pino(loggerConfig, pino.multistream(streams));

// Flush logs on process exit
process.on("beforeExit", () => {
  appLogDestination.flushSync();
  errorLogDestination.flushSync();
});

process.on("exit", () => {
  appLogDestination.flushSync();
  errorLogDestination.flushSync();
});

// Export logger instance
export default logger;

// Export helper function to create child loggers with context
export const createChildLogger = (context: Record<string, any>) => {
  return logger.child(context);
};
