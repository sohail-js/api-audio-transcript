# Audio Transcription API

A simple Node.js API server built with Hono and transformers.js that transcribes audio files using the Whisper model.

## Features

- Fast audio transcription using Whisper-tiny model
- Simple REST API with Hono framework
- TypeScript support
- Multipart form-data file upload support

## Prerequisites

- Node.js 18+ 
- npm or yarn

## Installation

1. Install dependencies:

```bash
npm install
```

## Running the Server

Start the server:

```bash
npm start
```

Or use the dev script for auto-reload:

```bash
npm run dev
```

The server will start on `http://localhost:3000`.

**Note:** The first run will download the Whisper model (`Xenova/whisper-tiny`), which may take a few moments. Subsequent runs will be faster as the model is cached.

## API Endpoints

### POST /transcribe

Transcribes an audio file to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `audio` field containing the audio file

**Example using curl:**

```bash
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@path/to/your/audio.mp3"
```

**Response:**
```json
{
  "text": "transcribed text here"
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "message": "Whisper Transcription API",
  "status": "ok"
}
```

## Model Information

This API uses the `Xenova/whisper-tiny` model, which is the fastest Whisper model while still providing good transcription quality. The model is automatically downloaded and cached on first use.

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (e.g., no audio file provided)
- `500` - Server error (e.g., transcription failure)

## System Requirements

- Node.js 18 or higher
- Sufficient disk space for model cache (~100-200MB)
- Memory: Recommended 2GB+ RAM for smooth operation
