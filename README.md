# Audio Transcription API

A fast Node.js API server for audio transcription using OpenAI's Whisper model via nodejs-whisper.

## Features

- **Multilingual transcription** supporting Hindi, Urdu, English, and 99+ languages
- **Automatic language detection** - no need to specify the language
- Fast audio transcription using Whisper base model
- Simple REST API with Hono framework
- TypeScript support
- Supports all common audio formats (MP3, WAV, M4A, FLAC, OGG, etc.)

## Prerequisites

- Node.js 18+
- npm or yarn
- **CMake** (required for building Whisper C++ binaries)
  - On Mac: `brew install cmake`
  - On Ubuntu/Debian: `sudo apt-get install cmake`
  - On Windows: Download from [cmake.org](https://cmake.org/download/)

## Installation

```bash
npm install
```

## Running the Server

```bash
npm start
```

Or use the dev script for auto-reload:

```bash
npm run dev
```

The server will start on `http://localhost:3001`.

**Note:** The first run will download the Whisper base model (~150MB). Subsequent runs will be instant as the model is cached.

## API Endpoints

### POST /transcribe

Transcribes an audio file to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `audio` field containing the audio file

**Example using curl:**

```bash
curl -X POST http://localhost:3001/transcribe \
  -F "audio=@path/to/your/audio.mp3"
```

**Response:**
```json
{
  "text": "transcribed text in original language",
  "processingTimeSeconds": 5.2,
  "processingTimeMs": 5200
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "message": "Whisper Transcription API",
  "status": "ok",
  "model": "base",
  "languages": "Auto-detect (Hindi, Urdu, English, 99+ more)"
}
```

## Model Information

This API uses the Whisper `base` model via nodejs-whisper, which provides:
- **Automatic language detection** - detects the language(s) in your audio
- **Support for 99+ languages** including Hindi, Urdu, and English
- **Good balance** between speed and accuracy
- **Fast processing** - typically 5-10 seconds for 30-second audio

### Available Models

You can change the model in `services/transcription-service.ts`:

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | ~75MB | Fastest | Good | Quick tests |
| **base** | **~150MB** | **Fast** | **Very Good** | **Default** ⭐ |
| small | ~500MB | Moderate | Excellent | High accuracy |
| medium | ~1.5GB | Slow | Best | Maximum accuracy |
| large | ~3GB | Very Slow | Best | Research use |

## Supported Languages

The model automatically detects and transcribes:
- Hindi (हिन्दी)
- Urdu (اردو)
- English
- Arabic (العربية)
- Spanish (Español)
- French (Français)
- German (Deutsch)
- And 90+ more languages

## Testing

Test with the diagnostic tool:

```bash
npm run test path/to/audio-file.mp3
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (e.g., no audio file provided)
- `500` - Server error (e.g., transcription failure)

## System Requirements

- Node.js 18 or higher
- Disk space: ~200MB for model cache
- Memory: 2GB+ RAM recommended

## Performance

Typical processing times with base model:

| Audio Duration | Processing Time |
|----------------|-----------------|
| 10 seconds | ~2-4 seconds |
| 30 seconds | ~5-10 seconds |
| 1 minute | ~10-20 seconds |
| 5 minutes | ~50-100 seconds |

## License

MIT
