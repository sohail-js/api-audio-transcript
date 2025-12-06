# Audio Transcription API

A fast Node.js API server for audio transcription using OpenAI's Whisper API.

## Features

- **Multilingual transcription** supporting Hindi, Urdu, English, and 99+ languages
- **Automatic language detection** - no need to specify the language
- Fast audio transcription using OpenAI's Whisper API
- Simple REST API with Hono framework
- TypeScript support
- Supports all common audio formats (MP3, WAV, M4A, FLAC, OGG, etc.)
- No local model downloads or dependencies required

## Prerequisites

- Node.js 18+
- npm or yarn
- **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/api-keys)

## Installation

```bash
npm install
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

**Note:** If you're using a `.env` file, you may want to install `dotenv` package and load it in your server file.

## Running the Server

```bash
npm start
```

Or use the dev script for auto-reload:

```bash
npm run dev
```

The server will start on `http://localhost:3001`.

## API Endpoints

### POST /transcribe

Transcribes an audio file to text using OpenAI's Whisper API.

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
  "processingTimeMs": 5200,
  "requestId": "req-1234567890-abcde",
  "debugPath": "/path/to/requests/req-1234567890-abcde"
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "message": "OpenAI Whisper Transcription API",
  "status": "ok",
  "model": "whisper-1",
  "provider": "OpenAI",
  "languages": "Auto-detect (Hindi, Urdu, English, 99+ more)"
}
```

## Model Information

This API uses OpenAI's **Whisper-1** model, which provides:
- **Automatic language detection** - detects the language(s) in your audio
- **Support for 99+ languages** including Hindi, Urdu, and English
- **High accuracy** - state-of-the-art transcription quality
- **Fast processing** - typically 2-5 seconds for 30-second audio
- **No local setup** - runs entirely via API, no model downloads needed

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
- `500` - Server error (e.g., transcription failure, API key issues)

Common errors:
- **Missing API Key**: Set `OPENAI_API_KEY` environment variable
- **Invalid API Key**: Check your API key is correct and active
- **Rate Limit**: OpenAI API has rate limits based on your plan
- **File Format**: Ensure the audio file is in a supported format

## System Requirements

- Node.js 18 or higher
- Internet connection (for API calls)
- OpenAI API account with credits

## Performance

Typical processing times with OpenAI Whisper API:

| Audio Duration | Processing Time |
|----------------|-----------------|
| 10 seconds | ~1-2 seconds |
| 30 seconds | ~2-5 seconds |
| 1 minute | ~3-8 seconds |
| 5 minutes | ~10-20 seconds |

**Note:** Processing time includes network latency and OpenAI API processing time.

## Cost Considerations

OpenAI Whisper API pricing (as of 2024):
- $0.006 per minute of audio
- Very affordable for most use cases
- Check [OpenAI Pricing](https://openai.com/pricing) for current rates

## License

MIT
