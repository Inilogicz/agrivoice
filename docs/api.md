# AgriVoice API: frontend guide

AgriVoice answers farmers' questions by voice or text in **Yoruba, Hausa, Igbo and Nigerian English**.

**Base URL:** `https://132-145-21-82.sslip.io/api/v1`
**Interactive docs (try every endpoint in the browser):** `https://132-145-21-82.sslip.io/docs`

No authentication is needed. All responses are JSON. Timestamps are UTC (ISO 8601).

---

## The main flow: asking by voice

1. Record the farmer (browser `MediaRecorder` output is fine; see [Recording](#recording)).
2. `POST /voice/ask` with the audio. **Don't send `language`** the first time: the server detects it.
3. Then one of two things happens:
   - **200:** show the answer. Show `"Detected language: {language_name}"` when `language_source` is `"detected"`.
   - **422 with `error: "LANGUAGE_NOT_DETECTED"`:** the server couldn't tell the language. Show a picker built from `detail.available_languages`, then **send the same audio again** with `language` set to the chosen `code`.
4. To continue the same conversation, send the `conversation_id` you got back with every later request.

```
record ─► POST /voice/ask (no language)
             ├─ 200 ───────────────────────────────► show transcript + answer
             └─ 422 LANGUAGE_NOT_DETECTED ─► picker ─► POST /voice/ask (language=yo|ha|ig|en-ng) ─► 200
```

### Expect slow responses

A voice request takes **roughly 15–60 seconds**: language detection and transcription (~10–30 s on the server's CPU), then N-ATLaS writing the answer (~20–30 s). Set your request timeout to **at least 120 seconds**, and show a progress state ("Listening… / Thinking…") rather than a plain spinner.

### When N-ATLaS is offline

The answer model sometimes runs on a temporary GPU. When it's offline, `/voice/ask` still returns **200** with the transcript, but with `llm_available: false` and `response` set to a short "temporarily unavailable" notice. Show the transcript, show the notice as a system message (not as the assistant's answer), and offer a retry. Text chat returns **503 `LLM_UNAVAILABLE`** in the same situation.

---

## Endpoints

### `GET /languages`

The languages farmers can choose from. Use it for the language picker.

```json
{
  "languages": [
    { "code": "yo",    "name": "Yoruba",           "native_name": "Yorùbá" },
    { "code": "ha",    "name": "Hausa",            "native_name": "Hausa" },
    { "code": "ig",    "name": "Igbo",             "native_name": "Asụsụ Igbo" },
    { "code": "en-ng", "name": "Nigerian English", "native_name": "English" }
  ]
}
```

### `POST /voice/ask`

`multipart/form-data`:

| Field | Required | Description |
|---|---|---|
| `audio` | yes | The recording. WAV, WebM, MP3, M4A/MP4 or OGG, max 25 MB |
| `language` | no | `yo`, `ha`, `ig` or `en-ng`. Omit to auto-detect; set it after the user picks a language |
| `conversation_id` | no | Continue an existing conversation |

**200:**

```json
{
  "conversation_id": "a2b78e63-b2b6-4992-8730-9ab8cff5b91b",
  "message_id": "52bb02db-9205-494f-9547-be1b1daaa596",
  "language": "en-ng",
  "language_name": "Nigerian English",
  "language_source": "detected",
  "language_confidence": 0.99,
  "transcript": "please what fertiliser should i use for my cassava this rainy season?",
  "response": "For your cassava farm during the rainy season, …",
  "asr_model": "NCAIR1/NigerianAccentedEnglish",
  "llm_model": "NCAIR1/N-ATLaS",
  "processing_time_ms": 24180,
  "llm_available": true
}
```

| Field | Meaning |
|---|---|
| `conversation_id` | Send it with the next request to keep context |
| `message_id` | The assistant's answer; use it for `/feedback`. `null` when `llm_available` is `false` |
| `language` / `language_name` | Language used for transcription and the answer |
| `language_source` | `"detected"` (auto) or `"user_selected"` (you sent `language`) |
| `language_confidence` | 0–1. Always `1.0` when user-selected |
| `transcript` | What the farmer said, as text |
| `response` | The answer, in the same language, may contain Markdown (`**bold**`, numbered lists) |
| `llm_available` | `false` → `response` is an "unavailable" notice, not an answer |

**422, language not detected:**

```json
{
  "detail": {
    "error": "LANGUAGE_NOT_DETECTED",
    "message": "We couldn't detect the language of this recording. Please choose your language and send it again.",
    "available_languages": [
      { "code": "yo", "name": "Yoruba", "native_name": "Yorùbá" },
      { "code": "ha", "name": "Hausa", "native_name": "Hausa" },
      { "code": "ig", "name": "Igbo", "native_name": "Asụsụ Igbo" },
      { "code": "en-ng", "name": "Nigerian English", "native_name": "English" }
    ]
  }
}
```

### `POST /chat`

Text questions. JSON body:

```json
{ "message": "When should I plant maize?", "language": "en-ng", "conversation_id": null }
```

`language` is required in practice (defaults to `en-ng`); there is no auto-detection for text. `message` is 1–4096 characters.

**200:**

```json
{
  "conversation_id": "a2b78e63-b2b6-4992-8730-9ab8cff5b91b",
  "message_id": "93eedacc-4b5c-4f80-aaf8-d885614ae913",
  "language": "en-ng",
  "response": "…",
  "llm_model": "NCAIR1/N-ATLaS",
  "processing_time_ms": 21040
}
```

Voice and text can share one conversation: pass the same `conversation_id` to either.

### `POST /translate`

Translate an answer (or any saved message) into one of the other supported languages, e.g. so a farmer's child or an extension officer can read a Hausa answer in English. Show a "Translate" option under each answer with the **other three** languages.

```json
{ "message_id": "52bb02db-9205-494f-9547-be1b1daaa596", "target_language": "en-ng" }
```

**200:**

```json
{
  "message_id": "52bb02db-9205-494f-9547-be1b1daaa596",
  "source_language": "ha",
  "target_language": "en-ng",
  "target_language_name": "Nigerian English",
  "text": "To protect your maize farm from armyworm, follow these steps: …",
  "llm_model": "NCAIR1/N-ATLaS",
  "cached": false,
  "processing_time_ms": 24310
}
```

- Takes about as long as an answer (**~15–45 s**) the first time. Translations are saved: asking again for the same message and language returns instantly with `cached: true`.
- Only `yo`, `ha`, `ig` and `en-ng` are accepted. Translating into the message's own language returns `ALREADY_IN_LANGUAGE`.
- When N-ATLaS is offline it returns **503 `LLM_UNAVAILABLE`**; nothing is saved, so the user can retry later.
- Saved translations come back in `GET /conversations/{id}` under each message's `translations`.

### `GET /conversations/{conversation_id}`

The full history, oldest first. Use it to restore a chat screen.

```json
{
  "id": "a2b78e63-b2b6-4992-8730-9ab8cff5b91b",
  "language": "en-ng",
  "domain": "agriculture",
  "created_at": "2026-10-01T09:50:25",
  "updated_at": "2026-10-01T09:50:25",
  "messages": [
    {
      "id": "914c40e4-f741-4082-b909-d815143857a4",
      "role": "user",
      "content": "please what fertiliser should i use for my cassava this rainy season?",
      "language": "en-ng",
      "input_type": "voice",
      "model_used": "NCAIR1/NigerianAccentedEnglish",
      "created_at": "2026-10-01T09:50:25"
    },
    {
      "id": "52bb02db-9205-494f-9547-be1b1daaa596",
      "role": "assistant",
      "content": "For your cassava farm during the rainy season, …",
      "language": "en-ng",
      "input_type": "voice",
      "model_used": "NCAIR1/N-ATLaS",
      "created_at": "2026-10-01T09:50:25",
      "translations": [
        { "language": "yo", "content": "Fún oko ẹ̀gẹ́ rẹ ní àsìkò òjò, …", "created_at": "2026-10-01T09:52:10" }
      ]
    }
  ]
}
```

`role` is `"user"` or `"assistant"`; `input_type` is `"voice"` or `"text"`. `translations` lists any saved translations of that message (empty if none).

### `POST /feedback`

Thumbs up/down on an answer. JSON body:

```json
{ "message_id": "52bb02db-9205-494f-9547-be1b1daaa596", "rating": "helpful", "reason": null }
```

`rating` is `"helpful"`, `"not_helpful"` or `"incorrect"`. `reason` is optional free text. Returns **201** with the saved feedback.

### `GET /health`

`{"status": "ok"}` when the server is up. `GET /health/ready` lists each model's status. Useful for an "AgriVoice is starting up" screen after a server restart.

---

## Errors

Every error has the same shape:

```json
{ "detail": { "error": "ERROR_CODE", "message": "Human-readable explanation" } }
```

Show `message` to the user, and branch on `error`:

| Status | `error` | When | What to do |
|---|---|---|---|
| 422 | `LANGUAGE_NOT_DETECTED` | Auto-detect wasn't confident | Show the language picker, resend with `language` |
| 422 | `UNSUPPORTED_LANGUAGE` | `language` isn't one of the four codes | Fix the request |
| 422 | `AUDIO_VALIDATION_ERROR` | Too large, wrong type, or unreadable audio | Ask the user to record again |
| 404 | `CONVERSATION_NOT_FOUND` | Unknown `conversation_id` | Start a new conversation (drop the id) |
| 404 | `MESSAGE_NOT_FOUND` | Unknown `message_id` (translate) | Refresh the conversation |
| 422 | `ALREADY_IN_LANGUAGE` | Translating into the message's own language | Hide that language in the menu |
| 503 | `LLM_UNAVAILABLE` | N-ATLaS offline (`/chat`, `/translate`) | Show the message, offer retry |
| 503 | `MODEL_NOT_LOADED` | Server still starting up | Retry in ~30 s |
| 500 | `PIPELINE_ERROR` / `GENERATION_ERROR` / `TRANSLATION_ERROR` | Unexpected server error | Show a generic error, offer retry |

One exception: if a request is missing required fields (e.g. no `audio`), FastAPI returns 422 with `detail` as a **list** of field errors instead of an object. That's a frontend bug to fix rather than a state to show users.

---

## Recording

- **Formats:** WAV, WebM, MP3, M4A/MP4, OGG. Chrome/Firefox `MediaRecorder` (`audio/webm;codecs=opus`) and Safari (`audio/mp4`) both work as-is.
- **Length:** keep questions **under 30 seconds**. The speech models transcribe one 30-second window, so anything after that is currently ignored.
- **Size:** max 25 MB.
- Quiet surroundings help a lot; transcription accuracy drops with background noise.

---

## Example (TypeScript)

```ts
const API = "https://132-145-21-82.sslip.io/api/v1";

type Language = { code: string; name: string; native_name: string };

async function askByVoice(
  audio: Blob,
  opts: { language?: string; conversationId?: string } = {},
) {
  const form = new FormData();
  const ext = audio.type.includes("mp4") ? "m4a" : audio.type.includes("ogg") ? "ogg" : "webm";
  form.append("audio", audio, `question.${ext}`);
  if (opts.language) form.append("language", opts.language);
  if (opts.conversationId) form.append("conversation_id", opts.conversationId);

  const res = await fetch(`${API}/voice/ask`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  const data = await res.json();

  if (res.ok) return { kind: "answer" as const, data };
  if (data.detail?.error === "LANGUAGE_NOT_DETECTED") {
    return { kind: "pickLanguage" as const, languages: data.detail.available_languages as Language[] };
  }
  throw new Error(data.detail?.message ?? "Something went wrong");
}

// Usage
const first = await askByVoice(recording);
if (first.kind === "pickLanguage") {
  const code = await showLanguagePicker(first.languages); // your UI
  const second = await askByVoice(recording, { language: code });
  // render second.data
}
```

---

## Notes for development

- **Restarts:** after a server restart the models take ~1–4 minutes to load. `GET /health` responds once it's up; `GET /health/ready` shows when every model is loaded.
- **Mock responses:** text starting with `[MOCK]` comes from placeholder models used in local development, never from the deployed server.
- **CORS:** the server allows `http://localhost:3000` and `http://localhost:5173`. Tell us your dev or production origin if it's different.
