"""
N-ATLaS on Hugging Face ZeroGPU.

Follows the official usage on https://huggingface.co/NCAIR1/N-ATLaS:
  - AutoTokenizer + AutoModelForCausalLM, float16
  - apply_chat_template(..., tokenize=False, date_string=<today>)
  - tokenizer(text, add_special_tokens=False)
  - generate(max_new_tokens, use_cache=True, repetition_penalty=1.12, greedy)

Two differences from the model card's snippet, both needed to run it:
  - `from datetime import datetime` (missing in the card)
  - decode only the newly generated tokens, so the prompt isn't echoed back

The AgriVoice backend calls the `/generate` endpoint with a list of chat
messages. The "Chat" tab is for trying the model by hand.

With TTS_ENABLED=1, `/speak` also reads text aloud with YarnGPT2 (tts.py),
on the same GPU. It needs YARNGPT_DIR, WAVTOKENIZER_CKPT, WAVTOKENIZER_CONFIG.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "NCAIR1/N-ATLaS")
# "cuda": ZeroGPU (model placed on CUDA at import time, as ZeroGPU requires)
# "auto": spread across several GPUs, e.g. Kaggle's 2 × T4 (16 GB doesn't fit one)
# "cpu":  local testing with a small stand-in model only
DEVICE = os.getenv("DEVICE", "cuda")
HF_TOKEN = os.getenv("HF_TOKEN")  # Space secret; N-ATLaS is a gated model

DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_LIMIT = 1000  # The model card's value
REPETITION_PENALTY = 1.12

DEFAULT_SYSTEM_PROMPT = (
    "you are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if _tokenizer is None:
    raise RuntimeError(f"Could not load the tokenizer for {MODEL_ID}")
tokenizer = _tokenizer

if DEVICE == "auto":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, token=HF_TOKEN, device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
        token=HF_TOKEN,
    ).to(DEVICE)
model.eval()

# Optional text-to-speech, loaded once like the model (off unless TTS_ENABLED=1)
tts_engine = None
if os.getenv("TTS_ENABLED") == "1":
    from tts import YarnTTS  # noqa: PLC0415

    tts_engine = YarnTTS(
        os.environ["YARNGPT_DIR"], os.environ["WAVTOKENIZER_CKPT"], os.environ["WAVTOKENIZER_CONFIG"]
    )


def format_text_for_inference(messages: list[dict[str, str]]) -> str:
    current_date = datetime.now().strftime("%d %b %Y")
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        date_string=current_date,
    )
    if not isinstance(text, str):  # tokenize=False always returns a string
        raise TypeError(f"Unexpected chat template output: {type(text).__name__}")
    return text


@spaces.GPU(duration=90)
def generate(messages: list[dict[str, str]], max_new_tokens: float = DEFAULT_MAX_NEW_TOKENS) -> str:
    """
    Generate the assistant's reply.

    Args:
        messages: [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
        max_new_tokens: Upper bound on reply length (capped at 1000). A float because
            Gradio sliders and API clients may send e.g. 512.0.
    """
    if not messages or not all(
        isinstance(m, dict) and m.get("role") and isinstance(m.get("content"), str)
        for m in messages
    ):
        raise gr.Error("`messages` must be a list of {role, content} objects.")

    max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))
    text = format_text_for_inference(messages)
    input_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            repetition_penalty=REPETITION_PENALTY,
            # The card passes temperature=0.1 without do_sample, which decodes
            # greedily; this is the same behaviour stated explicitly.
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_tokens["input_ids"].shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if not isinstance(reply, str):  # one sequence decodes to one string
        raise TypeError(f"Unexpected decode output: {type(reply).__name__}")
    return reply.strip()


def _text(content: object) -> str:
    """Gradio 6 chat history may hold content as a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


@spaces.GPU(duration=120)
def speak(text: str, language: str, voice: str = "") -> str:
    """
    Read *text* aloud. Returns the path of a 24 kHz WAV file.

    Args:
        text: What to say (Markdown is stripped; long text is split into sentences).
        language: "yo", "ha", "ig" or "en-ng".
        voice: Optional YarnGPT voice name; empty uses the language's default.
    """
    if tts_engine is None:
        raise gr.Error("Text-to-speech is not enabled on this server.")
    import soundfile as sf  # noqa: PLC0415
    from tts import SAMPLE_RATE  # noqa: PLC0415

    try:
        audio = tts_engine.speak(text, language, voice or None)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, audio, SAMPLE_RATE)
    return path


def chat(message: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": _text(m["content"])} for m in history]
    messages.append({"role": "user", "content": message})
    return generate(messages)


with gr.Blocks(title="N-ATLaS") as demo:
    gr.Markdown("# N-ATLaS\nNCAIR's multilingual model (Yoruba, Hausa, Igbo, English).")
    with gr.Tab("Chat"):
        gr.ChatInterface(chat, api_visibility="private")
    with gr.Tab("API"):
        messages_in = gr.JSON(
            label="messages",
            value=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "menene ake nufi da gwagwarmaya"},
            ],
        )
        max_tokens_in = gr.Slider(1, MAX_NEW_TOKENS_LIMIT, value=DEFAULT_MAX_NEW_TOKENS, step=1,
                                  label="max_new_tokens")
        reply_out = gr.Textbox(label="reply")
        gr.Button("Generate").click(
            generate, inputs=[messages_in, max_tokens_in], outputs=reply_out, api_name="generate"
        )
    with gr.Tab("Speech"):
        speech_text = gr.Textbox(label="text", value="Ẹ káàbọ̀ sí AgriVoice.")
        speech_lang = gr.Dropdown(["yo", "ha", "ig", "en-ng"], value="yo", label="language")
        speech_voice = gr.Textbox(label="voice (optional)")
        speech_out = gr.Audio(label="speech", type="filepath")
        gr.Button("Speak").click(
            speak, inputs=[speech_text, speech_lang, speech_voice], outputs=speech_out, api_name="speak"
        )

if __name__ == "__main__":
    demo.launch()
