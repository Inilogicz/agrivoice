"""
Text-to-speech with YarnGPT2 (Nigerian voices for Yoruba, Hausa, Igbo, English).

Runs next to N-ATLaS on the GPU. An answer is split into sentences, which are
generated together in one batch (one sequence each) and joined with short
pauses, so a whole answer takes about as long as its longest sentence.

Needs the YarnGPT repository (for `yarngpt.audiotokenizer` and its voice
files) and the WavTokenizer decoder checkpoint + config that YarnGPT uses.
https://github.com/saheedniyi02/yarngpt · https://huggingface.co/saheedniyi/YarnGPT2
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from typing import Any

import numpy as np
import torch

MODEL_ID = "saheedniyi/YarnGPT2"
SAMPLE_RATE = 24000

# AgriVoice language code → YarnGPT language and default voice
LANGUAGES = {"en-ng": "english", "yo": "yoruba", "ha": "hausa", "ig": "igbo"}
DEFAULT_VOICES = {
    "en-ng": "idera",
    "yo": "yoruba_female2",
    "ha": "hausa_female1",
    "ig": "igbo_female2",
}

_MAX_WORDS = 22      # longer sentences are split; very long ones degrade
_BATCH_SIZE = 8
_PAUSE_SECONDS = 0.3


def split_sentences(text: str) -> list[str]:
    """Plain-text sentences of at most _MAX_WORDS words, without Markdown."""
    text = re.sub(r"\*\*|__|`|#+\s*", "", text)               # bold, code, headings
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)  # list markers
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    sentences: list[str] = []
    for part in (p.strip() for p in parts):
        words = part.split()
        for i in range(0, len(words), _MAX_WORDS):
            chunk = " ".join(words[i : i + _MAX_WORDS])
            if re.search(r"[A-Za-zÀ-ɏḀ-ỿ]", chunk):
                sentences.append(chunk)
    return sentences


def _load_outetts_decoder_only() -> None:
    """
    YarnGPT imports outetts.wav_tokenizer (the audio decoder). Importing it
    normally runs outetts/__init__.py, which pulls in its whole TTS stack
    (llama-cpp, MeCab, Whisper, …) that YarnGPT never uses. Registering the
    package without running __init__ makes only the decoder load.
    """
    if "outetts" in sys.modules:
        return
    spec = importlib.util.find_spec("outetts")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("outetts is not installed (pip install --no-deps outetts==0.3.3)")
    package = types.ModuleType("outetts")
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules["outetts"] = package


class YarnTTS:
    def __init__(self, yarngpt_parent_dir: str, wavtokenizer_ckpt: str, wavtokenizer_config: str) -> None:
        if yarngpt_parent_dir not in sys.path:
            sys.path.insert(0, yarngpt_parent_dir)
        _load_outetts_decoder_only()
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        from yarngpt.audiotokenizer import AudioTokenizerV2  # type: ignore[import-not-found]  # noqa: PLC0415

        self.audio_tokenizer: Any = AudioTokenizerV2(MODEL_ID, wavtokenizer_ckpt, wavtokenizer_config)
        self.device = self.audio_tokenizer.device
        self.model: Any = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto").to(self.device).eval()
        tokenizer = self.audio_tokenizer.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def speak(self, text: str, language: str, voice: str | None = None) -> np.ndarray:
        """Float32 mono audio at SAMPLE_RATE for *text* in *language* (yo/ha/ig/en-ng)."""
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("Nothing to say.")
        voice = voice or DEFAULT_VOICES[language]
        pause = np.zeros(int(SAMPLE_RATE * _PAUSE_SECONDS), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for start in range(0, len(sentences), _BATCH_SIZE):
            for audio in self._batch(sentences[start : start + _BATCH_SIZE], LANGUAGES[language], voice):
                pieces += [audio, pause]
        return np.concatenate(pieces[:-1])

    def _batch(self, sentences: list[str], yarn_language: str, voice: str) -> list[np.ndarray]:
        tok = self.audio_tokenizer
        prompts = [tok.create_prompt(s, lang=yarn_language, speaker_name=voice) for s in sentences]
        enc = tok.tokenizer(
            prompts, add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=0.1,           # YarnGPT's recommended settings
                repetition_penalty=1.1,
                max_new_tokens=2500,       # ~30 s of audio per sentence
                pad_token_id=tok.tokenizer.pad_token_id,
            )
        prompt_length = enc["input_ids"].shape[1]
        audios: list[np.ndarray] = []
        for row in out:
            codes = tok.extract_integers(tok.tokenizer.decode(row[prompt_length:]))
            if not codes:
                continue
            audio = tok.get_audio(codes)
            audios.append(audio.squeeze().float().cpu().numpy().astype(np.float32))
        return audios
