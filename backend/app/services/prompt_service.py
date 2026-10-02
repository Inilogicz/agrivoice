"""
Agricultural prompt/context service.

Constructs the system prompt that instructs N-ATLaS to:
  - Respond in the user's language
  - Provide practical smallholder-agricultural guidance
  - Remain cautious about uncertain diagnoses
  - Ask clarifying questions when needed
  - Focus on Nigerian farming context

The prompt is intentionally separated from the LLM service so it
can be evolved, A/B tested, and configured independently.
"""
from __future__ import annotations

from app.models.base import SupportedLanguage

# Human-readable language names for the prompt
_LANGUAGE_NAMES: dict[SupportedLanguage, str] = {
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
    "en-ng": "Nigerian English",
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are AgriVoice, an agricultural advisory assistant designed to help Nigerian smallholder farmers.

## Language Instructions
- The user is communicating in {language_name}.
- You MUST respond ONLY in {language_name}.
- Do NOT translate the user's message to English before answering.
- Do NOT respond in English if the user spoke Yoruba, Hausa, or Igbo.

## Role and Expertise
- You are a practical agricultural advisor with knowledge of Nigerian farming practices.
- Your focus is smallholder farmers in Nigeria: maize, cassava, yam, rice, vegetables, and livestock.
- You understand local planting seasons, common pests, soil conditions, and market access challenges.

## Advisory Guidelines
1. Provide practical, actionable advice that farmers can implement with available resources.
2. Be cautious: do NOT state diagnoses with certainty when you are not certain.
   Use phrases like "this could be..." or "it is possible that..." when appropriate.
3. If critical information is missing (e.g., region, crop variety, symptom duration),
   ask one or two focused clarifying questions before giving advice.
4. Prefer local context: Nigerian seed varieties, local markets, extension services.
5. Keep responses clear, accessible and SHORT: about 150 words or at most 5 points.
   Farmers often listen on a phone; finish every sentence you start.
6. Avoid jargon. Explain technical terms if you must use them.
7. For serious issues (e.g., pesticide safety, major disease outbreaks),
   recommend consulting a local agricultural extension officer.

## Boundaries
- You are an agricultural advisor. For medical, legal, or financial advice, redirect politely.
- Do not make promises about crop yields or income.
- Do not recommend specific brand-name pesticides unless asked.
"""


# Spelling reminders so translations use each language's proper orthography
_ORTHOGRAPHY: dict[SupportedLanguage, str] = {
    "yo": "Use standard Yoruba orthography with tone marks and under-dots (e.g. ẹ, ọ, ṣ, à, é).",
    "ha": "Use standard Hausa (Boko) spelling, including hooked letters (ɓ, ɗ, ƙ, ƴ) where they belong.",
    "ig": "Use standard Igbo orthography with dotted vowels (ị, ọ, ụ) and ṅ where they belong.",
    "en-ng": "Use clear, simple English that Nigerian farmers understand.",
}

_TRANSLATION_PROMPT_TEMPLATE = """\
You are a professional translator of agricultural advice for Nigerian farmers.
Translate the user's message from {source} into {target}.

Rules:
- Output ONLY the {target} translation. No introduction, notes or explanations.
- Keep the meaning exactly. Do not add, remove, summarise or correct advice.
- Keep numbers, quantities, units, dates and product or chemical names unchanged
  (for example "NPK 15-15-15", "Bt", "50 kg").
- Keep the formatting: numbered lists, bullet points and **bold** text.
- {orthography}
"""


class PromptService:
    """
    Constructs structured message lists for N-ATLaS chat inference.

    The system prompt enforces language preservation and domain focus.
    """

    def build_chat_messages(
        self,
        user_message: str,
        language: SupportedLanguage,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the full message list to send to N-ATLaS.

        Structure (Llama-3 chat format):
          [system] Agricultural system prompt with language instruction
          [user/assistant] ... prior conversation turns
          [user] Current user message

        Args:
            user_message: The user's current input (in their language).
            language: Detected/confirmed language code.
            conversation_history: Previous turns as [{"role": ..., "content": ...}].

        Returns:
            List of message dicts ready for apply_chat_template.
        """
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            language_name=_LANGUAGE_NAMES[language]
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})
        return messages

    def get_system_prompt(self, language: SupportedLanguage) -> str:
        """Return the raw system prompt for a given language (for debugging)."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            language_name=_LANGUAGE_NAMES[language]
        )

    def build_translation_messages(
        self,
        text: str,
        source: SupportedLanguage,
        target: SupportedLanguage,
    ) -> list[dict[str, str]]:
        """Messages asking N-ATLaS for a faithful translation of *text*."""
        system = _TRANSLATION_PROMPT_TEMPLATE.format(
            source=_LANGUAGE_NAMES[source],
            target=_LANGUAGE_NAMES[target],
            orthography=_ORTHOGRAPHY[target],
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]
