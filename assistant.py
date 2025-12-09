# assistant.py
import openai
import os
from typing import List
from datetime import datetime

SYSTEM_PROMPT = """
You are an objective, analytical AI assistant. 
- Provide clear, evidence-oriented answers.
- When user's assumptions seem unstated or possibly incorrect, politely surface them and propose alternative viewpoints.
- Avoid persuasive flattery. If asked to reinforce harmful actions, refuse and propose safer alternatives.
- If user requests memory storage, only store facts that are useful and non-sensitive; tag them with importance.
"""

CHALLENGER_INSTRUCTION = """
If the user expresses a strong belief as a fact, produce at least one well-reasoned counterargument or question that tests that belief.
Be respectful and focused on ideas; do not attack the person.
"""

class Assistant:
    def __init__(self, openai_api_key: str, memory):
        openai.api_key = openai_api_key
        self.memory = memory

    async def generate_reply(self, user_id: str, user_text: str, session_id=None, context_memories: List[dict]=[]):
        # Build context
        mem_block = ""
        for m in context_memories:
            mem_block += f"- {m['summary'] or m['content'][:400]} (importance={m['importance']})\n"

        # Decide whether to challenge: simple heuristic
        should_challenge = self._should_challenge(user_text)

        system = SYSTEM_PROMPT
        if should_challenge:
            system += "\n" + CHALLENGER_INSTRUCTION

        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context memories:\n{mem_block}\n\nUser says: {user_text}\n\nInstructions:\n1) Answer succinctly and objectively.\n2) If there's an unstated assumption or questionable belief, label it and offer a counterpoint.\n3) At the end, propose what (if anything) should be stored to user's long-term memory as JSON: {{store: bool, content: '...', tags:[], importance:0.0}}"}
        ]

        # call OpenAI (ChatCompletion - adjust to updated SDK if needed)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace with available model in your infra
            messages=prompt,
            max_tokens=600,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]

        # Try to parse trailing JSON suggestion for memory commit
        mem_to_commit = []
        try:
            # naive: look for last JSON block
            import re, json
            match = re.search(r"(\{[\s\S]*\}\s*)$", text.strip())
            if match:
                j = json.loads(match.group(1))
                if j.get("store"):
                    mem_to_commit.append({"content": j.get("content"), "meta": {"tags": j.get("tags",[]), "importance": j.get("importance", 0.5)}})
                # remove json from visible reply
                text = re.sub(r"(\{[\s\S]*\}\s*)$", "", text).strip()
        except Exception:
            # ignore parsing errors
            pass

        return text, mem_to_commit

    def _should_challenge(self, user_text: str) -> bool:
        # heuristic: if user_text contains strong absolute words or "always/never/I know for sure"
        lows = ["always", "never", "definitely", "for sure", "i know", "no one", "everyone"]
        t = user_text.lower()
        score = sum(1 for w in lows if w in t)
        return score >= 1
