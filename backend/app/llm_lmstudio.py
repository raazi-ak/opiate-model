import os
import requests
from typing import Optional


class LMStudioClient:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or os.getenv("LMSTUDIO_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
        self.host = host or os.getenv("LMSTUDIO_HOST", "http://127.0.0.1:1234")

    def build_prompt(self, user_message: str, objective: Optional[str], retrieved_context: str, stress_level: Optional[str] = None) -> str:
        parts = []
        # System-style guardrails at top for better grounding
        guardrails = [
            "You are a study assistant. You must answer STRICTLY using the CONTEXT below.",
            "If the CONTEXT does not contain the answer, reply: 'I don't know based on the provided materials.'",
            "Do NOT invent facts or draw from outside knowledge unless it's commonly known fact or if explicitly asked to do so.",
        ]
        parts.append("SYSTEM RULES:\n- " + "\n- ".join(guardrails))

        if objective:
            parts.append(f"LEARNING OBJECTIVE: {objective}")

        if retrieved_context:
            parts.append("BEGIN CONTEXT\n" + retrieved_context + "\nEND CONTEXT")
        else:
            parts.append("BEGIN CONTEXT\n\nEND CONTEXT")
        
        # Add stress-aware instructions
        if stress_level == "high":
            parts.append("STRESS LEVEL: HIGH — Be very concise and practical. Focus on essentials. Suggest a 5-minute break if appropriate. Simple language.")
        elif stress_level == "low":
            parts.append("STRESS LEVEL: LOW — Provide detailed explanations with brief examples as needed.")
        else:  # medium
            parts.append("STRESS LEVEL: MEDIUM — Provide clear, structured explanations. Focus on key concepts.")
        
        parts.append("USER QUESTION:\n" + user_message)
        parts.append(
            "ANSWER REQUIREMENTS: 1) Use only CONTEXT. 2) If insufficient, say you don't know based on the materials or use commonly known facts. 3) Keep under 300 words. 4) Keep <think> under 60 tokens, then produce the final answer."
        )
        return "\n\n".join(parts)

    def generate_text(self, prompt: str) -> str:
        try:
            url = f"{self.host}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 700,
                "stream": False
            }
            
            resp = requests.post(url, json=data, headers=headers, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            
            # Extract the response content
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "[LLM error] No response generated"
                
        except requests.exceptions.ConnectionError:
            return "[LLM error] Could not connect to LM Studio. Make sure LM Studio is running and the model is loaded."
        except requests.exceptions.Timeout:
            return "[LLM error] Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"[LLM error] {e}"

    def stream_text(self, prompt: str):
        """Yield chunks of text using LM Studio streaming API (Server-Sent Events style)."""
        url = f"{self.host}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": True,
        }
        try:
            with requests.post(url, json=data, headers=headers, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[len("data: "):]
                        if payload == "[DONE]":
                            break
                        # Each payload is a JSON with delta tokens
                        try:
                            import json as _json
                            obj = _json.loads(payload)
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or choices[0].get("message", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                        except Exception:
                            continue
        except Exception as e:
            yield f"[LLM error] {e}"
