import os
import time
import logging
from groq import Groq, RateLimitError, APIError

class LLMManager:
    """
    Centralized Manager for LLM interactions with robust fallback strategies.
    Cycles through multiple models to handle Rate Limit (429) errors.
    """
    
    # Priority list of models to try (Groq retired the Llama/Mixtral/Gemma ids
    # this originally listed; check client.models.list() if these 404 again).
    # 1. Qwen 3.8 27B (non-reasoning: honors small max_tokens)
    # 2. GPT-OSS 20B (reasoning: needs a generous max_tokens, empty replies are skipped)
    # 3. GPT-OSS 120B (reasoning, separate quota)
    MODELS = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found. LLM features will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            print(f"🧠 LLMManager initialized with {len(self.MODELS)} fallback models.")

    def chat_completion(self, messages, max_tokens=100, temperature=0.7):
        """
        Attempts to get a chat completion, trying models in sequence if Rate Limits are hit.
        """
        if not self.client:
            return None
            
        for model in self.MODELS:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = (completion.choices[0].message.content or "").strip()
                if not content:
                    # Reasoning models can spend the whole token budget thinking
                    print(f"   ⚠️ Empty reply from {model}. Switching to backup...")
                    continue
                return content
                
            except RateLimitError as e:
                print(f"   ⚠️ Rate Limit hit for {model}. Switching to backup...")
                time.sleep(1)
                continue
            except APIError as e:
                print(f"   ⚠️ API Error with {model}: {e}")
                time.sleep(1)
                continue
                
            except Exception as e:
                print(f"   ❌ Unexpected LLM Error ({model}): {e}")
                break # Stop on non-API errors (like network/auth)
                
        print("   ❌ All LLM models exhausted or failed.")
        return None

    def tool_chat_completion(self, messages, tools=None, tool_choice="auto", temperature=0.7):
        """
        Get a chat completion with tool calling support.
        Returns the full message object (content + tool_calls).
        """
        if not self.client:
            return None
            
        for model in self.MODELS:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice
                )
                return completion.choices[0].message
                
            except RateLimitError as e:
                print(f"   ⚠️ Rate Limit hit for {model}. Switching to backup...")
                time.sleep(1)
                continue
            except APIError as e:
                print(f"   ⚠️ API Error with {model}: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"   ❌ Unexpected LLM Error ({model}): {e}")
                break
                
        print("   ❌ All LLM models exhausted or failed.")
        return None

# Singleton Pattern
_llm_manager = None

def get_llm_manager():
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
