"""
Small VLM & LLM Integration Module for Construction Safety AI.
Autonomous Edge Inference powered by AMD Lemonade SDK.

Supports local OpenAI-compatible endpoint at http://localhost:13305/v1
with automatic model discovery, streaming token delivery, and cloud benchmark comparison.
"""

import time
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional, Generator

from config import (
    LEMONADE_BASE_URL,
    LEMONADE_FALLBACK_URL,
    LEMONADE_API_KEY,
    VLM_MODELS,
    DEFAULT_VLM,
)


def check_lemonade_health(base_url: str = LEMONADE_BASE_URL) -> Dict[str, Any]:
    """
    Check if the AMD Lemonade SDK server is active and reachable.
    
    Returns:
        Dict with 'is_healthy', 'url_used', 'models', and 'error_message'
    """
    candidate_urls = [base_url, LEMONADE_FALLBACK_URL]
    
    for url in candidate_urls:
        try:
            # Check models endpoint
            models_endpoint = f"{url.rstrip('/')}/models"
            resp = requests.get(models_endpoint, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                model_list = [m.get("id") for m in data.get("data", [])] if "data" in data else []
                return {
                    "is_healthy": True,
                    "url_used": url,
                    "models": model_list,
                    "error_message": None,
                }
        except Exception:
            continue
            
    return {
        "is_healthy": False,
        "url_used": base_url,
        "models": [],
        "error_message": f"Could not connect to Lemonade SDK at {base_url}. Ensure 'lemonade serve --port 13305' is running.",
    }


class VLMInterface:
    """
    Interface for Local Edge VLM / LLM inference via AMD Lemonade SDK,
    with optional Azure OpenAI cloud benchmark support.
    """

    def __init__(self, model_key: str = DEFAULT_VLM):
        """
        Initialize the VLM interface.
        
        Args:
            model_key: Key from VLM_MODELS config dict
        """
        if model_key not in VLM_MODELS:
            # Check if it's a dynamic lemonade model name
            self.model_key = model_key
            self.model_config = {
                "model_id": model_key,
                "label": f"Lemonade Local ({model_key})",
                "backend": "lemonade",
                "description": "Locally hosted model via AMD Lemonade SDK.",
            }
        else:
            self.model_key = model_key
            self.model_config = VLM_MODELS[model_key]

        self.backend = self.model_config.get("backend", "lemonade")
        self.model_id = self.model_config.get("model_id", "auto")
        
        self._client = None
        self._device = "AMD Ryzen™ AI / Local Edge" if self.backend == "lemonade" else "api (cloud)"
        self._is_loaded = False
        self._resolved_model_id = self.model_id

    def _load_model(self):
        """
        Lazy-load the appropriate OpenAI-compatible client.
        """
        if self._is_loaded and self._client is not None:
            return

        if self.backend == "lemonade":
            try:
                from openai import OpenAI
                
                # Check health and resolve base url
                health = check_lemonade_health(LEMONADE_BASE_URL)
                base_url = health["url_used"] if health["is_healthy"] else LEMONADE_BASE_URL
                
                self._client = OpenAI(
                    base_url=base_url,
                    api_key=LEMONADE_API_KEY,
                )
                
                # If model_id is 'auto', pick the first loaded model or default to qwen
                if self.model_id == "auto":
                    if health["is_healthy"] and health["models"]:
                        self._resolved_model_id = health["models"][0]
                    else:
                        self._resolved_model_id = "qwen2.5-vl-7b-instruct"
                else:
                    self._resolved_model_id = self.model_id
                    
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to local AMD Lemonade SDK at {LEMONADE_BASE_URL}.\n"
                    f"Error: {str(e)}\n"
                    f"Quick Fix: Start Lemonade with 'lemonade serve --port 13305' in your terminal."
                )
        else:
            # Azure OpenAI fallback (for cloud latency benchmarking)
            try:
                from openai import AzureOpenAI
                from dotenv import load_dotenv
                import os
                
                load_dotenv()
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
                self._resolved_model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                
                if not api_key or not endpoint:
                    raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT missing in .env")
                
                self._client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                )
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Azure OpenAI client: {str(e)}")

    def infer(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on prompt with optional image using the local Lemonade SDK or cloud baseline.
        
        Returns:
            Dictionary with:
            - "raw_output": Full generated text
            - "inference_time_ms": Latency in milliseconds
            - "model_key": Model identifier
            - "model_label": Human-readable label
            - "device": Compute backend/device
            - "backend": 'lemonade' or 'azure'
        """
        self._load_model()
        
        start_time = time.time()
        raw_output = self._generate(image, prompt, max_new_tokens, temperature, system_prompt)
        inference_time_ms = (time.time() - start_time) * 1000.0
        
        return {
            "raw_output": raw_output,
            "inference_time_ms": round(inference_time_ms, 1),
            "model_key": self.model_key,
            "model_label": self.model_config.get("label", self.model_key),
            "resolved_model_id": self._resolved_model_id,
            "device": self._device,
            "backend": self.backend,
        }

    def infer_stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens in real-time from the Lemonade SDK server.
        """
        self._load_model()
        
        response_stream = self._client.chat.completions.create(
            model=self._resolved_model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )
        
        for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content

    def _generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Execute chat completion request with vision/text content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        user_content = []
        if image is not None:
            # Convert PIL image to base64 data URL
            buffered = BytesIO()
            # Convert RGBA to RGB if needed
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_base64}"
            
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self._client.chat.completions.create(
                model=self._resolved_model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # If vision payload fails on text-only local models, fallback to text-only prompt
            if image is not None and "image" in str(e).lower():
                text_messages = []
                if system_prompt:
                    text_messages.append({"role": "system", "content": system_prompt})
                text_messages.append({"role": "user", "content": prompt})
                response = self._client.chat.completions.create(
                    model=self._resolved_model_id,
                    messages=text_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            raise

    def is_loaded(self) -> bool:
        """Check if client is ready."""
        return self._is_loaded

    def get_device(self) -> str:
        """Get the compute device descriptor."""
        return self._device

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "resolved_model_id": self._resolved_model_id,
            "label": self.model_config.get("label", self.model_key),
            "backend": self.backend,
            "parameters": self.model_config.get("parameters", "Unknown"),
            "paper_f1": self.model_config.get("paper_f1", "N/A"),
            "paper_bertscore": self.model_config.get("paper_bertscore", "N/A"),
            "description": self.model_config.get("description", ""),
            "device": self._device,
            "is_loaded": self._is_loaded,
        }

    def unload(self):
        """Reset client."""
        if hasattr(self, "_client"):
            del self._client
            self._client = None
        self._is_loaded = False
