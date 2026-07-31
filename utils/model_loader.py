import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentIntelligenceError
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        redacted_keys = {
            k: (v[:6] + "..." if isinstance(v, str) and len(v) >= 6 else "set")
            for k, v in self.api_keys.items()
        }
        log.info("API keys discovered", keys=redacted_keys)


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def list_available_embeddings(self):
        """
        Return configured embedding model options from YAML.
        """
        emb_block = self.config.get("embedding_model", {}) or {}
        options_block = emb_block.get("options", {})
        default_key = emb_block.get("default", next(iter(options_block), ""))
        options = [
            {
                "key": key,
                "provider": cfg.get("provider", key),
                "model_name": cfg.get("model_name", ""),
                "label": f"{cfg.get('provider', key)}: {cfg.get('model_name', '')}",
            }
            for key, cfg in options_block.items()
        ]
        return {"default": default_key, "embeddings": options}

    def load_embeddings(self, embedding_key: str | None = None):
        """
        Load and return the configured embedding model. Provider is driven by config.
        """
        try:
            emb_block = self.config["embedding_model"]
            options = emb_block.get("options", {})
            key = embedding_key or emb_block.get("default", next(iter(options)))
            cfg = options.get(key) or next(iter(options.values()))
            provider = cfg.get("provider", "google")
            model_name = cfg.get("model_name")
            log.info("Loading embedding model", provider=provider, model=model_name)

            if provider == "google":
                if GoogleGenerativeAIEmbeddings is None:
                    raise ImportError("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),  # type: ignore
                )
            elif provider == "openai":
                if OpenAIEmbeddings is None:
                    raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
                return OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),  # type: ignore
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentIntelligenceError("Failed to load embedding model", sys)

    def list_available_llms(self):
        """
        Return configured LLM options from YAML.
        """
        llm_block = self.config.get("llm", {}) or {}
        options = []
        for key, llm_config in llm_block.items():
            provider = llm_config.get("provider", key)
            model_name = llm_config.get("model_name", "")
            options.append(
                {
                    "key": key,
                    "provider": provider,
                    "model_name": model_name,
                    "label": f"{provider}: {model_name}",
                }
            )
        return options

    def load_llm(self, model_key: str | None = None):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = model_key or os.getenv("LLM_PROVIDER", "openai")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            if ChatGroq is None:
                raise ImportError("langchain-groq is not installed. Run: pip install langchain-groq")
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        elif provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "anthropic":
            if ChatAnthropic is None:
                raise ImportError(
                    "langchain-anthropic is not installed. Install it with 'pip install langchain-anthropic'."
                )
            return ChatAnthropic(
                model=model_name,
                anthropic_api_key=self.api_key_mgr.get("ANTHROPIC_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # List configured LLM options
    llm_options = loader.list_available_llms()
    print(f"Configured LLM options: {llm_options}")

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
