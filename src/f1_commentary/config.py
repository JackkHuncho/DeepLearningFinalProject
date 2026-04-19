"""Application configuration using Pydantic BaseSettings.

Supports environment variable overrides with the F1_ prefix
and double-underscore nesting (e.g. F1_RETRIEVAL__EMBEDDING_MODEL).
"""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval / memory subsystem."""

    embedding_model: str = "all-MiniLM-L6-v2"
    use_faiss: bool = True
    tier2_priority_threshold: float = 0.7
    tier2_latency_budget_ms: int = 500


class GuardConfig(BaseModel):
    """Configuration for the grounding guard."""

    enable: bool = True
    softening_threshold: float = 0.5


class TrainingConfig(BaseModel):
    """Configuration for SFT fine-tuning (Phase 8)."""

    base_model: str = "unsloth/Qwen2.5-7B-bnb-4bit"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    seed: int = 42
    output_dir: str = "data/artifacts/qwen25-7b-f1-lora"


class EvalConfig(BaseModel):
    """Configuration for evaluation metrics."""

    bert_score_model: str = "microsoft/deberta-xlarge-mnli"


class AppConfig(BaseSettings):
    """Top-level application configuration.

    All fields can be overridden via environment variables prefixed with ``F1_``.
    Nested models use ``__`` as a delimiter, e.g. ``F1_RETRIEVAL__USE_FAISS=false``.
    """

    model_config = {"env_prefix": "F1_", "env_nested_delimiter": "__"}

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    log_dir: Path = Path("data/logs")
    trace_dir: Path = Path("data/logs/traces")

    retrieval: RetrievalConfig = RetrievalConfig()
    guard: GuardConfig = GuardConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvalConfig = EvalConfig()
