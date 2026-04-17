"""Tests for AppConfig loading and defaults."""

from pathlib import Path

from f1_commentary.config import AppConfig, EvalConfig, GuardConfig, RetrievalConfig, TrainingConfig


def test_default_config_loads() -> None:
    """AppConfig should instantiate with all defaults."""
    cfg = AppConfig()
    assert cfg.project_root == Path(".")
    assert cfg.data_dir == Path("data")
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert isinstance(cfg.guard, GuardConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.evaluation, EvalConfig)


def test_retrieval_defaults() -> None:
    """RetrievalConfig should have sensible defaults."""
    cfg = RetrievalConfig()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.use_faiss is True
    assert cfg.tier2_priority_threshold == 0.7
    assert cfg.tier2_latency_budget_ms == 500


def test_guard_defaults() -> None:
    """GuardConfig should default to enabled."""
    cfg = GuardConfig()
    assert cfg.enable is True
    assert cfg.softening_threshold == 0.5


def test_eval_defaults() -> None:
    """EvalConfig should specify the BERTScore model."""
    cfg = EvalConfig()
    assert "deberta" in cfg.bert_score_model.lower()


def test_config_env_override(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Environment variables with F1_ prefix should override defaults."""
    monkeypatch.setenv("F1_DATA_DIR", "/tmp/test_data")
    cfg = AppConfig()
    assert cfg.data_dir == Path("/tmp/test_data")


def test_nested_env_override(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Nested config should be overridable via double-underscore delimiter."""
    monkeypatch.setenv("F1_RETRIEVAL__USE_FAISS", "false")
    cfg = AppConfig()
    assert cfg.retrieval.use_faiss is False


def test_training_defaults() -> None:
    """TrainingConfig should have sensible defaults."""
    cfg = TrainingConfig()
    assert cfg.base_model == "unsloth/Qwen2.5-7B-bnb-4bit"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.epochs == 3
    assert cfg.batch_size == 1
    assert cfg.learning_rate == 2e-4


def test_sample_config_fixture(sample_config: AppConfig) -> None:
    """The sample_config fixture should point at a real tmp directory."""
    assert sample_config.project_root.exists()
    assert sample_config.log_dir.exists()
