"""All settings in one place, read from environment variables (or a local .env file)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")

    # ImplicitAVE data, downloaded by the visual-product-tagging project
    iave_root: Path = Path.home() / "catalog-vlm-cascade" / "data" / "raw" / "implicitave"
    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"

    # Tagger: the fine-tuned Qwen3.5-4B served by vLLM (OpenAI-compatible API)
    vllm_url: str = "http://127.0.0.1:8000"
    tagger_model: str = str(
        Path.home() / "catalog-vlm-cascade" / "outputs" / "iave-qwen35-4b-merged"
    )
    model_version: str = "iave-qwen35-4b-v1"  # part of the idempotency key
    image_size: int = 512

    # Routing: auto-accept at or above this confidence (chosen on validation in the tagger
    # project for a 95% accuracy target); below it, a person reviews.
    accept_threshold: float = 0.72
    # Catalog QC: flag a stated value when the tagger picks a different one this confidently.
    # Chosen on dev for >= 90% precision (reports/qc_dev.json).
    qc_threshold: float = 0.86

    # Agent
    concurrency: int = 32
    checkpoint_db: Path = ROOT / "data" / "checkpoints.sqlite"
    review_db: Path = ROOT / "data" / "review.sqlite"

    # Connectors and notifications
    catalog_url: str = "http://127.0.0.1:8100"  # REST catalog (the mock store by default)
    slack_webhook_url: str = ""  # empty -> messages go to the local mock / log
    api_key: str = "change-me"


settings = Settings()
