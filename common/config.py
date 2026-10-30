
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str
    groq_base_url: str

    model_dir: str
    audio_dir: str

    sample_rate: int
    channels: int
    device: str
    compute_type: str
    duration: int

    speech_prob_threshold: float
    block_size: int
    silence_timeout: int
    pre_roll_chunks: int

    dictate_idle_timeout_ms: int

    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from: str

    admin_api_key: str

    @property
    def model_path(self) -> str:
        return str(BASE_DIR / self.model_dir)

    @property
    def audio_path(self) -> Path:
        return BASE_DIR / self.audio_dir

    
    model_config = SettingsConfigDict(
        env_file = ( 
            BASE_DIR / ".env",
            BASE_DIR / ".env.secret"
        )
    )

settings = Settings()

#this can be imported as from app.config.settings.py import settings
