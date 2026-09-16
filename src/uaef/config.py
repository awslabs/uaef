# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for UAEF."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class AWSConfig(BaseModel):
    """AWS configuration for Bedrock LLM Judge."""
    
    region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock"
    )
    access_key_id: Optional[str] = Field(
        None,
        description="AWS access key ID (optional, can use IAM role)"
    )
    secret_access_key: Optional[str] = Field(
        None,
        description="AWS secret access key (optional, can use IAM role)"
    )
    session_token: Optional[str] = Field(
        None,
        description="AWS session token (optional)"
    )
    
    @classmethod
    def from_env(cls) -> "AWSConfig":
        """Load AWS configuration from environment variables."""
        return cls(
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )


#: Security review L-03: judge model IDs UAEF's own request-body construction
#: actually supports. Every real metric module (response_quality, reasoning,
#: responsible_ai, multi_agent, multi_turn) hardcodes the Anthropic Claude
#: Messages API request shape (`anthropic_version` / `messages` body) when
#: calling `bedrock_client.invoke_model` — a non-Claude model_id was already
#: silently broken (Bedrock rejects the malformed request body) before this
#: check existed. This validator makes that failure explicit and immediate
#: (at config load) instead of a confusing per-call Bedrock error, and closes
#: off deploying against an unvetted or non-Anthropic model family that
#: UAEF's prompt/response handling has never been validated against. It does
#: NOT change which models actually work today.
_VETTED_JUDGE_MODEL_PREFIXES: tuple = (
    "anthropic.claude",
    "us.anthropic.claude",
    "eu.anthropic.claude",
    "apac.anthropic.claude",
)


def _validate_judge_model_id(v: str) -> str:
    """Reject a judge model_id outside the vetted Anthropic Claude family.

    Accepts both a bare model ID (``anthropic.claude-...``) and a
    cross-region inference-profile ID (``us.anthropic.claude-...``), since
    both are valid ``modelId`` values for ``bedrock-runtime.invoke_model``.
    """
    if not any(v.startswith(prefix) for prefix in _VETTED_JUDGE_MODEL_PREFIXES):
        raise ValueError(
            f"Unvetted judge model_id: {v!r}. UAEF's LLM-judge metrics hardcode "
            "the Anthropic Claude Messages API request format, so only "
            f"Anthropic Claude model IDs are supported (one of the prefixes "
            f"{_VETTED_JUDGE_MODEL_PREFIXES}). A different model family would "
            "already fail at the Bedrock API call with a malformed-request "
            "error; this check surfaces that at config load instead."
        )
    return v


class LLMJudgeConfig(BaseModel):
    """Configuration for LLM Judge."""
    
    model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        description="Bedrock model ID for LLM judge"
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for LLM generation",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response",
        gt=0
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed API calls",
        ge=1
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds",
        gt=0.0
    )
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Temperature must be between 0 and 1, got {v}")
        return v

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Reject a judge model_id outside the vetted Anthropic Claude family."""
        return _validate_judge_model_id(v)


class InsightConfig(BaseModel):
    """Configuration for LLM-powered insight generation.
    
    Defaults match LLMJudgeConfig but with slightly higher temperature
    for more natural prose output.
    """
    
    model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        description="Bedrock model ID for insight generation"
    )
    temperature: float = Field(
        default=0.3,
        description="Temperature for insight generation (slightly creative)",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for insight response",
        gt=0
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed API calls",
        ge=1
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds",
        gt=0.0
    )


class StorageConfig(BaseModel):
    """Configuration for DynamoDB + S3 persistence."""
    
    dynamodb_table_name: str = Field(
        default="uaef-experiments",
        description="DynamoDB table name for experiment metadata"
    )
    s3_bucket: str = Field(
        default="uaef-results",
        description="S3 bucket for full evaluation results"
    )
    s3_prefix: str = Field(
        default="evaluations/",
        description="S3 key prefix for result files"
    )
    region: Optional[str] = Field(
        None,
        description="AWS region override (defaults to aws.region if not set)"
    )
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load storage configuration from environment variables."""
        return cls(
            dynamodb_table_name=os.getenv("UAEF_DYNAMODB_TABLE", "uaef-experiments"),
            s3_bucket=os.getenv("UAEF_S3_BUCKET", "uaef-results"),
            s3_prefix=os.getenv("UAEF_S3_PREFIX", "evaluations/"),
            region=os.getenv("UAEF_STORAGE_REGION"),
        )


class UAEFConfig(BaseModel):
    """Main UAEF configuration."""
    
    aws: AWSConfig = Field(
        default_factory=AWSConfig,
        description="AWS configuration"
    )
    llm_judge: LLMJudgeConfig = Field(
        default_factory=LLMJudgeConfig,
        description="LLM Judge configuration"
    )
    insight: InsightConfig = Field(
        default_factory=InsightConfig,
        description="LLM-powered insight generation configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="DynamoDB + S3 storage configuration"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching for LLM responses"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds",
        gt=0
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    @classmethod
    def from_env(cls) -> "UAEFConfig":
        """Load configuration from environment variables."""
        return cls(
            aws=AWSConfig.from_env(),
            llm_judge=LLMJudgeConfig(
                model_id=os.getenv(
                    "UAEF_LLM_MODEL_ID",
                    "us.anthropic.claude-sonnet-4-6"
                ),
                temperature=float(os.getenv("UAEF_LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("UAEF_LLM_MAX_TOKENS", "2048")),
            ),
            insight=InsightConfig(
                model_id=os.getenv(
                    "UAEF_INSIGHT_MODEL_ID",
                    os.getenv("UAEF_LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
                ),
                temperature=float(os.getenv("UAEF_INSIGHT_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("UAEF_INSIGHT_MAX_TOKENS", "4096")),
            ),
            storage=StorageConfig.from_env(),
            log_level=os.getenv("UAEF_LOG_LEVEL", "INFO"),
            enable_caching=os.getenv("UAEF_ENABLE_CACHING", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("UAEF_CACHE_TTL_SECONDS", "3600")),
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "UAEFConfig":
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            UAEFConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                import json
                config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def save_to_file(self, config_path: Path) -> None:
        """
        Save configuration to a YAML or JSON file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        with open(config_path, "w") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix == ".json":
                import json
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )


# Global configuration instance
_config: Optional[UAEFConfig] = None


def get_config() -> UAEFConfig:
    """
    Get the global UAEF configuration.
    
    Loads from environment variables on first call.
    
    Returns:
        UAEFConfig instance
    """
    global _config
    if _config is None:
        _config = UAEFConfig.from_env()
    return _config


def set_config(config: UAEFConfig) -> None:
    """
    Set the global UAEF configuration.
    
    Args:
        config: UAEFConfig instance to set as global
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _config
    _config = None
