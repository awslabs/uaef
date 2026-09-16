# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for UAEF configuration management."""

import os
import pytest
from unittest.mock import patch

from uaef.config import (
    AWSConfig,
    LLMJudgeConfig,
    StorageConfig,
    UAEFConfig,
    get_config,
    reset_config,
    set_config,
)


class TestAWSConfig:
    def test_defaults(self):
        cfg = AWSConfig()
        assert cfg.region == "us-east-1"
        assert cfg.access_key_id is None

    def test_from_env(self):
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            cfg = AWSConfig.from_env()
            assert cfg.region == "eu-west-1"


class TestLLMJudgeConfig:
    def test_defaults(self):
        cfg = LLMJudgeConfig()
        assert cfg.model_id == "us.anthropic.claude-sonnet-4-6"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2048
        assert cfg.retry_attempts == 3

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            LLMJudgeConfig(temperature=1.5)


class TestUAEFConfig:
    def test_defaults(self):
        cfg = UAEFConfig()
        assert cfg.log_level == "INFO"
        assert cfg.enable_caching is True
        assert cfg.llm_judge.model_id == "us.anthropic.claude-sonnet-4-6"

    def test_from_env(self):
        # Security review L-03: LLMJudgeConfig.model_id now validates against
        # the vetted Anthropic Claude family (the only family UAEF's judge
        # metrics actually build a correct request body for), so the env var
        # plumbing this test exercises needs a validly-shaped placeholder
        # instead of an arbitrary string.
        with patch.dict(os.environ, {
            "UAEF_LLM_MODEL_ID": "anthropic.claude-test-model",
            "UAEF_LOG_LEVEL": "DEBUG",
        }):
            cfg = UAEFConfig.from_env()
            assert cfg.llm_judge.model_id == "anthropic.claude-test-model"
            assert cfg.log_level == "DEBUG"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError):
            UAEFConfig(log_level="INVALID")


class TestGlobalConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, UAEFConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_overrides(self):
        custom = UAEFConfig(log_level="DEBUG")
        set_config(custom)
        assert get_config().log_level == "DEBUG"

    def test_reset_config_clears(self):
        get_config()  # Initialize
        reset_config()
        # After reset, next call creates fresh instance
        cfg = get_config()
        assert cfg.log_level == "INFO"
