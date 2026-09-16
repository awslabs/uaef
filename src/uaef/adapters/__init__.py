# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for transforming framework-specific traces to canonical format."""

from uaef.adapters.base import BaseAdapter, AdapterTransformationError
from uaef.adapters.registry import (
    get_adapter,
    get_registry,
    is_adapter_registered,
    list_adapters,
    register_adapter,
)
from uaef.adapters.langgraph import LangGraphAdapter
from uaef.adapters.langchain import LangChainAdapter
from uaef.adapters.bedrock import BedrockAgentAdapter
from uaef.adapters.generic import GenericJSONAdapter
from uaef.adapters.langfuse import LangfuseAdapter
from uaef.adapters.agentcore import AgentCoreAdapter
from uaef.adapters.strands import StrandsAdapter
from uaef.adapters.invocation import (
    invoke_http_agent,
    invoke_bedrock_agent,
    invoke_langfuse_trace,
    invoke_http_agent_for_framework,
)

__all__ = [
    "BaseAdapter",
    "AdapterTransformationError",
    "get_adapter",
    "get_registry",
    "is_adapter_registered",
    "list_adapters",
    "register_adapter",
    "LangGraphAdapter",
    "LangChainAdapter",
    "BedrockAgentAdapter",
    "GenericJSONAdapter",
    "LangfuseAdapter",
    "AgentCoreAdapter",
    "StrandsAdapter",
    "invoke_http_agent",
    "invoke_bedrock_agent",
    "invoke_langfuse_trace",
    "invoke_http_agent_for_framework",
]
