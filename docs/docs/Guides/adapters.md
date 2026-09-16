# Adapter Integration Guide

## Overview

UAEF adapters transform framework-specific agent traces into a canonical format that can be evaluated consistently. This guide shows you how to integrate different agent frameworks with UAEF.

## Supported Frameworks

UAEF provides built-in adapters for:

- **LangGraph**: State-based agent orchestration
- **LangChain**: Chain-based agent workflows
- **Bedrock Agents**: AWS managed agents
- **Langfuse**: Observability and tracing
- **AgentCore**: Custom agent runtime
- **Generic JSON**: Any custom framework

## Architecture

```
┌─────────────────────┐
│ Framework-Specific  │
│ Trace Data          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ UAEF Adapter        │
│ (Transform)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Canonical Format    │
│ (AgentTrace)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ UAEF Evaluation     │
└─────────────────────┘
```

## LangGraph Integration

### Overview

LangGraph uses stream events to capture agent execution. UAEF's LangGraph adapter extracts messages, tool calls, and metadata from these events.

### Basic Usage

```python
from uaef.adapters import get_adapter

# Get the LangGraph adapter
adapter = get_adapter("langgraph")

# Your LangGraph output
langgraph_output = {
    "stream_events": [
        {
            "agent": {
                "messages": {
                    "content": "I'll search for that information.",
                    "tool_calls": [
                        {
                            "name": "search_database",
                            "args": {"query": "customer data"}
                        }
                    ],
                    "usage_metadata": {
                        "input_tokens": 150,
                        "output_tokens": 75
                    }
                }
            }
        }
    ],
    "session_id": "session_123"
}

# Transform to canonical format
trace = adapter.transform_to_canonical(langgraph_output)

# Now evaluate
from uaef.evaluation import SingleAgentEvaluator
evaluator = SingleAgentEvaluator()
result = evaluator.evaluate(trace)
```

### Advanced Configuration

```python
from uaef.adapters.langgraph import LangGraphAdapter

# Custom node names
adapter = LangGraphAdapter(
    agent_node_name="my_agent",
    tool_node_name="my_tools"
)

# Transform with custom configuration
trace = adapter.transform_to_canonical(
    langgraph_output,
    extract_metadata=True,
    include_intermediate_steps=True
)
```

### LangGraph Example: Complete Workflow

```python
from langgraph.graph import StateGraph
from uaef.adapters import get_adapter
from uaef.evaluation import SingleAgentEvaluator
from uaef.models import EvaluationInput

# 1. Run your LangGraph agent
graph = StateGraph(...)
result = graph.invoke({"input": "What's the weather?"})

# 2. Transform to UAEF format
adapter = get_adapter("langgraph")
trace = adapter.transform_to_canonical(result)

# 3. Evaluate
evaluator = SingleAgentEvaluator(
    metrics=["tool_accuracy", "answer_relevance"]
)
evaluation_result = evaluator.evaluate(EvaluationInput(trace=trace))

print(f"Score: {evaluation_result.overall_score:.2f}")
```

## Bedrock Agents Integration

### Overview

AWS Bedrock Agents provide managed agent capabilities. UAEF extracts traces from Bedrock's invoke_agent response.

### Basic Usage

```python
import boto3
from uaef.adapters import get_adapter

# Invoke Bedrock Agent
bedrock_client = boto3.client('bedrock-agent-runtime')
response = bedrock_client.invoke_agent(
    agentId='your-agent-id',
    agentAliasId='your-alias-id',
    sessionId='session-123',
    inputText='Show me customer orders'
)

# Transform to UAEF format
adapter = get_adapter("bedrock")
trace = adapter.transform_to_canonical(response)

# Evaluate
from uaef.evaluation import SingleAgentEvaluator
evaluator = SingleAgentEvaluator()
result = evaluator.evaluate(trace)
```

### Handling Streaming Responses

```python
from uaef.adapters.bedrock import BedrockAgentAdapter

adapter = BedrockAgentAdapter()

# Collect streaming events
events = []
for event in response['completion']:
    events.append(event)

# Transform all events
bedrock_data = {
    'event_stream': events,
    'user_input': 'Show me customer orders',
    'session_id': 'session-123',
    'agent_id': 'your-agent-id',
    'alias_id': 'your-alias-id'
}

trace = adapter.transform_to_canonical(bedrock_data)
```

### Extracting Orchestration Steps

```python
# Bedrock provides detailed orchestration traces
trace = adapter.transform_to_canonical(
    bedrock_data,
    include_orchestration=True
)

# Access orchestration metadata
print(f"Orchestration steps: {trace.metadata.get('orchestration_steps')}")
print(f"Action groups used: {trace.metadata.get('action_groups')}")
```

## LangChain Integration

### Overview

LangChain uses callbacks to track agent execution. UAEF extracts traces from callback data.

### Basic Usage

```python
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
from uaef.adapters import get_adapter

# Run LangChain agent with callback
with get_openai_callback() as cb:
    result = agent_executor.invoke({"input": "What's the weather?"})
    
    # Prepare data for adapter
    langchain_data = {
        "result": result,
        "callback_data": cb,
        "agent_type": "openai-functions"
    }

# Transform to UAEF format
adapter = get_adapter("langchain")
trace = adapter.transform_to_canonical(langchain_data)

# Evaluate
from uaef.evaluation import SingleAgentEvaluator
evaluator = SingleAgentEvaluator()
evaluation_result = evaluator.evaluate(trace)
```

### LangChain with Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler
from uaef.adapters.langchain import LangChainAdapter

class UAEFCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.messages = []
        self.tool_calls = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Track LLM calls
        pass
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        # Track tool calls
        self.tool_calls.append({
            "name": serialized.get("name"),
            "input": input_str
        })

# Use custom callback
callback = UAEFCallbackHandler()
result = agent_executor.invoke(
    {"input": "Search for data"},
    callbacks=[callback]
)

# Transform with callback data
adapter = LangChainAdapter()
trace = adapter.transform_to_canonical({
    "result": result,
    "messages": callback.messages,
    "tool_calls": callback.tool_calls
})
```

## Generic JSON Adapter

### Overview

For custom frameworks or APIs, use the Generic JSON adapter with schema mapping.

### Basic Usage

```python
from uaef.adapters.generic import GenericJSONAdapter

# Define how to extract data from your JSON
schema_mapping = {
    "messages": {
        "path": "$.conversation.messages[*]",
        "role_field": "sender",
        "content_field": "text",
        "timestamp_field": "created_at"
    },
    "tool_calls": {
        "path": "$.execution.tools[*]",
        "name_field": "tool_name",
        "arguments_field": "params",
        "result_field": "output"
    },
    "metadata": {
        "input_tokens": "$.usage.input",
        "output_tokens": "$.usage.output",
        "latency": "$.performance.duration"
    },
    "session_id": "$.session.id"
}

# Create adapter
adapter = GenericJSONAdapter(schema_mapping)

# Your custom JSON data
custom_data = {
    "conversation": {
        "messages": [
            {
                "sender": "user",
                "text": "Find customer data",
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "sender": "assistant",
                "text": "I found 5 customers.",
                "created_at": "2024-01-15T10:30:05Z"
            }
        ]
    },
    "execution": {
        "tools": [
            {
                "tool_name": "search_customers",
                "params": {"query": "customer data"},
                "output": {"count": 5}
            }
        ]
    },
    "usage": {"input": 120, "output": 80},
    "performance": {"duration": 1.5},
    "session": {"id": "session_789"}
}

# Transform
trace = adapter.transform_to_canonical(custom_data)
```

### JSONPath Expressions

The Generic adapter uses JSONPath for flexible field extraction:

```python
# Common JSONPath patterns
schema_mapping = {
    # Array of items
    "messages": {
        "path": "$.messages[*]",  # All messages
        "role_field": "role"
    },
    
    # Nested fields
    "metadata": {
        "input_tokens": "$.metadata.usage.input_tokens",
        "output_tokens": "$.metadata.usage.output_tokens"
    },
    
    # Conditional selection
    "tool_calls": {
        "path": "$.events[?(@.type=='tool_call')]",  # Filter by type
        "name_field": "tool.name"
    },
    
    # Direct value
    "session_id": "$.session_id"  # Simple field
}
```

### Complex Nested Structures

```python
# Handle deeply nested data
schema_mapping = {
    "messages": {
        "path": "$.data.conversation.turns[*].messages[*]",
        "role_field": "metadata.role",
        "content_field": "content.text",
        "timestamp_field": "metadata.timestamp"
    },
    "tool_calls": {
        "path": "$.data.execution.steps[*].actions[?(@.type=='tool')]",
        "name_field": "tool.identifier",
        "arguments_field": "tool.parameters",
        "result_field": "tool.response.data"
    }
}
```

## Langfuse Integration

### Overview

Langfuse provides observability for LLM applications. UAEF can extract traces from Langfuse observations.

### Basic Usage

```python
from langfuse import Langfuse
from uaef.adapters import get_adapter

# Initialize Langfuse
langfuse = Langfuse()

# Get trace from Langfuse
trace_id = "your-trace-id"
langfuse_trace = langfuse.get_trace(trace_id)

# Transform to UAEF format
adapter = get_adapter("langfuse")
uaef_trace = adapter.transform_to_canonical({
    "trace": langfuse_trace,
    "observations": langfuse_trace.observations
})

# Evaluate
from uaef.evaluation import SingleAgentEvaluator
evaluator = SingleAgentEvaluator()
result = evaluator.evaluate(uaef_trace)
```

### Extracting Spans and Generations

```python
from uaef.adapters.langfuse import LangfuseAdapter

adapter = LangfuseAdapter()

# Transform with detailed span information
trace = adapter.transform_to_canonical(
    langfuse_data,
    include_spans=True,
    include_generations=True
)

# Access span metadata
print(f"Total spans: {len(trace.metadata.get('spans', []))}")
print(f"LLM generations: {len(trace.metadata.get('generations', []))}")
```

## Creating Custom Adapters

### When to Create a Custom Adapter

Create a custom adapter when:
- Your framework isn't supported
- You need special transformation logic
- You want to extract custom metadata

### Custom Adapter Template

```python
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict

from uaef.adapters.base import BaseAdapter
from uaef.models import AgentTrace, Message, MessageRole, ToolCall

class MyCustomAdapter(BaseAdapter):
    """Adapter for MyCustomFramework."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
    
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """Transform custom data to AgentTrace."""
        
        # Extract messages
        messages = self._extract_messages(raw_data)
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(raw_data)
        
        # Extract metadata
        metadata = self._extract_metadata(raw_data)
        
        # Create AgentTrace
        return AgentTrace(
            trace_id=uuid4(),
            framework="my_custom_framework",
            messages=messages,
            tool_calls=tool_calls,
            session_id=raw_data.get("session_id"),
            input_tokens=metadata.get("input_tokens", 0),
            output_tokens=metadata.get("output_tokens", 0),
            latency=metadata.get("latency", 0.0),
            metadata=metadata
        )
    
    def _extract_messages(self, raw_data: Dict[str, Any]) -> list[Message]:
        """Extract messages from raw data."""
        messages = []
        
        for msg in raw_data.get("messages", []):
            messages.append(Message(
                role=self._map_role(msg["role"]),
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"])
            ))
        
        return messages
    
    def _extract_tool_calls(self, raw_data: Dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from raw data."""
        tool_calls = []
        
        for tool in raw_data.get("tools", []):
            tool_calls.append(ToolCall(
                name=tool["name"],
                arguments=tool["args"],
                result=tool.get("result"),
                timestamp=datetime.fromisoformat(tool["timestamp"])
            ))
        
        return tool_calls
    
    def _extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from raw data."""
        return {
            "input_tokens": raw_data.get("usage", {}).get("input", 0),
            "output_tokens": raw_data.get("usage", {}).get("output", 0),
            "latency": raw_data.get("latency", 0.0),
            "custom_field": raw_data.get("custom_field")
        }
    
    def _map_role(self, role: str) -> MessageRole:
        """Map custom role to MessageRole enum."""
        role_mapping = {
            "human": MessageRole.USER,
            "ai": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM
        }
        return role_mapping.get(role.lower(), MessageRole.ASSISTANT)
```

### Registering Custom Adapters

```python
from uaef.adapters import register_adapter

# Register your custom adapter
register_adapter("my_custom_framework", MyCustomAdapter)

# Now you can use it like built-in adapters
from uaef.adapters import get_adapter
adapter = get_adapter("my_custom_framework")
```

## Best Practices

### 1. Validate Input Data

```python
def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
    # Validate required fields
    if "messages" not in raw_data:
        raise ValueError("Missing required field: messages")
    
    if not isinstance(raw_data["messages"], list):
        raise ValueError("messages must be a list")
    
    # Continue with transformation
    ...
```

### 2. Handle Missing Data Gracefully

```python
# Use defaults for optional fields
input_tokens = raw_data.get("usage", {}).get("input_tokens", 0)
output_tokens = raw_data.get("usage", {}).get("output_tokens", 0)

# Handle missing tool calls
tool_calls = self._extract_tool_calls(raw_data) if "tools" in raw_data else []
```

### 3. Preserve Original Data

```python
# Store original data in metadata for debugging
metadata = {
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "original_data": raw_data  # Keep for reference
}
```

### 4. Add Logging

```python
import logging

logger = logging.getLogger(__name__)

def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
    logger.info(f"Transforming {self.framework} trace")
    
    try:
        trace = self._do_transform(raw_data)
        logger.info(f"Successfully transformed trace {trace.trace_id}")
        return trace
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise
```

### 5. Test Thoroughly

```python
import pytest
from uaef.adapters import get_adapter

def test_custom_adapter():
    adapter = get_adapter("my_custom_framework")
    
    # Test with valid data
    raw_data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "session_id": "test-123"
    }
    trace = adapter.transform_to_canonical(raw_data)
    
    assert trace.framework == "my_custom_framework"
    assert len(trace.messages) == 1
    assert trace.session_id == "test-123"
    
    # Test with missing data
    with pytest.raises(ValueError):
        adapter.transform_to_canonical({})
```

## Troubleshooting

### Issue: Adapter Not Found

**Error**: `AdapterNotFoundError: No adapter registered for 'my_framework'`

**Solution**: Check available adapters:
```python
from uaef.adapters import list_adapters
print(list_adapters())
```

### Issue: Transformation Fails

**Error**: `AdapterTransformationError: Failed to extract messages`

**Solution**: Validate your data structure:
```python
# Check data structure
print(json.dumps(raw_data, indent=2))

# Verify required fields
required_fields = ["messages", "session_id"]
for field in required_fields:
    if field not in raw_data:
        print(f"Missing field: {field}")
```

### Issue: Missing Tool Calls

**Problem**: Tool calls not extracted

**Solution**: Check tool call format:
```python
# Debug tool extraction
adapter = get_adapter("your_framework")
tool_calls = adapter._extract_tool_calls(raw_data)
print(f"Extracted {len(tool_calls)} tool calls")

for tool in tool_calls:
    print(f"  - {tool.name}: {tool.arguments}")
```

### Issue: Incorrect Token Counts

**Problem**: Token counts are zero or incorrect

**Solution**: Verify metadata extraction:
```python
metadata = adapter._extract_metadata(raw_data)
print(f"Input tokens: {metadata.get('input_tokens')}")
print(f"Output tokens: {metadata.get('output_tokens')}")

# Check raw data structure
print(f"Raw usage data: {raw_data.get('usage')}")
```

## Next Steps

Now that you can integrate your agent framework:

1. **[Custom Metrics Guide](custom-metrics.md)**: Create custom evaluation metrics
2. **[Experiment Workflow Guide](experiments.md)**: Track agent versions
3. **[HITL Workflow Guide](hitl.md)**: Add human feedback

## Additional Resources

- **API Reference**: Complete adapter API documentation
- **Examples**: Sample adapters in `examples/adapter_usage.py`
- **Source Code**: Adapter implementations in [`src/uaef/adapters/`](https://github.com/awslabs/uaef/tree/main/src/uaef/adapters)

---

**Ready to create custom metrics?** Continue to the [Custom Metrics Guide](custom-metrics.md).
