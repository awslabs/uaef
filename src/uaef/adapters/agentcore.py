# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore adapter for transforming AgentCore traces to canonical format.

Supports two modes:
1. Pre-collected traces — pass raw_data with ``trace``, ``final_response``, etc.
2. Live invocation — use the ``invoke()`` helper to call an AgentCore-deployed
   agent via ``bedrock-agentcore:invoke_agent_runtime`` and automatically
   collect the streaming response.

Multi-agent auto-detection:
    When ``raw_data`` contains an ``"agents"`` key mapping agent names to their
    individual response dicts, ``transform_to_canonical()`` automatically returns
    a ``MultiAgentTrace`` with per-agent traces and inferred coordination events.
    This mirrors the LangGraph adapter's auto-detect pattern.
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.multi_agent_trace import CoordinationEvent, MultiAgentTrace, WorkflowStatus
from uaef.models.tool_call import ToolCall


class AgentCoreAdapter(BaseAdapter):
    """
    Adapter for transforming AgentCore ``invoke_agent_runtime`` responses to
    canonical AgentTrace format.

    AgentCore agents are identified by their **runtime ARN** (or runtime ID +
    account ID).  The adapter calls the ``bedrock-agentcore`` service client,
    *not* the legacy ``bedrock-agent-runtime`` client.

    Expected raw_data structure::

        {
            "user_input": "...",
            "final_response": "..." | {...},
            "latency": 1.234,
            "trace": [...],               # CloudWatch log entries (JSON strings, optional)
            "response_chunks": [...],      # Raw streaming chunks (optional)
            "trace_id": "...",
            "session_id": "...",
            "agent_runtime_id": "...",     # Runtime ARN or ID
        }
    """

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "agentcore"

    # ------------------------------------------------------------------
    # Live invocation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def invoke(
        agent_runtime_arn: str,
        user_input: str,
        agent_version: str = None,
        *,
        region: Optional[str] = None,
        session_id: Optional[str] = None,
        qualifier: Optional[str] = None,
        bearer_token: Optional[str] = None,
        auto_detect_auth: bool = True,
        payload_override: Optional[Dict[str, Any]] = None,
        boto_session=None,
        log_stream_name: Optional[str] = None,
        fetch_logs: bool = True,
    ) -> Dict[str, Any]:
        """Invoke an AgentCore-deployed agent and collect its streaming response.

        Supports both authentication modes:

        * **SigV4 (default)** — uses the ``bedrock-agentcore`` boto3 client.
          Works when the agent runtime is configured for IAM auth.
        * **OAuth / JWT bearer token** — when *bearer_token* is provided the
          adapter makes a raw HTTPS ``POST`` to the AgentCore data-plane
          endpoint instead, because the AWS SDK always signs with SigV4 which
          is rejected by OAuth-configured runtimes.

        When *auto_detect_auth* is True (default) and no *bearer_token* is
        provided, the adapter will first try SigV4.  If that fails with an
        authorization mismatch, it queries the runtime's authorizer config
        and attempts to auto-fetch a Cognito token via client_credentials
        grant before retrying with OAuth.

        Args:
            agent_runtime_arn: The AgentCore Runtime ARN.
            user_input: The prompt to send.
            region: AWS region (reads from UAEF config if not provided).
            session_id: Optional runtime session ID (generated if omitted).
            qualifier: Optional qualifier / endpoint name.
            bearer_token: If the agent runtime uses OAuth/JWT inbound auth,
                pass the access token here.  When set, the invocation is made
                via HTTPS with an ``Authorization: Bearer`` header instead of
                the boto3 SDK.
            auto_detect_auth: If True and SigV4 fails with auth mismatch,
                automatically detect OAuth config and retry. Default True.
            payload_override: Optional dict to use as the JSON payload instead
                of the default ``{"prompt": ..., "runtimeSessionId": ...}``.
                Use this when your agent container expects a custom payload
                format.
            boto_session: Optional ``boto3.Session`` (SigV4 mode only).

        Returns:
            A dict suitable for ``transform_to_canonical()``.
        """
        from uaef.config import get_config

        if region is None:
            region = get_config().aws.region

        session_id = session_id or str(uuid4())

        # Build payload — use override if provided, otherwise default format
        if payload_override is not None:
            payload_dict = payload_override
        else:
            payload_dict = {
                "prompt": user_input,
                "runtimeSessionId": session_id,
            }

        if bearer_token:
            return AgentCoreAdapter._invoke_oauth(
                agent_runtime_arn=agent_runtime_arn,
                user_input=user_input,
                bearer_token=bearer_token,
                payload_dict=payload_dict,
                region=region,
                session_id=session_id,
                qualifier=qualifier,
                agent_version=agent_version,
                log_stream_name=log_stream_name,
            )

        # Try SigV4 first
        try:
            return AgentCoreAdapter._invoke_sigv4(
                agent_runtime_arn=agent_runtime_arn,
                user_input=user_input,
                agent_version=agent_version,
                payload_dict=payload_dict,
                region=region,
                session_id=session_id,
                qualifier=qualifier,
                boto_session=boto_session,
                log_stream_name=log_stream_name,
                fetch_logs=fetch_logs,
            )
        except Exception as e:
            if not auto_detect_auth:
                raise
            error_msg = str(e)
            if "Authorization method mismatch" not in error_msg and "explicit deny" not in error_msg:
                raise

        # SigV4 failed with auth mismatch — attempt OAuth auto-detection
        token = AgentCoreAdapter._auto_detect_oauth_token(
            agent_runtime_arn=agent_runtime_arn,
            region=region,
            boto_session=boto_session,
        )
        if not token:
            raise RuntimeError(
                f"SigV4 auth failed and OAuth auto-detection could not fetch a token. "
                f"Provide a bearer_token manually."
            )
        return AgentCoreAdapter._invoke_oauth(
            agent_runtime_arn=agent_runtime_arn,
            user_input=user_input,
            bearer_token=token,
            payload_dict=payload_dict,
            region=region,
            session_id=session_id,
            qualifier=qualifier,
            agent_version=agent_version,
            log_stream_name=log_stream_name,
            fetch_logs=fetch_logs,
        )

    # ---- OAuth auto-detection ----------------------------------------

    @staticmethod
    def _auto_detect_oauth_token(
        agent_runtime_arn: str,
        region: str,
        boto_session=None,
    ) -> str:
        """Attempt to auto-detect OAuth config and fetch a Cognito token.

        Queries the runtime's authorizer configuration and tries the
        client_credentials grant on the discovered token endpoint.

        Returns:
            Access token string, or empty string if detection fails.
        """
        import base64
        import urllib.parse
        import urllib.request

        try:
            import boto3
        except ImportError:
            return ""

        try:
            session = boto_session or boto3.Session(region_name=region)
            runtime_id = agent_runtime_arn.split("/")[-1]
            control = session.client("bedrock-agentcore-control", region_name=region)
            info = control.get_agent_runtime(agentRuntimeId=runtime_id)
            auth_config = info.get("authorizerConfiguration", {})

            if not auth_config:
                return ""

            issuer = auth_config.get("issuerUrl", auth_config.get("issuer", ""))
            audience = auth_config.get("allowedAudiences", auth_config.get("audience", []))
            client_id = auth_config.get("clientId", "")
            client_secret = auth_config.get("clientSecret", "")

            if not issuer or not client_id or not client_secret:
                return ""

            token_url = issuer.rstrip("/") + "/oauth2/token"
            if urllib.parse.urlsplit(token_url).scheme != "https":
                # Never send the client secret over a non-https token endpoint.
                raise ValueError(f"Refusing non-https token URL: {token_url!r}")
            auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "scope": " ".join(audience) if audience else "",
            }).encode()

            req = urllib.request.Request(token_url, data=data, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {auth_header}",
            })

            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- False positive: token_url scheme is validated to https above; endpoint is the first-party Cognito/AgentCore token URL, not user-controlled input.
            with urllib.request.urlopen(req) as resp:  # nosec B310 - token_url scheme validated to https above
                body = json.loads(resp.read().decode())
                return body.get("access_token", "")

        except Exception:
            return ""

    # ---- SigV4 path (default) ----------------------------------------

    @classmethod
    def _invoke_sigv4(
        cls,
        agent_runtime_arn: str,
        user_input: str,
        agent_version: str,
        payload_dict: Dict[str, Any],
        *,
        region: str,
        session_id: str,
        qualifier: Optional[str] = None,
        boto_session=None,
        log_stream_name: Optional[str] = None,
        fetch_logs: bool = True,
    ) -> Dict[str, Any]:
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for SigV4 invocation — pip install boto3")

        session = boto_session or boto3.Session(region_name=region)
        client = session.client("bedrock-agentcore", region_name=region)

        payload = json.dumps(payload_dict).encode()

        # Generate a trace ID client-side so we can query logs by it later
        trace_id = cls.generate_xray_trace_id()

        invoke_kwargs: Dict[str, Any] = {
            "agentRuntimeArn": agent_runtime_arn,
            "runtimeSessionId": session_id,
            "traceId": trace_id,
            "payload": payload,
        }
        if qualifier:
            invoke_kwargs["qualifier"] = qualifier

        start = time.time()
        response = client.invoke_agent_runtime(**invoke_kwargs)
        latency = time.time() - start

        content = []
        for chunk in response.get("response", []):
            content.append(chunk.decode('utf-8'))

        joined_content = ''.join(content)
        if joined_content:
            try:
                final_response = json.loads(joined_content)
            except json.JSONDecodeError:
                # Response may not be JSON — treat as plain text
                final_response = joined_content
        else:
            final_response = ""

        # Optionally enrich with CloudWatch otel logs (tool/multi-agent info).
        # This polls CloudWatch Logs Insights for up to several minutes per
        # invocation, so it is skipped unless explicitly requested AND an
        # ``agent_version`` is supplied. Without ``agent_version`` the log-group
        # name would be ``...-None`` (a group that never exists), making every
        # StartQuery fail and the poll loop burn its full timeout for nothing —
        # which, across a multi-row batch, can exceed a caller's time budget
        # (e.g. a 15-minute Lambda). Callers that only need the response text
        # (most response-quality evaluation) should pass ``fetch_logs=False``.
        trace: List[str] = []
        tools_info: Dict[str, Any] = {}
        if fetch_logs and agent_version:
            # Extract agent_id from the ARN
            # ARN: arn:aws:bedrock-agentcore:<region>:<account>:runtime/<name>
            arn_parts = agent_runtime_arn.split("/")
            agent_id = arn_parts[1] if len(arn_parts) > 1 else "unknown"

            your_log_groups = [f"/aws/bedrock-agentcore/runtimes/{agent_id}-{agent_version}"]

            # Query logs by the trace ID we passed to the invocation.
            # The trace ID appears in the otel logs, stripped of the X-Ray prefix.
            your_trace_id = trace_id.replace("Root=1-", "").split(";")[0].replace("-", "")

            # Poll for logs using the trace ID (matching helper.py approach)
            timeout_minutes = 3
            check_interval_seconds = 5
            query_start_time = datetime.now()
            timeout_time = query_start_time + timedelta(minutes=timeout_minutes)

            while datetime.now() < timeout_time:
                trace = cls.get_logs_by_trace_id(
                    your_trace_id, your_log_groups,
                    region_name=region,
                    log_stream_name=log_stream_name,
                )
                if trace and len(trace) > 0:
                    break
                time.sleep(check_interval_seconds)

            # Final fetch after additional wait for all logs to flush
            time.sleep(10)
            trace = cls.get_logs_by_trace_id(
                your_trace_id, your_log_groups,
                region_name=region,
                log_stream_name=log_stream_name,
            )

            # Auto-extract tool info from the logs
            tools_info = cls.extract_tools_info(trace)

        return {
            "user_input": user_input,
            "final_response": final_response,
            "latency": latency,
            "trace": trace,
            "tools_info": tools_info,
            "trace_id": response.get("ResponseMetadata", {}).get("RequestId", ""),
            "session_id": response.get("runtimeSessionId", session_id),
            "agent_runtime_id": agent_runtime_arn,
        }

    # ---- OAuth / JWT bearer-token path --------------------------------

    @staticmethod
    def _invoke_oauth(
        agent_runtime_arn: str,
        user_input: str,
        bearer_token: str,
        payload_dict: Dict[str, Any],
        *,
        region: str,
        session_id: str,
        qualifier: Optional[str] = None,
        agent_version: Optional[str] = None,
        log_stream_name: Optional[str] = None,
        fetch_logs: bool = True,
    ) -> Dict[str, Any]:
        """HTTPS invocation for OAuth-configured agent runtimes.

        The AWS SDK cannot be used when the runtime expects a JWT bearer
        token because the SDK always signs with SigV4.  Instead we POST
        directly to the AgentCore data-plane endpoint.

        Handles SSE (Server-Sent Events) streaming responses where each line
        is formatted as ``data: {...}``.  Extracts the final text from the
        ``{"data": "..."}`` chunks emitted by Strands-based agents.

        When *fetch_logs* is True and *agent_version* is provided, polls
        CloudWatch for otel logs after invocation to enable multi-agent
        detection from log patterns.
        """
        try:
            import urllib.request
            import urllib.parse
            import urllib.error
        except ImportError:
            raise  # stdlib — should never fail

        # Build the URL:
        # POST /runtimes/{agentRuntimeArn}/invocations
        escaped_arn = urllib.parse.quote(agent_runtime_arn, safe="")
        url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_arn}/invocations"
        if qualifier:
            url += f"?qualifier={urllib.parse.quote(qualifier, safe='')}"
        if urllib.parse.urlsplit(url).scheme != "https":
            raise ValueError(f"Refusing to open non-https URL: {url!r}")

        payload = json.dumps(payload_dict).encode()

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {bearer_token}",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        start = time.time()
        try:
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- False positive: url scheme is validated to https above; endpoint is the first-party AgentCore runtime URL, not user-controlled input.
            with urllib.request.urlopen(req) as resp:  # nosec B310 - url scheme validated to https above
                latency = time.time() - start
                body = resp.read().decode("utf-8")
                resp_session_id = resp.headers.get(
                    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id", session_id
                )
                request_id = resp.headers.get("x-amzn-RequestId", "")
        except urllib.error.HTTPError as e:
            body_bytes = e.read() if hasattr(e, "read") else b""
            raise AdapterTransformationError(
                f"AgentCore OAuth invocation failed ({e.code}): {body_bytes.decode('utf-8', errors='replace')}",
                adapter_name="agentcore",
            )

        # Parse SSE response: lines formatted as "data: {...}"
        response_chunks: List[str] = []
        text_parts: List[str] = []

        for line in body.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            json_str = line[6:]  # strip "data: " prefix
            try:
                chunk = json.loads(json_str)
                response_chunks.append(json_str)

                # Strands format: {"data": "text content"}
                if isinstance(chunk, dict) and "data" in chunk and isinstance(chunk["data"], str):
                    text_parts.append(chunk["data"])
            except json.JSONDecodeError:
                # Not JSON — treat as raw text
                response_chunks.append(json_str)
                text_parts.append(json_str)

        # If no SSE lines found, fall back to parsing body as plain JSON/text
        if not response_chunks:
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    text_parts.append(parsed.get("response", parsed.get("output", parsed.get("data", body))))
                else:
                    text_parts.append(body)
                response_chunks.append(body)
            except json.JSONDecodeError:
                text_parts.append(body)
                response_chunks.append(body)

        final_text = "".join(text_parts)

        # Optionally fetch CloudWatch logs for multi-agent detection
        trace: List[str] = []
        tools_info: Dict[str, Any] = {}
        if fetch_logs and agent_version:
            arn_parts = agent_runtime_arn.split("/")
            agent_id = arn_parts[1] if len(arn_parts) > 1 else "unknown"
            log_groups = [f"/aws/bedrock-agentcore/runtimes/{agent_id}-{agent_version}"]

            # Wait for logs to flush to CloudWatch (typically 15-30s after invocation)
            time.sleep(15)

            # Step 1: Query by session_id to find the otel traceId
            # (strands.telemetry.tracer entries have session.id in attributes)
            timeout_minutes = 3
            check_interval_seconds = 10
            query_start_time = datetime.now()
            timeout_time = query_start_time + timedelta(minutes=timeout_minutes)
            otel_trace_id = None

            while datetime.now() < timeout_time:
                partial = AgentCoreAdapter.get_logs_by_session_id(
                    session_id, log_groups,
                    region_name=region,
                    log_stream_name=log_stream_name,
                    invocation_time=start,
                )
                if partial:
                    # Extract the otel traceId from any entry
                    otel_trace_id = AgentCoreAdapter._extract_otel_trace_id(partial)
                    if otel_trace_id:
                        break
                time.sleep(check_interval_seconds)

            # Step 2: If we found a traceId, query ALL logs for that trace
            # (this picks up __main__ scope entries like [Step N] that don't
            # have session.id but DO share the same traceId)
            if otel_trace_id:
                trace = AgentCoreAdapter.get_logs_by_trace_id(
                    otel_trace_id, log_groups,
                    region_name=region,
                    log_stream_name=log_stream_name,
                )
            else:
                # Fallback: use whatever we found by session_id
                trace = partial if partial else []

            # Extract tool info from logs
            if trace:
                tools_info = AgentCoreAdapter.extract_tools_info(trace)

        return {
            "user_input": user_input,
            "final_response": final_text,
            "latency": latency,
            "trace": trace,
            "tools_info": tools_info,
            "response_chunks": response_chunks,
            "trace_id": request_id,
            "session_id": resp_session_id,
            "agent_runtime_id": agent_runtime_arn,
        }

    # ---- Shared response reader --------------------------------------

    @staticmethod
    def _read_response_body(response: Dict[str, Any]) -> List[str]:
        """Read chunks from a boto3 streaming response."""
        content_type = response.get("contentType", "")
        response_body = response.get("response", b"")
        chunks: List[str] = []

        if "text/event-stream" in content_type:
            for line in response_body.iter_lines(chunk_size=10):
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    chunks.append(decoded)
        elif content_type == "application/json":
            parts = []
            for chunk in response_body:
                parts.append(chunk.decode("utf-8"))
            chunks.append("".join(parts))
        else:
            if hasattr(response_body, "read"):
                chunks.append(response_body.read().decode("utf-8"))
            elif isinstance(response_body, bytes):
                chunks.append(response_body.decode("utf-8"))

        return chunks

    # ------------------------------------------------------------------
    # Multi-agent invocation helper
    # ------------------------------------------------------------------

    # @staticmethod
    # def invoke_multi(
    #     agents: Dict[str, Dict[str, Any]],
    #     user_input: str,
    #     *,
    #     region: Optional[str] = None,
    #     session_id: Optional[str] = None,
    #     chain: bool = True,
    #     boto_session=None,
    # ) -> Dict[str, Any]:
    #     """Invoke multiple AgentCore-deployed agents and collect responses.

    #     This produces a raw_data dict with an ``"agents"`` key that
    #     ``transform_to_canonical()`` auto-detects as multi-agent, returning
    #     a ``MultiAgentTrace``.

    #     Args:
    #         agents: Mapping of agent name → config dict. Each config must have
    #             ``"arn"`` and may include ``"bearer_token"``, ``"qualifier"``,
    #             ``"payload_builder"``, ``"payload_override"``.
    #         user_input: The user prompt to send.
    #         region: AWS region (reads from UAEF config if not provided).
    #         session_id: Optional session ID (generated if omitted).
    #         chain: If True (default), each agent receives the previous agent's
    #             response as additional context. If False, all agents receive
    #             the original user_input.
    #         boto_session: Optional ``boto3.Session`` (SigV4 mode only).

    #     Returns:
    #         A dict suitable for ``transform_to_canonical()`` with auto-detected
    #         multi-agent structure::

    #             {
    #                 "user_input": "...",
    #                 "agents": {
    #                     "agent_name": { <single-agent raw_data> },
    #                     ...
    #                 },
    #                 "agent_order": ["agent1", "agent2", ...],
    #                 "session_id": "...",
    #                 "latency": <total>,
    #             }
    #     """
    #     from uaef.config import get_config

    #     if region is None:
    #         region = get_config().aws.region

    #     session_id = session_id or str(uuid4())
    #     agent_results: Dict[str, Dict[str, Any]] = {}
    #     agent_order: List[str] = []
    #     total_latency = 0.0
    #     current_input = user_input

    #     for agent_name, cfg in agents.items():
    #         agent_order.append(agent_name)

    #         # Build payload override if a builder is provided
    #         payload_override = cfg.get("payload_override")
    #         if "payload_builder" in cfg:
    #             payload_override = cfg["payload_builder"](current_input, session_id)

    #         raw = AgentCoreAdapter.invoke(
    #             agent_runtime_arn=cfg["arn"],
    #             user_input=current_input,
    #             region=region,
    #             session_id=session_id,
    #             bearer_token=cfg.get("bearer_token"),
    #             qualifier=cfg.get("qualifier"),
    #             payload_override=payload_override,
    #             boto_session=boto_session,
    #         )

    #         agent_results[agent_name] = raw
    #         total_latency += raw.get("latency", 0.0)

    #         # Chain: pass this agent's response as context to the next
    #         if chain and raw.get("final_response"):
    #             response_text = AgentCoreAdapter._extract_text_from_response(
    #                 raw["final_response"]
    #             )
    #             current_input = (
    #                 f"Previous agent ({agent_name}) responded:\n"
    #                 f"{response_text}\n\n"
    #                 f"Original question: {user_input}"
    #             )

    #     return {
    #         "user_input": user_input,
    #         "agents": agent_results,
    #         "agent_order": agent_order,
    #         "session_id": session_id,
    #         "latency": total_latency,
    #     }

    # ------------------------------------------------------------------
    # Canonical transformation
    # ------------------------------------------------------------------

    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> Union[AgentTrace, MultiAgentTrace]:
        """Transform AgentCore response data to canonical format.

        Auto-detects multi-agent traces when ``raw_data`` contains an
        ``"agents"`` key mapping agent names to their individual response
        dicts. Otherwise returns a single ``AgentTrace``.

        This mirrors the LangGraph adapter's auto-detect pattern: the caller
        does not need to specify agent node names — the adapter infers the
        multi-agent structure from the data shape.

        Args:
            raw_data: AgentCore response data. For multi-agent, include::

                {
                    "user_input": "...",
                    "agents": {
                        "agent_name": { <single-agent raw_data> },
                        ...
                    },
                    "agent_order": ["agent1", "agent2", ...],  # optional
                    "session_id": "...",
                }

        Returns:
            AgentTrace for single-agent, MultiAgentTrace for multi-agent.
        """
        try:
            # Auto-detect: if "agents" key is present with multiple entries,
            agents_data = raw_data.get("agents")
            if isinstance(agents_data, dict) and len(agents_data) > 0:
                return self._transform_multi_agent(raw_data)

            # Auto-detect from otel logs: if trace contains multiple agent
            trace_logs = raw_data.get("trace", [])
            if trace_logs:
                detected_agents = self._detect_agents_from_otel_logs(trace_logs)
                if len(detected_agents) > 1:
                    # Restructure raw_data into multi-agent format
                    multi_raw = self._build_multi_agent_raw_data(raw_data, detected_agents)
                    return self._transform_multi_agent(multi_raw)

            return self._transform_single_agent(raw_data)

        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform AgentCore trace: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    def _transform_single_agent(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """Transform single-agent response to AgentTrace (original behavior)."""
        messages = self.extract_messages(raw_data)
        tool_calls = self.extract_tool_calls(raw_data)
        metadata = self.extract_metadata(raw_data)

        # If tools_info is present (from invoke with otel log extraction),
        # use it to populate tool_calls — it's more complete than the legacy path.
        tools_info = raw_data.get("tools_info")
        if tools_info and tools_info.get("tool_calls") and not tool_calls:
            for tc_info in tools_info["tool_calls"]:
                result_str = None
                if tc_info.get("response") is not None:
                    resp = tc_info["response"]
                    if isinstance(resp, list):
                        # Extract text from content list: [{"text": "..."}]
                        texts = []
                        for item in resp:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                        result_str = "\n".join(texts) if texts else str(resp)
                    elif isinstance(resp, str):
                        result_str = resp
                    else:
                        result_str = str(resp)

                tool_calls.append(
                    ToolCall(
                        name=tc_info.get("name", "unknown"),
                        arguments=tc_info.get("args", {}),
                        result=result_str,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            # Also update token usage from tools_info if available
            token_usage = tools_info.get("token_usage", {})
            if token_usage.get("input_tokens") and not metadata.get("input_tokens"):
                metadata["input_tokens"] = token_usage["input_tokens"]
            if token_usage.get("output_tokens") and not metadata.get("output_tokens"):
                metadata["output_tokens"] = token_usage["output_tokens"]

        trace_id_str = raw_data.get("trace_id")
        if trace_id_str:
            trace_id_str = str(trace_id_str).replace("Root=1-", "").split(";")[0].replace("-", "")

        trace = AgentTrace(
            trace_id=uuid4(),
            session_id=raw_data.get("session_id"),
            messages=messages,
            tool_calls=tool_calls,
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            latency=metadata.get("latency"),
            cost=metadata.get("cost"),
            metadata=metadata,
            framework="agentcore",
            timestamp=datetime.now(timezone.utc),
        )

        if trace_id_str:
            trace.metadata["xray_trace_id"] = trace_id_str

        return trace

    # ------------------------------------------------------------------
    # Auto-detect multi-agent from otel logs
    # ------------------------------------------------------------------

    def _detect_agents_from_otel_logs(self, trace_logs: List[str]) -> List[Dict[str, Any]]:
        """Detect individual agent steps from otel log entries.

        Looks for patterns like "[Step N] Agent Name analyzing..." in the log
        body text, and also detects agent boundaries from span transitions
        in the strands.telemetry.tracer scope.

        Returns:
            List of dicts with agent info, ordered by appearance:
            [{"name": "trend_agent", "display_name": "Trend Agent", "start_idx": 0, "end_idx": 10}, ...]
        """
        import re

        agents: List[Dict[str, Any]] = []
        step_pattern = re.compile(r'\[Step\s+(\d+)\]\s+(.+?)(?:\s+analyzing|\s+consolidating|\s+processing|\.\.\.)', re.IGNORECASE)

        for idx, entry in enumerate(trace_logs):
            try:
                entry_data = json.loads(entry)
            except (json.JSONDecodeError, TypeError):
                continue

            body = entry_data.get("body", "")
            if not isinstance(body, str):
                continue

            match = step_pattern.search(body)
            if match:
                step_num = int(match.group(1))
                agent_display_name = match.group(2).strip()
                # Normalize to a key name
                agent_key = agent_display_name.lower().replace(" ", "_")

                # Close the previous agent's range
                if agents:
                    agents[-1]["end_idx"] = idx

                agents.append({
                    "name": agent_key,
                    "display_name": agent_display_name,
                    "step": step_num,
                    "start_idx": idx,
                    "end_idx": len(trace_logs),  # will be updated by next agent
                })

        return agents

    def _build_multi_agent_raw_data(self, raw_data: Dict[str, Any], detected_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build multi-agent raw_data structure from detected agent segments.

        Splits the otel trace logs into per-agent segments and constructs
        the expected multi-agent raw_data format with an "agents" key.

        Args:
            raw_data: Original single-invocation raw_data with "trace" logs.
            detected_agents: List from _detect_agents_from_otel_logs.

        Returns:
            Dict in multi-agent format suitable for _transform_multi_agent().
        """
        trace_logs = raw_data.get("trace", [])
        agents_dict: Dict[str, Dict[str, Any]] = {}
        agent_order: List[str] = []

        for agent_info in detected_agents:
            agent_name = agent_info["name"]
            start_idx = agent_info["start_idx"]
            end_idx = agent_info["end_idx"]
            agent_order.append(agent_name)

            # Slice the trace logs for this agent's segment
            agent_logs = trace_logs[start_idx:end_idx]

            # Extract tools_info for this agent's segment
            tools_info = self.extract_tools_info(agent_logs)

            # Build per-agent raw_data
            agents_dict[agent_name] = {
                "user_input": raw_data.get("user_input", ""),
                "final_response": "",  # intermediate agents don't have a final response
                "latency": 0,
                "trace": agent_logs,
                "tools_info": tools_info,
                "session_id": raw_data.get("session_id"),
            }

        # The last agent gets the final_response
        if agent_order and agents_dict:
            last_agent = agent_order[-1]
            agents_dict[last_agent]["final_response"] = raw_data.get("final_response", "")

        # Calculate approximate per-agent latency from timestamps
        total_latency = raw_data.get("latency", 0)
        if agent_order and total_latency:
            per_agent_latency = total_latency / len(agent_order)
            for name in agent_order:
                agents_dict[name]["latency"] = per_agent_latency

        return {
            "user_input": raw_data.get("user_input", ""),
            "agents": agents_dict,
            "agent_order": agent_order,
            "session_id": raw_data.get("session_id"),
            "latency": total_latency,
        }

    def _transform_multi_agent(self, raw_data: Dict[str, Any]) -> MultiAgentTrace:
        """Transform multi-agent response data to MultiAgentTrace.

        Infers agent ordering and coordination events from the ``agents`` dict
        and optional ``agent_order`` list. Each agent's raw data is transformed
        into a separate AgentTrace.
        """
        agents_data = raw_data["agents"]
        agent_order = raw_data.get("agent_order", list(agents_data.keys()))
        session_id = raw_data.get("session_id")
        user_input = raw_data.get("user_input", "")

        # Build per-agent AgentTraces
        agent_traces: Dict[str, AgentTrace] = {}
        for agent_name in agent_order:
            agent_raw = agents_data.get(agent_name)
            if agent_raw is None:
                continue
            agent_traces[agent_name] = self._transform_single_agent(agent_raw)

        # Infer coordination events from sequential agent ordering
        coordination_events: List[CoordinationEvent] = []
        for i in range(1, len(agent_order)):
            from_agent = agent_order[i - 1]
            to_agent = agent_order[i]
            coordination_events.append(CoordinationEvent(
                from_agent=from_agent,
                to_agent=to_agent,
                event_type="HANDOFF",
                message=f"Control passed from {from_agent} to {to_agent}",
                timestamp=datetime.now(timezone.utc),
            ))

        # Final handoff to user
        if agent_order:
            coordination_events.append(CoordinationEvent(
                from_agent=agent_order[-1],
                to_agent="user",
                event_type="RESPONSE",
                message="Final answer delivered to user",
                timestamp=datetime.now(timezone.utc),
            ))

        # Aggregate metadata
        total_latency = raw_data.get("latency")
        if total_latency is None:
            latencies = [
                agents_data[name].get("latency", 0.0)
                for name in agent_order
                if name in agents_data
            ]
            total_latency = sum(latencies) if latencies else None

        return MultiAgentTrace(
            trace_id=uuid4(),
            session_id=session_id,
            agent_traces=agent_traces,
            coordination_events=coordination_events,
            workflow_status=WorkflowStatus.COMPLETED,
            total_latency=total_latency,
            metadata={
                "agents_invoked": agent_order,
                "query": user_input,
                "framework": "agentcore",
            },
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Message extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_response(response) -> str:
        """
        Extract plain text from various AgentCore/Strands response formats.

        Handles:
        - Plain string: "hello"
        - Dict with text: {"response": "hello"} or {"result": "hello"}
        - Bedrock Converse format: {"response": {"role": "assistant", "content": [{"text": "hello"}]}}
        - Nested JSON string: '{"response": {"content": [...]}}'
        """
        if isinstance(response, str):
            # Try parsing as JSON first
            try:
                import json
                parsed = json.loads(response)
                return AgentCoreAdapter._extract_text_from_response(parsed)
            except (json.JSONDecodeError, ValueError):
                return response

        if isinstance(response, dict):
            # Direct text fields
            for key in ("result", "text", "output", "message"):
                if key in response and isinstance(response[key], str):
                    return response[key]

            # Nested response dict (Bedrock Converse format)
            inner = response.get("response")
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                # {"role": "assistant", "content": [{"text": "..."}]}
                content = inner.get("content")
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                    if texts:
                        return "\n".join(texts)
                # {"content": "plain string"}
                if isinstance(content, str):
                    return content
                # Fallback for inner dict
                return AgentCoreAdapter._extract_text_from_response(inner)

            # Direct content field
            content = response.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                if texts:
                    return "\n".join(texts)

            # Last resort: stringify
            return str(response)

        return str(response)

    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """Extract messages from AgentCore response and logs."""
        try:
            messages: List[Message] = []

            user_input = raw_data.get("user_input")
            if user_input:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=user_input,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            # Try to extract intermediate reasoning from trace_events
            for te in raw_data.get("trace_events", []):
                trace_detail = te.get("trace", {})
                orch = trace_detail.get("orchestrationTrace", {})
                rationale = orch.get("rationale", {})
                if rationale and rationale.get("text"):
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=rationale["text"],
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

            final_response = raw_data.get("final_response")
            if final_response:
                content = self._extract_text_from_response(final_response)
                if content:
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=content,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

            return messages

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Tool-call extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from AgentCore trace events and CloudWatch logs."""
        try:
            tool_calls: List[ToolCall] = []

            # --- Source 1: trace_events from invoke_agent response stream ---
            for te in raw_data.get("trace_events", []):
                trace_detail = te.get("trace", {})
                orch = trace_detail.get("orchestrationTrace", {})

                invocation = orch.get("invocationInput", {})
                action_group = invocation.get("actionGroupInvocationInput", {})
                if action_group:
                    tool_name = action_group.get("function", action_group.get("apiPath", "unknown"))
                    params = {}
                    for p in action_group.get("parameters", []):
                        params[p.get("name", "")] = p.get("value", "")
                    tool_calls.append(
                        ToolCall(
                            name=tool_name,
                            arguments=params,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )

                observation = orch.get("observation", {})
                action_group_obs = observation.get("actionGroupInvocationOutput", {})
                if action_group_obs and tool_calls:
                    tool_calls[-1].result = action_group_obs.get("text", "")

            # --- Source 2: CloudWatch log entries (legacy format) ---
            log_entries = raw_data.get("trace", [])
            if log_entries and not tool_calls:
                tool_call_map: Dict[str, ToolCall] = {}

                for entry in log_entries:
                    if not isinstance(entry, str):
                        continue
                    try:
                        entry_data = json.loads(entry)
                    except json.JSONDecodeError:
                        continue

                    body = entry_data.get("body")
                    if not isinstance(body, dict):
                        continue

                    if "content" in body:
                        content = body.get("content", "")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "tool_use":
                                    tool_name = item.get("name")
                                    tool_input = item.get("input", {})
                                    tool_id = item.get("id")
                                    if tool_name:
                                        tc = ToolCall(
                                            name=tool_name,
                                            arguments=tool_input if isinstance(tool_input, dict) else {},
                                            timestamp=datetime.now(timezone.utc),
                                        )
                                        if tool_id:
                                            tool_call_map[tool_id] = tc
                                        tool_calls.append(tc)

                    if "id" in body:
                        tool_id = body.get("id")
                        content = body.get("content")
                        attributes = entry_data.get("attributes", {})
                        if isinstance(attributes, dict):
                            event_name = attributes.get("event.name")
                            if event_name == "gen_ai.tool.message" and tool_id in tool_call_map:
                                tool_call_map[tool_id].result = content

            return tool_calls

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from AgentCore response."""
        try:
            metadata: Dict[str, Any] = {}

            if "latency" in raw_data:
                metadata["latency"] = raw_data["latency"]
            if "trace_id" in raw_data:
                metadata["trace_id"] = raw_data["trace_id"]
            if "agent_runtime_id" in raw_data:
                metadata["agent_runtime_id"] = raw_data["agent_runtime_id"]
            # Legacy keys for backward compatibility
            if "agent_id" in raw_data:
                metadata["agent_id"] = raw_data["agent_id"]
            if "agent_version" in raw_data:
                metadata["agent_version"] = raw_data["agent_version"]
            if "agent_arn" in raw_data:
                metadata["agent_arn"] = raw_data["agent_arn"]

            # Token usage from trace_events
            input_tokens = 0
            output_tokens = 0
            for te in raw_data.get("trace_events", []):
                trace_detail = te.get("trace", {})
                usage = trace_detail.get("orchestrationTrace", {}).get("modelInvocationOutput", {}).get("metadata", {}).get("usage", {})
                if usage:
                    input_tokens += usage.get("inputTokens", 0)
                    output_tokens += usage.get("outputTokens", 0)

            # Fallback: token usage from CloudWatch log entries
            if input_tokens == 0 and output_tokens == 0:
                for entry in raw_data.get("trace", []):
                    if not isinstance(entry, str):
                        continue
                    try:
                        entry_data = json.loads(entry)
                        if "usage" in entry_data:
                            usage = entry_data["usage"]
                            if isinstance(usage, dict):
                                input_tokens += usage.get("input_tokens", 0) or usage.get("inputTokens", 0)
                                output_tokens += usage.get("output_tokens", 0) or usage.get("outputTokens", 0)
                    except json.JSONDecodeError:
                        continue

            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens

            return metadata

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )


    @classmethod
    def generate_xray_trace_id(cls) -> str:
        """Generate a synthetic X-Ray trace ID."""
        # Generate timestamp part (8 hex digits)
        timestamp = hex(int(time.time()))[2:].zfill(8)

        # Generate random part (24 hex digits)
        random_hex = uuid.uuid4().hex[:24]

        # Generate parent ID (16 hex digits)
        parent_id = uuid.uuid4().hex[:16]

        # Construct the trace ID (always sampled)
        trace_id = f"Root=1-{timestamp}-{random_hex};Parent={parent_id};Sampled=1"

        return trace_id

    @classmethod
    def _extract_otel_trace_id(cls, log_entries: List[str]) -> Optional[str]:
        """Extract the otel traceId from a list of log entries.

        Looks for the top-level 'traceId' field in the otel log format.
        Returns the first non-empty traceId found, or None.
        """
        for entry in log_entries:
            try:
                entry_data = json.loads(entry)
                trace_id = entry_data.get("traceId")
                if trace_id:
                    return trace_id
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    @classmethod
    def get_logs_by_trace_id(cls, trace_id: str, log_group_names: List[str], region_name: str = 'us-east-1', log_stream_name: Optional[str] = None) -> List[str]:
        """
        Run a CloudWatch Logs Insights query to find all logs for a given trace ID.

        Args:
            trace_id: The X-Ray trace ID to search for.
            log_group_names: A list of CloudWatch log group names to search.
            region_name: The AWS region where the log groups are located.
            log_stream_name: Optional log stream name to scope the query to
                (e.g. "otel-rt-logs"). If None, searches all streams.

        Returns:
            List of log message strings matching the trace ID.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for log retrieval — pip install boto3")

        client = boto3.client('logs', region_name=region_name)

        # Build query with optional log stream filter
        filters = [f'filter @message like /"{trace_id}"/']
        if log_stream_name:
            filters.insert(0, f'filter @logStream = "{log_stream_name}"')

        query_string = f'fields @timestamp, @message | {" | ".join(filters)} | sort @timestamp asc'
        out: List[str] = []
        try:
            # Start the query
            response = client.start_query(
                logGroupNames=log_group_names,
                startTime=int((time.time() - 3600) * 1000),  # Search logs from the last hour
                endTime=int(time.time() * 1000),
                queryString=query_string,
                limit=10000,
            )
            query_id = response['queryId']

            # Wait for the query to complete
            status = None
            while status not in ('Complete', 'Failed', 'Cancelled'):
                time.sleep(5)
                query_status_response = client.get_query_results(queryId=query_id)
                status = query_status_response['status']

            # Retrieve the results
            if status == 'Complete':
                results = query_status_response['results']
                if not results:
                    print(f"No logs found for trace ID: {trace_id} - waiting 5 more seconds ...")
                else:
                    for log_event in results:
                        message = next((field['value'] for field in log_event if field['field'] == '@message'), None)
                        if message:
                            out.append(message)
            else:
                print(f"Query failed or was cancelled. Status: {status}")

        except Exception as e:
            print(f"An error occurred: {e}")
        return out

    @classmethod
    def get_logs_by_session_id(
        cls,
        session_id: str,
        log_group_names: List[str],
        region_name: str = 'us-east-1',
        log_stream_name: Optional[str] = None,
        invocation_time: Optional[float] = None,
    ) -> List[str]:
        """
        Run a CloudWatch Logs Insights query to find all logs for a given session ID.

        Uses the session ID (runtimeSessionId) which is passed to the agent
        during invocation and appears in the otel logs, making it a reliable
        correlation key.

        Args:
            session_id: The runtime session ID used in the invocation.
            log_group_names: A list of CloudWatch log group names to search.
            region_name: The AWS region where the log groups are located.
            log_stream_name: Optional log stream name to scope the query to
                (e.g. "otel-rt-logs"). If None, searches all streams.
            invocation_time: Unix timestamp of when the invocation started.
                If provided, searches from 1 minute before this time.
                If None, searches the last hour.

        Returns:
            List of log message strings matching the session ID.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for log retrieval — pip install boto3")

        client = boto3.client('logs', region_name=region_name)

        # Build query with optional log stream filter
        filters = [f'filter @message like /"{session_id}"/']
        if log_stream_name:
            filters.insert(0, f'filter @logStream = "{log_stream_name}"')

        query_string = f'fields @timestamp, @message | {" | ".join(filters)} | sort @timestamp asc'
        out: List[str] = []

        # Time window: from 1 min before invocation (or last hour) to now
        if invocation_time:
            start_ms = int((invocation_time - 60) * 1000)
        else:
            start_ms = int((time.time() - 3600) * 1000)
        end_ms = int(time.time() * 1000)

        try:
            response = client.start_query(
                logGroupNames=log_group_names,
                startTime=start_ms,
                endTime=end_ms,
                queryString=query_string,
                limit=10000,
            )
            query_id = response['queryId']

            # Wait for the query to complete
            status = None
            while status not in ('Complete', 'Failed', 'Cancelled'):
                time.sleep(5)
                query_status_response = client.get_query_results(queryId=query_id)
                status = query_status_response['status']

            # Retrieve the results
            if status == 'Complete':
                results = query_status_response['results']
                if not results:
                    print(f"No logs found for session ID: {session_id}")
                else:
                    for log_event in results:
                        message = next((field['value'] for field in log_event if field['field'] == '@message'), None)
                        if message:
                            out.append(message)
            else:
                print(f"Query failed or was cancelled. Status: {status}")

        except Exception as e:
            print(f"An error occurred: {e}")
        return out

    @classmethod
    def get_logs_by_time_window(
        cls,
        log_group_names: List[str],
        invocation_time: float,
        region_name: str = 'us-east-1',
        log_stream_name: Optional[str] = None,
    ) -> List[str]:
        """
        Fetch all otel logs from a narrow time window around an invocation.

        Since the otel traceId is generated server-side and can't be predicted
        client-side, this method fetches all logs from the invocation time
        window and returns them. The caller can then filter/group by traceId
        if needed.

        Args:
            log_group_names: CloudWatch log group names to search.
            invocation_time: Unix timestamp of when the invocation started.
            region_name: AWS region.
            log_stream_name: Optional log stream name filter (e.g. "otel-rt-logs").

        Returns:
            List of log message strings from the time window.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for log retrieval — pip install boto3")

        client = boto3.client('logs', region_name=region_name)

        # Build query with optional log stream filter
        filters = []
        if log_stream_name:
            filters.append(f'filter @logStream = "{log_stream_name}"')

        filter_clause = f' | {" | ".join(filters)}' if filters else ''
        query_string = f'fields @timestamp, @message{filter_clause} | sort @timestamp asc'

        # Narrow time window: from invocation start to now
        start_ms = int(invocation_time * 1000)
        end_ms = int(time.time() * 1000)

        out: List[str] = []
        try:
            response = client.start_query(
                logGroupNames=log_group_names,
                startTime=start_ms,
                endTime=end_ms,
                queryString=query_string,
                limit=10000,
            )
            query_id = response['queryId']

            status = None
            while status not in ('Complete', 'Failed', 'Cancelled'):
                time.sleep(5)
                query_status_response = client.get_query_results(queryId=query_id)
                status = query_status_response['status']

            if status == 'Complete':
                results = query_status_response['results']
                if not results:
                    print(f"No logs found in time window starting at {invocation_time}")
                else:
                    for log_event in results:
                        message = next((field['value'] for field in log_event if field['field'] == '@message'), None)
                        if message:
                            out.append(message)
            else:
                print(f"Query failed or was cancelled. Status: {status}")

        except Exception as e:
            print(f"An error occurred: {e}")
        return out

    @classmethod
    def extract_tools_info(cls, log_entries: List[str]) -> Dict[str, Any]:
        """Extract tool calls, responses, and token metrics from otel-rt-logs.

        Parses OpenTelemetry-formatted log entries (Strands/AgentCore format)
        to extract tool calls, their responses, and token usage.

        Handles multiple formats found in otel-rt-logs:
        - gen_ai.assistant.message: contains toolUse items in body.content
          and/or body.tool_calls with function format
        - gen_ai.choice: contains body.message.tool_calls
        - gen_ai.tool.message: contains tool responses with body.id and body.content

        Args:
            log_entries: List of JSON-encoded log message strings from CloudWatch.

        Returns:
            Dict with tool_calls list and token_usage metrics.
        """
        tool_calls: List[Dict[str, Any]] = []
        tool_call_map: Dict[str, Dict[str, Any]] = {}  # id -> tool_call dict
        input_tokens = 0
        output_tokens = 0

        for entry in log_entries:
            try:
                entry_data = json.loads(entry)
            except (json.JSONDecodeError, TypeError):
                continue

            body = entry_data.get("body", {})
            attributes = entry_data.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            event_name = attributes.get("event.name", "") or entry_data.get("eventName", "")

            # If body is a JSON string, try to parse it
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    continue

            if not isinstance(body, dict):
                continue

            # --- Format 1: body.content with toolUse items (Bedrock Converse format) ---
            if "content" in body:
                content = body.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        # toolUse format: {"toolUse": {"name": ..., "input": ..., "toolUseId": ...}}
                        if "toolUse" in item:
                            tu = item["toolUse"]
                            tool_name = tu.get("name")
                            tool_input = tu.get("input", {})
                            tool_id = tu.get("toolUseId")
                            if tool_name and tool_id not in tool_call_map:
                                tc = {
                                    "name": tool_name,
                                    "args": tool_input if isinstance(tool_input, dict) else {},
                                    "id": tool_id,
                                    "response": None,
                                }
                                tool_calls.append(tc)
                                if tool_id:
                                    tool_call_map[tool_id] = tc

                        # type: tool_use format (legacy)
                        if item.get("type") == "tool_use":
                            tool_name = item.get("name")
                            tool_input = item.get("input", {})
                            tool_id = item.get("id")
                            if tool_name and tool_id not in tool_call_map:
                                tc = {
                                    "name": tool_name,
                                    "args": tool_input if isinstance(tool_input, dict) else {},
                                    "id": tool_id,
                                    "response": None,
                                }
                                tool_calls.append(tc)
                                if tool_id:
                                    tool_call_map[tool_id] = tc

            # --- Format 2: body.tool_calls with function format ---
            if "tool_calls" in body:
                for tc_entry in body["tool_calls"]:
                    if isinstance(tc_entry, dict) and tc_entry.get("type") == "function":
                        func = tc_entry.get("function", {})
                        tool_name = func.get("name")
                        tool_args = func.get("arguments", {})
                        tool_id = tc_entry.get("id")
                        if tool_name and tool_id not in tool_call_map:
                            tc = {
                                "name": tool_name,
                                "args": tool_args if isinstance(tool_args, dict) else {},
                                "id": tool_id,
                                "response": None,
                            }
                            tool_calls.append(tc)
                            if tool_id:
                                tool_call_map[tool_id] = tc

            # --- Format 3: body.message.tool_calls (gen_ai.choice) ---
            if "message" in body and isinstance(body["message"], dict):
                msg = body["message"]
                if "tool_calls" in msg:
                    for tc_entry in msg["tool_calls"]:
                        if isinstance(tc_entry, dict) and tc_entry.get("type") == "function":
                            func = tc_entry.get("function", {})
                            tool_name = func.get("name")
                            tool_args = func.get("arguments", {})
                            tool_id = tc_entry.get("id")
                            if tool_name and tool_id not in tool_call_map:
                                tc = {
                                    "name": tool_name,
                                    "args": tool_args if isinstance(tool_args, dict) else {},
                                    "id": tool_id,
                                    "response": None,
                                }
                                tool_calls.append(tc)
                                if tool_id:
                                    tool_call_map[tool_id] = tc

            # --- Format 4: strands.telemetry.tracer format ---
            # Strands agents emit logs with body.output.messages[].content.message
            # containing a JSON-encoded string with toolUse items, and
            # body.output.messages[].content["tool.result"] with toolResult items.
            if "output" in body and isinstance(body.get("output"), dict):
                output = body["output"]
                output_messages = output.get("messages", [])
                for msg in output_messages:
                    if not isinstance(msg, dict):
                        continue
                    msg_content = msg.get("content", {})

                    # Handle content as a dict with "message" and "tool.result" keys
                    if isinstance(msg_content, dict):
                        message_str = msg_content.get("message", "")
                        tool_result_str = msg_content.get("tool.result", "")

                        # Parse tool calls from the "message" field
                        if message_str and isinstance(message_str, str):
                            try:
                                message_items = json.loads(message_str)
                                if isinstance(message_items, list):
                                    for item in message_items:
                                        if not isinstance(item, dict):
                                            continue
                                        if "toolUse" in item:
                                            tu = item["toolUse"]
                                            tool_name = tu.get("name")
                                            tool_input = tu.get("input", {})
                                            tool_id = tu.get("toolUseId")
                                            if tool_name and tool_id not in tool_call_map:
                                                tc = {
                                                    "name": tool_name,
                                                    "args": tool_input if isinstance(tool_input, dict) else {},
                                                    "id": tool_id,
                                                    "response": None,
                                                }
                                                tool_calls.append(tc)
                                                if tool_id:
                                                    tool_call_map[tool_id] = tc
                            except (json.JSONDecodeError, TypeError):
                                pass

                        # Parse tool results from the "tool.result" field
                        if tool_result_str and isinstance(tool_result_str, str):
                            try:
                                result_items = json.loads(tool_result_str)
                                if isinstance(result_items, list):
                                    for item in result_items:
                                        if not isinstance(item, dict):
                                            continue
                                        if "toolResult" in item:
                                            tr = item["toolResult"]
                                            tool_id = tr.get("toolUseId")
                                            if tool_id and tool_id in tool_call_map:
                                                # Extract text from content list
                                                tr_content = tr.get("content", [])
                                                texts = []
                                                for c in tr_content:
                                                    if isinstance(c, dict) and "text" in c:
                                                        texts.append(c["text"])
                                                tool_call_map[tool_id]["response"] = texts if len(texts) != 1 else texts[0]
                            except (json.JSONDecodeError, TypeError):
                                pass

                    # Handle content as a JSON string (tool results in a separate message)
                    elif isinstance(msg_content, str):
                        try:
                            content_items = json.loads(msg_content)
                            if isinstance(content_items, list):
                                for item in content_items:
                                    if not isinstance(item, dict):
                                        continue
                                    if "toolResult" in item:
                                        tr = item["toolResult"]
                                        tool_id = tr.get("toolUseId")
                                        if tool_id and tool_id in tool_call_map:
                                            tr_content = tr.get("content", [])
                                            texts = []
                                            for c in tr_content:
                                                if isinstance(c, dict) and "text" in c:
                                                    texts.append(c["text"])
                                            tool_call_map[tool_id]["response"] = texts if len(texts) != 1 else texts[0]
                        except (json.JSONDecodeError, TypeError):
                            pass

            # --- Extract tool responses (gen_ai.tool.message) ---
            if event_name == "gen_ai.tool.message" and "id" in body:
                tool_id = body.get("id")
                content = body.get("content")
                if tool_id and tool_id in tool_call_map:
                    tool_call_map[tool_id]["response"] = content

            # --- Extract token usage ---
            if "usage" in body and isinstance(body.get("usage"), dict):
                usage = body["usage"]
                input_tokens += usage.get("input_tokens", 0) or usage.get("inputTokens", 0)
                output_tokens += usage.get("output_tokens", 0) or usage.get("outputTokens", 0)

            if "gen_ai.usage.input_tokens" in attributes:
                input_tokens += int(attributes["gen_ai.usage.input_tokens"])
            if "gen_ai.usage.output_tokens" in attributes:
                output_tokens += int(attributes["gen_ai.usage.output_tokens"])

        return {
            "tool_calls": tool_calls,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
