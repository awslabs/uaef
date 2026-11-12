# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the Tracekit bundle that tests/test_tracekit_adapter.py reads.

A scripted support agent runs against a fresh Tracekit signer, started from its own
signer.yaml in a temp dir as a deployment runs one. The agent is token:support-agent; the
person who answers held calls is this OS user, named alice. Six calls: an allow, a flag, a
refund the person approves, one the person rejects, a deny, and a lookup whose tool fails.
The run is exported as a v2 bundle, with the trust config that pins the signer's log key.
POSIX only (the person is a uid). Needs the 'tracekit' extra.

    python tests/data/tracekit/make_bundle.py tests/data/tracekit
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from tracekit.sdk.client import Client, result_field

POLICY = """\
version: 'uaef-fixture-1'
description: a support agent's tools, for UAEF's Tracekit adapter tests
unknown_tools: ask
tools:
  lookup_order: app
  send_email: app
  issue_refund: app
  delete_account: app
deny:
  - id: SHOP-D001
    tool: delete_account
    pattern: '^'
    reason: deleting an account is a person's job
ask:
  - id: SHOP-A001
    tool: issue_refund
    pattern: '^'
    reason: a refund moves money
flag:
  - id: SHOP-F001
    tool: send_email
    pattern: '^'
    reason: an email leaves the company
"""
AGENT = [
    "register_run",
    "decide",
    "complete",
    "observe",
    "approval_request",
    "approval_wait",
    "approval_consume",
    "approval_abandon",
    "close_run",
    "status",
]
PERSON = ["approval_list", "approval_get", "approval_decide", "checkpoint_nudge", "status"]
# (call id, tool, arguments, what the tool returns, the person's answer to a held call)
CALLS = [
    ("call-1", "lookup_order", {"order_id": "A-1001"}, {"status": "delivered"}, None),
    (
        "call-2",
        "send_email",
        {"to": "customer@example.com", "subject": "Your order A-1001"},
        {"sent": True},
        None,
    ),
    (
        "call-3",
        "issue_refund",
        {"order_id": "A-1001", "amount": 40},
        {"refund_id": "R-1"},
        "approve",
    ),
    (
        "call-4",
        "issue_refund",
        {"order_id": "B-2002", "amount": 900},
        {"refund_id": "R-2"},
        "reject",
    ),
    ("call-5", "delete_account", {"customer_id": "C-7"}, {"deleted": True}, None),
    ("call-6", "lookup_order", {"order_id": "Z-9999"}, None, None),  # the tool fails: no such order
]


def tracekit(*argv):
    p = subprocess.run(
        [sys.executable, "-I", "-m", "tracekit", *argv], capture_output=True, text=True
    )
    if p.returncode:
        sys.exit(f"tracekit {' '.join(argv)} failed:\n{p.stdout}{p.stderr}")
    return p.stdout


def main(out):
    os.makedirs(out, exist_ok=True)
    d, proc = tempfile.mkdtemp(prefix="tk-"), None
    try:
        me = f"uid:{os.getuid()}"
        with open(os.path.join(d, "policy.yaml"), "w") as f:
            f.write(POLICY)
        cfg = os.path.join(d, "signer.yaml")
        with open(cfg, "w") as f:  # JSON is YAML
            json.dump(
                {
                    "data_dir": "data",
                    "socket": "s.sock",
                    "policy": "policy.yaml",
                    "grace_s": 1,
                    "http": {
                        "listen": "127.0.0.1:0",
                        "insecure_loopback": True,
                        "authenticators": ["token"],
                        "tokens": "tokens.json",
                    },
                    "approvals": {"approvers": [me], "persons": {me: "alice"}},
                    "authorize": {me: PERSON, "token:support-agent": AGENT},
                },
                f,
            )
        token = os.path.join(d, "agent.token")
        with open(token, "w") as f:
            f.write(tracekit("signer", "token", "add", "support-agent", "--config", cfg))
        proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "tracekit", "signer", "serve", "--config", cfg],
            stdout=subprocess.PIPE,
            text=True,
        )
        line = proc.stdout.readline()
        if " serving on " not in line:
            sys.exit(f"the signer did not start: {line!r}")
        os.environ["TRACEKIT_SIGNER"] = os.path.join(d, "s.sock")  # the person's commands
        os.environ["TRACEKIT_SIGNER_TOKEN_FILE"] = token  # the agent's client
        with Client("http://" + line.rsplit(" on ", 1)[1].strip()).run(
            agent="support-agent"
        ) as run:
            for cid, tool, args, returns, answer in CALLS:
                dec, aid = run.decide(cid, tool, args), None
                print(cid, tool, dec["decision"], dec["rule_ids"])
                if dec["decision"] == "ask":
                    aid = run.call("approval_request", tool_call_id=cid)["approval_id"]
                    tracekit("approvals", answer, aid)
                    if (
                        run.call("approval_wait", approval_id=aid, timeout_ms=60000)["state"]
                        != "approved"
                    ):
                        continue
                if (
                    dec["decision"] == "deny"
                    or not run.approval_consume(cid, tool, args, approval_id_hint=aid)["ok"]
                ):
                    continue
                if returns is None:
                    run.complete(cid, status="error")
                else:
                    run.complete(cid, result=result_field(dec.get("send_result"), returns))
        bundle, trust = os.path.join(out, "run.tkb"), os.path.join(out, "trust.json")
        tracekit("signer", "trust", "--config", cfg, "-o", trust)
        deadline = time.monotonic() + 30
        while True:  # the signer finalises the closed run and checkpoints it a moment later
            if os.path.exists(bundle):
                os.remove(bundle)
            tracekit("export", "--v2", "--run", run.run_id, "--config", cfg, "-o", bundle)
            p = subprocess.run(
                [sys.executable, "-I", "-m", "tracekit", "verify", bundle, "--trust", trust],
                capture_output=True,
                text=True,
            )
            if p.returncode == 0 and "VERIFIED:" in p.stdout or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        print(p.stdout, p.stderr)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
        shutil.rmtree(d, True)


if __name__ == "__main__":
    main(sys.argv[1])
