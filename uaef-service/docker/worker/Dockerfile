# Worker Lambda — container-image fallback (Requirement 6.3).
#
# This Dockerfile is the documented escape hatch used WHEN the unzipped
# `uaef[server]` install exceeds the 250 MB zip-package Lambda limit. It is NOT
# used by default: the WorkerStack ships the Worker as a zip-asset Lambda and
# only switches to this container image when the operator opts in via
# `cdk deploy -c worker_container_image=true` (see infra/worker_stack.py).
#
# The image stays well under the 10 GB Lambda image-size limit and keeps the
# single-source-of-truth contract: it `pip install`s the SAME pinned
# `uaef[server]==X.Y.Z` spec (passed as a build arg from infra/constants.py)
# and never vendors UAEF source.

ARG UAEF_DEPENDENCY="uaef[server]==0.2.0"

# AWS-provided Lambda Python base image (includes the Runtime Interface Client).
# checkov:skip=CKV_DOCKER_2:HEALTHCHECK is inert on a Lambda container image — the
# Lambda service (not Docker/an orchestrator) manages container lifecycle and health
# via the Runtime Interface Client, so a Dockerfile HEALTHCHECK is never executed.
# (The EKS-path deploy_eks/Dockerfile, which IS orchestrated, keeps a real HEALTHCHECK.)
#
# Pinned by digest (security review M-07): a floating `:3.12` tag can silently
# point at a different image on a later build, bypassing any review of what's
# actually shipped. This is the manifest-list digest for the `3.12` tag as of
# this change; `docker build --platform linux/amd64` (set via
# ecr_assets.Platform.LINUX_AMD64 in infra/worker_stack.py) resolves the
# correct architecture-specific manifest from it. Re-pin deliberately when
# picking up a new base image (`docker manifest inspect
# public.ecr.aws/lambda/python:3.12 --verbose` or the public.ecr.aws registry
# API) rather than reverting to a floating tag.
FROM public.ecr.aws/lambda/python@sha256:69da445c4d7a08522495cef038a84391d2cb94c76fa651f23f277aeeafde78b0

# Re-declare after FROM so the build arg is visible in this stage.
ARG UAEF_DEPENDENCY

# Local wheels for the registry-free deploy path (Path B, independent of the
# CodeArtifact/PyPI publish). The build context's `wheels/` dir is copied in;
# `--find-links /wheels` prefers a locally-built uaef wheel placed there and
# otherwise falls back to the configured index. uaef's own (public) dependencies
# resolve from the index either way. No UAEF *source* is vendored — only the
# built wheel artifact, if present.
COPY wheels/ /wheels/
# Install in several layers instead of one giant one. A single ~1GB layer is
# what fails to push over Colima ("broken pipe" / "closed network connection");
# smaller per-layer blobs upload reliably and resume across retries. The heavy
# native wheels are pre-installed first (names only — pip resolves compatible
# versions), then the pinned uaef[server] pulls in the remainder.
RUN pip install --no-cache-dir numpy scipy
RUN pip install --no-cache-dir pandas pyarrow
RUN pip install --no-cache-dir scikit-learn pillow
# Pre-install the heavy framework sub-trees in their own layers so the final
# uaef[server] layer stays small. A single ~1GB layer is what fails to push
# over Colima's user-mode network ("broken pipe" mid-blob); splitting it into
# several few-hundred-MB blobs lets each upload complete (and the ones that
# succeed are cached as "Layer already exists" across retries).
RUN pip install --no-cache-dir langchain langchain-core langchain-community langchain-aws langchain-openai
RUN pip install --no-cache-dir langgraph langgraph-checkpoint langgraph-prebuilt langgraph-sdk
RUN pip install --no-cache-dir datasets huggingface-hub
RUN pip install --no-cache-dir deepeval openai aiobotocore
RUN pip install --no-cache-dir ragas
# uaef itself + whatever small bits remain unsatisfied (most deps already
# installed above, so this final layer is small).
RUN pip install --no-cache-dir --find-links /wheels "${UAEF_DEPENDENCY}"

# Copy ONLY the service handler code (no infra/CDK code) into the task root.
# `job_state.py` lives at the app root and is imported by handlers/worker.py.
COPY handlers ${LAMBDA_TASK_ROOT}/handlers
COPY job_state.py schemas.py validation.py ${LAMBDA_TASK_ROOT}/

# Run as a non-root user. A Lambda *container image* runs as whatever USER the
# Dockerfile sets — with none set it defaults to root (flagged by kics "Missing
# User Instruction" / checkov CKV_DOCKER_3). The Runtime Interface Client runs
# fine as non-root: it only needs read access to the world-readable task root
# and write access to the always-writable /tmp.
# The AWS Lambda base image ships without shadow-utils, so install it first,
# then create the `app` user/group. Clean the package cache to keep the layer small.
RUN microdnf install -y shadow-utils \
    && /usr/sbin/groupadd -r app && /usr/sbin/useradd -r -g app app \
    && microdnf clean all
USER app

# Lambda entry point: handlers/worker.py -> handler().
CMD ["handlers.worker.handler"]
