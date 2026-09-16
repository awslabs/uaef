# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt template library for UAEF LLM Judge.

This module provides versioned prompt templates for LLM-based metrics
with validation and custom template support.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from uaef.logging import get_logger
from uaef.metrics.utils import (
    PROVENANCE_FIELD,
    loads_judge_json,
    validate_judge_response,
    validate_provenance,
)

logger = get_logger(__name__)


class PromptTemplateError(Exception):
    """Base exception for prompt template errors."""
    pass


class PromptTemplateNotFoundError(PromptTemplateError):
    """Exception raised when prompt template is not found."""
    pass


class PromptTemplateValidationError(PromptTemplateError):
    """Exception raised when prompt template validation fails."""
    pass


class PromptTemplateLoader:
    """
    Loader for versioned prompt templates.
    
    Supports:
    - Loading templates from versioned directories
    - Custom template overrides
    - Template validation
    - Variable substitution
    """
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        default_version: str = "v1"
    ):
        """
        Initialize prompt template loader.
        
        Args:
            templates_dir: Directory containing prompt templates (defaults to package prompts/)
            default_version: Default template version to use
        """
        if templates_dir is None:
            # Use package prompts directory
            templates_dir = Path(__file__).parent / "prompts"
        
        self.templates_dir = Path(templates_dir)
        self.default_version = default_version
        self._template_cache: Dict[str, str] = {}
        
        logger.info(
            f"Initialized PromptTemplateLoader with templates_dir={self.templates_dir}, "
            f"default_version={self.default_version}"
        )
    
    def load_template(
        self,
        metric_name: str,
        version: Optional[str] = None,
        custom_template: Optional[str] = None
    ) -> str:
        """
        Load a prompt template.
        
        Args:
            metric_name: Name of the metric (e.g., "answer_relevance")
            version: Template version (defaults to default_version)
            custom_template: Custom template string to use instead of file
            
        Returns:
            Template string
            
        Raises:
            PromptTemplateNotFoundError: If template file not found
            PromptTemplateError: For other errors
        """
        # Use custom template if provided
        if custom_template:
            logger.debug(f"Using custom template for {metric_name}")
            return custom_template
        
        # Use default version if not specified
        version = version or self.default_version
        
        # Check cache
        cache_key = f"{metric_name}:{version}"
        if cache_key in self._template_cache:
            logger.debug(f"Using cached template for {cache_key}")
            return self._template_cache[cache_key]
        
        # Load from file
        template_path = self.templates_dir / version / f"{metric_name}.txt"
        
        if not template_path.exists():
            raise PromptTemplateNotFoundError(
                f"Template not found: {template_path}. "
                f"Available templates: {self.list_templates(version)}"
            )
        
        try:
            with open(template_path, "r") as f:
                template = f.read()
            
            # Cache template
            self._template_cache[cache_key] = template
            
            logger.debug(f"Loaded template from {template_path}")
            return template
            
        except Exception as e:
            raise PromptTemplateError(
                f"Error loading template from {template_path}: {str(e)}"
            )
    
    def format_template(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Format a template with variables.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Formatted template string
            
        Raises:
            PromptTemplateValidationError: If required variables are missing
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise PromptTemplateValidationError(
                f"Missing required variable in template: {e}"
            )
        except Exception as e:
            raise PromptTemplateError(
                f"Error formatting template: {str(e)}"
            )
    
    def validate_response(
        self,
        response_text: str,
        expected_format: str = "json",
        candidate_text: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Validate LLM response against expected format.
        
        Args:
            response_text: Response text from LLM
            expected_format: Expected format ("json" or "text")
            candidate_text: Security review H-01: the untrusted candidate
                content this response scored (exactly what was passed to
                ``wrap_untrusted``). When provided, the response must carry a
                verbatim evidence quote from that content, and the quote is
                verified to occur in it — binding the score to the real
                candidate rather than to whatever the candidate may have
                instructed the judge to say. Only pass this when the rubric
                asked for the quote (see
                ``uaef.metrics.utils.build_provenance_instruction``).
            
        Returns:
            Parsed response (dict for JSON, str for text)
            
        Raises:
            PromptTemplateValidationError: If response doesn't match expected format
        """
        if expected_format == "json":
            # Security review H-01/M-05: parse strictly (loads_judge_json —
            # control characters are normalized rather than tolerated by a
            # permissive parser), then validate the parsed result's
            # shape/types with the same shared validator every built-in metric
            # module uses, and finally verify provenance when the caller
            # requested it. Shape validation alone cannot detect a judge
            # manipulated into emitting a well-formed but steered score; the
            # prompt-level boundary (wrap_untrusted /
            # build_judge_system_prompt) plus provenance binding is what
            # covers that.
            try:
                response_json = loads_judge_json(response_text)
            except json.JSONDecodeError as e:
                raise PromptTemplateValidationError(
                    f"Response is not valid JSON: {str(e)}"
                )

            try:
                if candidate_text is None:
                    return validate_judge_response(response_json)

                validate_judge_response(
                    response_json,
                    allow_keys={"score", "reasoning", PROVENANCE_FIELD}
                )
                return validate_provenance(response_json, candidate_text)
            except ValueError as e:
                raise PromptTemplateValidationError(str(e))
        
        elif expected_format == "text":
            return {"text": response_text}
        
        else:
            raise PromptTemplateValidationError(
                f"Unsupported expected format: {expected_format}"
            )
    
    def list_templates(self, version: Optional[str] = None) -> list[str]:
        """
        List available templates for a version.
        
        Args:
            version: Template version (defaults to default_version)
            
        Returns:
            List of template names (without .txt extension)
        """
        version = version or self.default_version
        version_dir = self.templates_dir / version
        
        if not version_dir.exists():
            return []
        
        templates = []
        for template_file in version_dir.glob("*.txt"):
            templates.append(template_file.stem)
        
        return sorted(templates)
    
    def list_versions(self) -> list[str]:
        """
        List available template versions.
        
        Returns:
            List of version names
        """
        if not self.templates_dir.exists():
            return []
        
        versions = []
        for version_dir in self.templates_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith("v"):
                versions.append(version_dir.name)
        
        return sorted(versions)
    
    def get_template_info(
        self,
        metric_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            metric_name: Name of the metric
            version: Template version (defaults to default_version)
            
        Returns:
            Dictionary with template info (path, size, variables)
        """
        version = version or self.default_version
        template_path = self.templates_dir / version / f"{metric_name}.txt"
        
        if not template_path.exists():
            raise PromptTemplateNotFoundError(
                f"Template not found: {template_path}"
            )
        
        # Load template to extract variables
        template = self.load_template(metric_name, version)
        
        # Extract variable names from template
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        return {
            "metric_name": metric_name,
            "version": version,
            "path": str(template_path),
            "size_bytes": template_path.stat().st_size,
            "variables": list(set(variables)),
            "line_count": len(template.split('\n'))
        }


# Global template loader instance
_template_loader: Optional[PromptTemplateLoader] = None


def get_template_loader() -> PromptTemplateLoader:
    """
    Get the global prompt template loader.
    
    Returns:
        PromptTemplateLoader instance
    """
    global _template_loader
    if _template_loader is None:
        _template_loader = PromptTemplateLoader()
    return _template_loader


def set_template_loader(loader: PromptTemplateLoader) -> None:
    """
    Set the global prompt template loader.
    
    Args:
        loader: PromptTemplateLoader instance to set as global
    """
    global _template_loader
    _template_loader = loader


def reset_template_loader() -> None:
    """Reset the global template loader to None."""
    global _template_loader
    _template_loader = None
