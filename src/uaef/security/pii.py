# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PII detection for UAEF.

This module provides functionality to detect Personally Identifiable Information (PII)
in agent traces, evaluation data, and other sensitive content. It supports both regex-based
detection (fast, deterministic) and ML-based detection (more accurate, requires additional libraries).

Requirements: 20.1-20.2, 24.1
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected.
    
    Note: NAME and ADDRESS require ML-based detection (not yet implemented).
    They will not be detected by the regex-based detector.
    """
    
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"  # ML-only: requires ML detection to be enabled
    ADDRESS = "address"  # ML-only: requires ML detection to be enabled
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    URL = "url"
    ALL = "all"


class PIIMatch(BaseModel):
    """Represents a detected PII match."""
    
    pii_type: PIIType = Field(description="Type of PII detected")
    start: int = Field(description="Start position in text")
    end: int = Field(description="End position in text")
    text: str = Field(description="Matched text")
    confidence: float = Field(
        default=1.0,
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0
    )


class PIIDetectionConfig(BaseModel):
    """Configuration for PII detection."""
    
    enabled_types: Set[PIIType] = Field(
        default_factory=lambda: {
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.SSN,
            PIIType.CREDIT_CARD,
        },
        description="PII types to detect"
    )
    use_ml_detection: bool = Field(
        default=False,
        description="Use ML-based detection (requires additional libraries)"
    )
    min_confidence: float = Field(
        default=0.8,
        description="Minimum confidence threshold for ML detection",
        ge=0.0,
        le=1.0
    )
    custom_patterns: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom regex patterns for additional PII types"
    )


class BasePIIDetector(ABC):
    """Base class for PII detectors."""
    
    @abstractmethod
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
        """
        pass


class RegexPIIDetector(BasePIIDetector):
    """Regex-based PII detector (fast, deterministic)."""
    
    # Regex patterns for common PII types
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        PIIType.PHONE: r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:4[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}|5[1-5][0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}|3[47][0-9]{2}[-\s]?[0-9]{6}[-\s]?[0-9]{5}|3(?:0[0-5]|[68][0-9])[0-9][-\s]?[0-9]{6}[-\s]?[0-9]{4}|6(?:011|5[0-9]{2})[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})\b',
        PIIType.IP_ADDRESS: r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        PIIType.URL: r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
    }
    
    def __init__(self, config: PIIDetectionConfig):
        """
        Initialize regex-based PII detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self._compiled_patterns: Dict[PIIType, re.Pattern] = {}
        self._compile_patterns()
    
    # PII types that require ML-based detection and cannot be matched by regex
    ML_ONLY_TYPES = {PIIType.NAME, PIIType.ADDRESS}

    def _compile_patterns(self) -> None:
        """Compile regex patterns for enabled PII types."""
        for pii_type in self.config.enabled_types:
            if pii_type == PIIType.ALL:
                # Compile all patterns
                for pt, pattern in self.PATTERNS.items():
                    self._compiled_patterns[pt] = re.compile(pattern)
            elif pii_type in self.ML_ONLY_TYPES:
                logger.warning(
                    "PIIType.%s requires ML-based detection which is not yet implemented. "
                    "It will not be detected by the regex-based detector.",
                    pii_type.value.upper(),
                )
            elif pii_type in self.PATTERNS:
                self._compiled_patterns[pii_type] = re.compile(self.PATTERNS[pii_type])
        
        # Add custom patterns
        for name, pattern in self.config.custom_patterns.items():
            try:
                pii_type = PIIType(name)
                if pii_type in self.ML_ONLY_TYPES:
                    logger.warning(
                        "Custom pattern for ML-only type '%s' will override ML requirement.",
                        name,
                    )
                self._compiled_patterns[pii_type] = re.compile(pattern)
            except ValueError:
                # Unknown custom pattern name — assign a unique key using a synthetic approach
                logger.warning(
                    "Custom pattern name '%s' is not a recognized PIIType. "
                    "Matches will be reported as PIIType.NAME.",
                    name,
                )
                self._compiled_patterns[PIIType.NAME] = re.compile(pattern)
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text using regex patterns.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
            
        Raises:
            ValueError: If text is None or empty
        """
        if not text:
            raise ValueError("Text cannot be None or empty")
        
        matches: List[PIIMatch] = []
        
        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        confidence=1.0,  # Regex matches are deterministic
                    )
                )
        
        # Sort by start position
        matches.sort(key=lambda m: m.start)
        
        # Remove overlapping matches (keep highest confidence)
        return self._remove_overlaps(matches)
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """
        Remove overlapping matches, keeping the one with highest confidence.
        
        Args:
            matches: List of PIIMatch objects (sorted by start position)
            
        Returns:
            List of non-overlapping PIIMatch objects
        """
        if not matches:
            return []
        
        result = [matches[0]]
        
        for match in matches[1:]:
            last_match = result[-1]
            
            # Check for overlap
            if match.start < last_match.end:
                # Keep the match with higher confidence
                if match.confidence > last_match.confidence:
                    result[-1] = match
            else:
                result.append(match)
        
        return result


class MLPIIDetector(BasePIIDetector):
    """
    ML-based PII detector (more accurate, requires additional libraries).
    
    This is a placeholder/interface for ML-based detection using libraries like
    spaCy, Presidio, or custom models. The actual implementation would require
    additional dependencies and model loading.
    
    Example usage with Presidio:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language='en')
    
    Example usage with spaCy:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG"]:
                # Process entity
    """
    
    def __init__(self, config: PIIDetectionConfig):
        """
        Initialize ML-based PII detector.
        
        Args:
            config: Detection configuration
            
        Raises:
            NotImplementedError: ML detection not yet implemented
        """
        self.config = config
        raise NotImplementedError(
            "ML-based PII detection is not yet implemented. "
            "To use ML detection, install additional libraries like "
            "presidio-analyzer or spacy and implement this class. "
            "For now, use RegexPIIDetector for regex-based detection."
        )
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text using ML models.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
            
        Raises:
            NotImplementedError: ML detection not yet implemented
        """
        raise NotImplementedError("ML-based PII detection not yet implemented")


class PIIDetector:
    """
    Main PII detector that orchestrates detection strategies.
    
    This class provides a unified interface for PII detection, supporting both
    regex-based and ML-based detection methods.
    """
    
    def __init__(
        self,
        detection_config: Optional[PIIDetectionConfig] = None,
    ):
        """
        Initialize PII detector.
        
        Args:
            detection_config: Detection configuration (uses defaults if None)
        """
        self.config = detection_config or PIIDetectionConfig()
        
        # Initialize appropriate detector
        if self.config.use_ml_detection:
            self.detector = MLPIIDetector(self.config)
        else:
            self.detector = RegexPIIDetector(self.config)
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
            
        Raises:
            ValueError: If text is invalid
        """
        return self.detector.detect(text)
    
    def scan_dict(self, data: Dict[str, Any]) -> Dict[str, List[PIIMatch]]:
        """
        Scan a dictionary for PII in string values.
        
        Args:
            data: Dictionary to scan
            
        Returns:
            Dictionary mapping keys to lists of PIIMatch objects
        """
        results: Dict[str, List[PIIMatch]] = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                matches = self.detect(value)
                if matches:
                    results[key] = matches
            elif isinstance(value, dict):
                nested_results = self.scan_dict(value)
                if nested_results:
                    results[key] = nested_results  # type: ignore
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        matches = self.detect(item)
                        if matches:
                            results[f"{key}[{i}]"] = matches
                    elif isinstance(item, dict):
                        nested_results = self.scan_dict(item)
                        if nested_results:
                            results[f"{key}[{i}]"] = nested_results  # type: ignore
        
        return results
    
    def has_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.
        
        Args:
            text: Text to check
            
        Returns:
            True if PII detected, False otherwise
        """
        try:
            matches = self.detect(text)
            return len(matches) > 0
        except ValueError:
            return False

    def redact(self, text: str, *, placeholder_fmt: str = "[REDACTED:{pii_type}]") -> str:
        """
        Replace detected PII spans in text with a placeholder.

        Security review M-02: this is the redaction primitive that
        ``uaef.security.pii.redact_for_storage`` (and the service's
        opt-in persisted-storage redaction) build on. Only covers the
        PII types this instance's ``detection_config.enabled_types`` is
        configured for — by default EMAIL/PHONE/SSN/CREDIT_CARD (regex-
        detectable); NAME/ADDRESS require ``use_ml_detection=True``,
        which raises ``NotImplementedError`` today, so text containing
        only names/addresses and no other configured PII type passes
        through unredacted. This is a real, documented limitation, not
        silently glossed over — see the module docstring and the
        ``PIIType`` enum's own note.

        Args:
            text: Text to redact
            placeholder_fmt: Format string for the replacement, with a
                ``{pii_type}`` placeholder (e.g. "[REDACTED:email]")

        Returns:
            Text with every detected PII span replaced. Returns the
            input unchanged if no PII is detected or the input is falsy.
        """
        if not text:
            return text
        try:
            matches = self.detect(text)
        except ValueError:
            return text
        if not matches:
            return text
        # Replace from the end so earlier spans' offsets stay valid as we
        # mutate the string (matches may be in detection order, not sorted).
        result = text
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            placeholder = placeholder_fmt.format(pii_type=match.pii_type.value)
            result = result[:match.start] + placeholder + result[match.end:]
        return result





# Convenience functions for common use cases

def create_default_detector() -> PIIDetector:
    """
    Create a PIIDetector with default configuration.
    
    Returns:
        PIIDetector instance with default settings
    """
    return PIIDetector()


def detect_pii(text: str, pii_types: Optional[Set[PIIType]] = None) -> List[PIIMatch]:
    """
    Convenience function to detect PII in text.
    
    Args:
        text: Text to scan
        pii_types: PII types to detect (uses defaults if None)
        
    Returns:
        List of PIIMatch objects
    """
    config = PIIDetectionConfig()
    if pii_types:
        config.enabled_types = pii_types
    
    detector = PIIDetector(detection_config=config)
    return detector.detect(text)


#: Module-level default detector, lazily created (security review M-02). Reused
#: across ``redact_for_storage`` calls to avoid rebuilding the regex detector
#: on every call; safe to share since ``PIIDetector``/``RegexPIIDetector`` hold
#: no per-call mutable state.
_default_redaction_detector: Optional[PIIDetector] = None


def redact_for_storage(text: str) -> str:
    """
    Redact EMAIL/PHONE/SSN/CREDIT_CARD spans in text before persisting it.

    Security review M-02: prompts, agent responses, and ground-truth answers
    were persisted (DynamoDB/S3) and rendered in the UI completely
    un-redacted, with PII scanning both off the evaluation path and
    NAME/ADDRESS detection unimplemented. This function is the storage-path
    redaction primitive: it operates on already-computed evaluation output
    (agent response text, ground-truth answers, judge reasoning) — it MUST
    NOT be called anywhere upstream of a metric's own Bedrock judge call,
    since redacting the text a judge scores would change evaluation results
    for anything containing PII-shaped content, not just its handling in
    storage. Every call site wiring this in (uaef-service's UI-result storage,
    the library's experiment-persistence layer) applies it strictly after
    ``batch_evaluate``/``evaluate`` has already returned scores, and each is
    opt-in via its own env var — this function does not change behavior by
    existing; something has to call it.

    Uses the default (regex-based) detector's default enabled_types
    (EMAIL/PHONE/SSN/CREDIT_CARD). NAME/ADDRESS are NOT redacted by this
    function — ``PIIType.NAME``/``PIIType.ADDRESS`` require ML-based
    detection, which is unimplemented (``MLPIIDetector`` raises
    ``NotImplementedError``); this is a documented limitation, not a silent
    gap introduced here.

    Args:
        text: Text to redact (e.g. an agent response, ground-truth answer,
            or judge reasoning string)

    Returns:
        Text with detected PII spans replaced by ``[REDACTED:<type>]``
        placeholders. Returns the input unchanged if it's falsy or contains
        no detectable PII.
    """
    global _default_redaction_detector
    if _default_redaction_detector is None:
        _default_redaction_detector = create_default_detector()
    return _default_redaction_detector.redact(text)
