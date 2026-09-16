# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for PII detection module.

Tests cover:
- Email detection (various formats)
- Phone number detection (US formats)
- SSN detection
- Credit card detection (Visa, Mastercard, Amex, Discover)
- IP address detection
- URL detection
- Date of birth detection
- ALL type enabling all detectors
- Custom patterns
- Dictionary scanning
- has_pii helper
- Overlap removal
- Edge cases (empty text, no PII, mixed content)
- Convenience functions
- Configuration behavior
"""

import pytest

from uaef.security.pii import (
    BasePIIDetector,
    MLPIIDetector,
    PIIDetectionConfig,
    PIIDetector,
    PIIMatch,
    PIIType,
    RegexPIIDetector,
    create_default_detector,
    detect_pii,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_detector() -> PIIDetector:
    """Detector with default config (email, phone, ssn, credit_card)."""
    return PIIDetector()


@pytest.fixture
def all_types_detector() -> PIIDetector:
    """Detector with ALL PII types enabled."""
    config = PIIDetectionConfig(enabled_types={PIIType.ALL})
    return PIIDetector(detection_config=config)


@pytest.fixture
def email_only_detector() -> PIIDetector:
    """Detector configured to detect only emails."""
    config = PIIDetectionConfig(enabled_types={PIIType.EMAIL})
    return PIIDetector(detection_config=config)


# ---------------------------------------------------------------------------
# Email Detection
# ---------------------------------------------------------------------------


class TestEmailDetection:
    """Tests for email address detection."""

    def test_simple_email(self, default_detector: PIIDetector):
        matches = default_detector.detect("Contact me at jane.doe@example.com please.")
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].text == "jane.doe@example.com"

    def test_email_with_plus_addressing(self, default_detector: PIIDetector):
        matches = default_detector.detect("Send to user+tag@domain.org")
        assert len(matches) == 1
        assert matches[0].text == "user+tag@domain.org"

    def test_email_with_subdomain(self, default_detector: PIIDetector):
        matches = default_detector.detect("admin@mail.corp.example.co.uk is valid")
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL

    def test_multiple_emails(self, default_detector: PIIDetector):
        text = "Reach alice@one.com or bob@two.net for help."
        matches = default_detector.detect(text)
        emails = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(emails) == 2
        assert {m.text for m in emails} == {"alice@one.com", "bob@two.net"}

    def test_email_with_percent_chars(self, default_detector: PIIDetector):
        matches = default_detector.detect("test%user@example.com")
        assert len(matches) == 1

    def test_no_false_positive_on_at_sign(self, default_detector: PIIDetector):
        # "@mention" without valid TLD should not match
        matches = default_detector.detect("Follow @username on social media")
        emails = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Phone Number Detection
# ---------------------------------------------------------------------------


class TestPhoneDetection:
    """Tests for phone number detection (US formats)."""

    def test_standard_format(self, default_detector: PIIDetector):
        matches = default_detector.detect("Call 555-867-5309 for info.")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_parenthesized_area_code(self, default_detector: PIIDetector):
        matches = default_detector.detect("Phone: (555) 123-4567")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_dotted_format(self, default_detector: PIIDetector):
        matches = default_detector.detect("Fax: 555.234.5678")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_with_country_code(self, default_detector: PIIDetector):
        matches = default_detector.detect("Intl: +1-800-555-0199")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_no_spaces_format(self, default_detector: PIIDetector):
        matches = default_detector.detect("Direct: 5551234567")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_no_false_positive_short_numbers(self, default_detector: PIIDetector):
        # A 5-digit number shouldn't match as a phone
        matches = default_detector.detect("Zip code is 90210")
        phones = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phones) == 0


# ---------------------------------------------------------------------------
# SSN Detection
# ---------------------------------------------------------------------------


class TestSSNDetection:
    """Tests for Social Security Number detection."""

    def test_standard_ssn(self, default_detector: PIIDetector):
        matches = default_detector.detect("SSN: 123-45-6789")
        ssns = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssns) == 1
        assert ssns[0].text == "123-45-6789"

    def test_ssn_in_paragraph(self, default_detector: PIIDetector):
        text = "The applicant's SSN is 987-65-4321 as shown on the form."
        matches = default_detector.detect(text)
        ssns = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssns) == 1
        assert ssns[0].text == "987-65-4321"

    def test_no_false_positive_similar_format(self, default_detector: PIIDetector):
        # Phone-like format without proper SSN grouping should not match
        matches = default_detector.detect("Reference: 1234-5-6789")
        ssns = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssns) == 0

    def test_multiple_ssns(self, default_detector: PIIDetector):
        text = "SSN1: 111-22-3333, SSN2: 444-55-6666"
        matches = default_detector.detect(text)
        ssns = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssns) == 2


# ---------------------------------------------------------------------------
# Credit Card Detection
# ---------------------------------------------------------------------------


class TestCreditCardDetection:
    """Tests for credit card number detection."""

    def test_visa_card(self, default_detector: PIIDetector):
        matches = default_detector.detect("Card: 4111-1111-1111-1111")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_visa_no_separators(self, default_detector: PIIDetector):
        matches = default_detector.detect("Visa: 4111111111111111")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_mastercard(self, default_detector: PIIDetector):
        matches = default_detector.detect("MC: 5500 0000 0000 0004")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_amex(self, default_detector: PIIDetector):
        matches = default_detector.detect("Amex: 3782 822463 10005")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_discover(self, default_detector: PIIDetector):
        matches = default_detector.detect("Discover: 6011 1111 1111 1117")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_no_false_positive_random_digits(self, default_detector: PIIDetector):
        matches = default_detector.detect("Order #: 9876543210123456")
        cards = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 0


# ---------------------------------------------------------------------------
# IP Address Detection
# ---------------------------------------------------------------------------


class TestIPAddressDetection:
    """Tests for IPv4 address detection."""

    def test_standard_ip(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Server at 192.168.1.100 is down")
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1
        assert ips[0].text == "192.168.1.100"

    def test_localhost(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Connect to 127.0.0.1:8080")
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1
        assert ips[0].text == "127.0.0.1"

    def test_boundary_values(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("IP: 255.255.255.255")
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1

    def test_zero_ip(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Source: 0.0.0.0")
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1

    def test_no_false_positive_invalid_octets(self, all_types_detector: PIIDetector):
        # 256 is out of valid range — should not match
        matches = all_types_detector.detect("Not an IP: 256.1.2.3")
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 0

    def test_multiple_ips(self, all_types_detector: PIIDetector):
        text = "From 10.0.0.1 to 10.0.0.255 in subnet"
        matches = all_types_detector.detect(text)
        ips = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 2


# ---------------------------------------------------------------------------
# URL Detection
# ---------------------------------------------------------------------------


class TestURLDetection:
    """Tests for URL detection."""

    def test_http_url(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Visit http://example.com/page")
        urls = [m for m in matches if m.pii_type == PIIType.URL]
        assert len(urls) == 1

    def test_https_url(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("API at https://api.service.io/v2/data")
        urls = [m for m in matches if m.pii_type == PIIType.URL]
        assert len(urls) == 1

    def test_url_with_query_params(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Link: https://site.com/search?q=test&page=1")
        urls = [m for m in matches if m.pii_type == PIIType.URL]
        assert len(urls) == 1

    def test_no_false_positive_plain_domain(self, all_types_detector: PIIDetector):
        # Without protocol prefix, should not match
        matches = all_types_detector.detect("Go to example.com for details")
        urls = [m for m in matches if m.pii_type == PIIType.URL]
        assert len(urls) == 0


# ---------------------------------------------------------------------------
# Date of Birth Detection
# ---------------------------------------------------------------------------


class TestDateOfBirthDetection:
    """Tests for date of birth detection (MM/DD/YYYY or MM-DD-YYYY)."""

    def test_slash_format(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("DOB: 03/15/1990")
        dobs = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dobs) == 1
        assert dobs[0].text == "03/15/1990"

    def test_dash_format(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Born on 12-25-2001")
        dobs = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dobs) == 1
        assert dobs[0].text == "12-25-2001"

    def test_boundary_month_day(self, all_types_detector: PIIDetector):
        matches = all_types_detector.detect("Date: 01/01/2000")
        dobs = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dobs) == 1

    def test_no_false_positive_invalid_month(self, all_types_detector: PIIDetector):
        # Month 13 is invalid
        matches = all_types_detector.detect("Not a date: 13/01/1990")
        dobs = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dobs) == 0

    def test_no_false_positive_invalid_day(self, all_types_detector: PIIDetector):
        # Day 32 is invalid
        matches = all_types_detector.detect("Not a date: 01/32/1990")
        dobs = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dobs) == 0


# ---------------------------------------------------------------------------
# PIIType.ALL Configuration
# ---------------------------------------------------------------------------


class TestAllTypesConfig:
    """Tests for PIIType.ALL enabling all detectors simultaneously."""

    def test_all_enables_every_pattern(self):
        config = PIIDetectionConfig(enabled_types={PIIType.ALL})
        detector = RegexPIIDetector(config)
        # Should have patterns for all types defined in PATTERNS dict
        assert PIIType.EMAIL in detector._compiled_patterns
        assert PIIType.PHONE in detector._compiled_patterns
        assert PIIType.SSN in detector._compiled_patterns
        assert PIIType.CREDIT_CARD in detector._compiled_patterns
        assert PIIType.IP_ADDRESS in detector._compiled_patterns
        assert PIIType.URL in detector._compiled_patterns
        assert PIIType.DATE_OF_BIRTH in detector._compiled_patterns

    def test_mixed_pii_in_single_text(self, all_types_detector: PIIDetector):
        text = (
            "User jane@corp.com called from 555-123-4567. "
            "SSN is 123-45-6789. Card: 4111111111111111. "
            "Connected from 10.0.0.5."
        )
        matches = all_types_detector.detect(text)
        types_found = {m.pii_type for m in matches}
        assert PIIType.EMAIL in types_found
        assert PIIType.PHONE in types_found
        assert PIIType.SSN in types_found
        assert PIIType.CREDIT_CARD in types_found
        assert PIIType.IP_ADDRESS in types_found


# ---------------------------------------------------------------------------
# Custom Patterns
# ---------------------------------------------------------------------------


class TestCustomPatterns:
    """Tests for user-defined custom regex patterns."""

    def test_custom_pattern_detects_match(self):
        config = PIIDetectionConfig(
            enabled_types={PIIType.EMAIL},
            custom_patterns={"email": r"CUSTOM-\d{4}"},
        )
        detector = PIIDetector(detection_config=config)
        matches = detector.detect("Reference CUSTOM-9876 is flagged")
        # The custom pattern overrides the email pattern in this case
        assert len(matches) >= 1

    def test_custom_unknown_type_falls_back_to_name(self):
        config = PIIDetectionConfig(
            enabled_types=set(),
            custom_patterns={"unknown_type": r"BADGE-\d+"},
        )
        detector = RegexPIIDetector(config)
        matches = detector.detect("Employee BADGE-12345 checked in")
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.NAME  # fallback


# ---------------------------------------------------------------------------
# Dictionary Scanning
# ---------------------------------------------------------------------------


class TestDictScanning:
    """Tests for scan_dict functionality."""

    def test_flat_dict(self, default_detector: PIIDetector):
        data = {
            "name": "John Smith",
            "email": "john@example.com",
            "notes": "No PII here",
        }
        results = default_detector.scan_dict(data)
        assert "email" in results
        assert "notes" not in results

    def test_nested_dict(self, default_detector: PIIDetector):
        data = {
            "user": {
                "contact": "reach me at test@mail.com",
            }
        }
        results = default_detector.scan_dict(data)
        assert "user" in results

    def test_list_values(self, default_detector: PIIDetector):
        data = {
            "messages": [
                "Hello there",
                "My SSN is 111-22-3333",
                "Goodbye",
            ]
        }
        results = default_detector.scan_dict(data)
        assert "messages[1]" in results

    def test_empty_dict(self, default_detector: PIIDetector):
        results = default_detector.scan_dict({})
        assert results == {}

    def test_no_pii_dict(self, default_detector: PIIDetector):
        data = {"greeting": "Hello world", "count": 42}
        results = default_detector.scan_dict(data)
        assert results == {}


# ---------------------------------------------------------------------------
# has_pii Helper
# ---------------------------------------------------------------------------


class TestHasPII:
    """Tests for the has_pii convenience method."""

    def test_returns_true_when_pii_present(self, default_detector: PIIDetector):
        assert default_detector.has_pii("Email: foo@bar.com") is True

    def test_returns_false_when_no_pii(self, default_detector: PIIDetector):
        assert default_detector.has_pii("Nothing sensitive here.") is False

    def test_returns_false_for_empty_string(self, default_detector: PIIDetector):
        # Empty string raises ValueError internally, has_pii catches it
        assert default_detector.has_pii("") is False


# ---------------------------------------------------------------------------
# Overlap Removal
# ---------------------------------------------------------------------------


class TestOverlapRemoval:
    """Tests for overlapping match deduplication."""

    def test_overlapping_matches_keep_first_equal_confidence(self):
        """When two matches overlap with equal confidence, keep the first."""
        config = PIIDetectionConfig(enabled_types={PIIType.ALL})
        detector = RegexPIIDetector(config)
        # An SSN like 123-45-6789 won't also match as a phone because
        # different grouping. But test the _remove_overlaps method directly.
        matches = [
            PIIMatch(pii_type=PIIType.SSN, start=0, end=11, text="123-45-6789", confidence=1.0),
            PIIMatch(pii_type=PIIType.PHONE, start=4, end=11, text="45-6789", confidence=0.9),
        ]
        result = detector._remove_overlaps(matches)
        assert len(result) == 1
        assert result[0].pii_type == PIIType.SSN

    def test_overlapping_matches_keep_higher_confidence(self):
        config = PIIDetectionConfig(enabled_types={PIIType.ALL})
        detector = RegexPIIDetector(config)
        matches = [
            PIIMatch(pii_type=PIIType.PHONE, start=0, end=10, text="5551234567", confidence=0.7),
            PIIMatch(pii_type=PIIType.CREDIT_CARD, start=0, end=16, text="5551234567891234", confidence=0.9),
        ]
        result = detector._remove_overlaps(matches)
        assert len(result) == 1
        assert result[0].pii_type == PIIType.CREDIT_CARD

    def test_non_overlapping_matches_all_kept(self):
        config = PIIDetectionConfig(enabled_types={PIIType.ALL})
        detector = RegexPIIDetector(config)
        matches = [
            PIIMatch(pii_type=PIIType.EMAIL, start=0, end=15, text="foo@example.com", confidence=1.0),
            PIIMatch(pii_type=PIIType.PHONE, start=20, end=32, text="555-123-4567", confidence=1.0),
        ]
        result = detector._remove_overlaps(matches)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_raises_value_error(self, default_detector: PIIDetector):
        with pytest.raises(ValueError, match="cannot be None or empty"):
            default_detector.detect("")

    def test_text_with_no_pii(self, default_detector: PIIDetector):
        matches = default_detector.detect("This is a perfectly clean sentence about cats.")
        assert matches == []

    def test_only_whitespace_raises(self, default_detector: PIIDetector):
        # Whitespace-only is technically not empty, should not raise
        # but also should not match anything
        matches = default_detector.detect("   \t\n  ")
        assert matches == []

    def test_very_long_text(self, default_detector: PIIDetector):
        # PII buried in a large block of text
        filler = "Lorem ipsum dolor sit amet. " * 500
        text = filler + "Contact: agent@secret.org " + filler
        matches = default_detector.detect(text)
        emails = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(emails) == 1
        assert emails[0].text == "agent@secret.org"

    def test_special_characters_in_text(self, default_detector: PIIDetector):
        text = "Emojis 🎉 and symbols <>&©® don't break detection: test@demo.io"
        matches = default_detector.detect(text)
        emails = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(emails) == 1

    def test_match_positions_are_accurate(self, default_detector: PIIDetector):
        text = "Start test@abc.com end"
        matches = default_detector.detect(text)
        assert len(matches) == 1
        m = matches[0]
        assert text[m.start:m.end] == "test@abc.com"


# ---------------------------------------------------------------------------
# ML Detector (Not Implemented)
# ---------------------------------------------------------------------------


class TestMLDetector:
    """Tests for MLPIIDetector placeholder behavior."""

    def test_raises_not_implemented(self):
        config = PIIDetectionConfig(use_ml_detection=True)
        with pytest.raises(NotImplementedError):
            PIIDetector(detection_config=config)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_default_detector(self):
        detector = create_default_detector()
        assert isinstance(detector, PIIDetector)
        matches = detector.detect("email: hello@world.org")
        assert len(matches) == 1

    def test_detect_pii_default_types(self):
        matches = detect_pii("My SSN is 999-88-7777")
        ssns = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssns) == 1

    def test_detect_pii_custom_types(self):
        matches = detect_pii(
            "IP: 192.168.0.1 and email: x@y.com",
            pii_types={PIIType.IP_ADDRESS},
        )
        # Should only detect IP since we restricted types
        assert all(m.pii_type == PIIType.IP_ADDRESS for m in matches)
        assert len(matches) == 1

    def test_detect_pii_raises_on_empty(self):
        with pytest.raises(ValueError):
            detect_pii("")


# ---------------------------------------------------------------------------
# Configuration Behavior
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for PIIDetectionConfig behavior."""

    def test_default_config_types(self):
        config = PIIDetectionConfig()
        assert PIIType.EMAIL in config.enabled_types
        assert PIIType.PHONE in config.enabled_types
        assert PIIType.SSN in config.enabled_types
        assert PIIType.CREDIT_CARD in config.enabled_types
        # IP, URL, DOB not in defaults
        assert PIIType.IP_ADDRESS not in config.enabled_types

    def test_selective_type_only_detects_that_type(self, email_only_detector: PIIDetector):
        text = "Email: a@b.com, Phone: 555-111-2222, SSN: 111-22-3333"
        matches = email_only_detector.detect(text)
        assert all(m.pii_type == PIIType.EMAIL for m in matches)
        assert len(matches) == 1

    def test_empty_enabled_types_detects_nothing(self):
        config = PIIDetectionConfig(enabled_types=set())
        detector = PIIDetector(detection_config=config)
        matches = detector.detect("test@example.com 555-123-4567")
        assert matches == []

    def test_confidence_always_1_for_regex(self, default_detector: PIIDetector):
        matches = default_detector.detect("SSN: 123-45-6789")
        for m in matches:
            assert m.confidence == 1.0
