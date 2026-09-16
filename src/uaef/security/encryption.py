# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Encryption utilities for data at rest.

This module provides AES-256 encryption for sensitive data including agent traces,
evaluation results, and experiment configurations. It supports customer-managed
encryption keys (CMEK) and key rotation.

**Validates: Requirements 24.2, 24.6**
"""

import base64
import json
import os
from typing import Any, Dict, Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pydantic import BaseModel, Field, field_validator

from uaef.logging import get_logger

logger = get_logger(__name__)


class EncryptionConfig(BaseModel):
    """Configuration for encryption at rest."""
    
    enabled: bool = Field(
        default=True,
        description="Enable/disable encryption"
    )
    key_source: str = Field(
        default="env",
        description=(
            "Key source: 'env' (environment variable), 'file' (key file), "
            "'generated' (auto-generate, in-memory only), or 'kms' (AWS KMS "
            "envelope encryption — a customer-managed key, never leaving KMS "
            "in plaintext form outside this process's memory)"
        )
    )
    key_path: Optional[str] = Field(
        None,
        description="Path to encryption key file (for key_source='file')"
    )
    key_env_var: str = Field(
        default="UAEF_ENCRYPTION_KEY",
        description="Environment variable name for encryption key"
    )
    kms_key_id: Optional[str] = Field(
        None,
        description=(
            "KMS key ID or ARN (for key_source='kms'). Required when "
            "key_source='kms'."
        )
    )
    kms_region: Optional[str] = Field(
        None,
        description=(
            "AWS region for the KMS client (for key_source='kms'). Defaults "
            "to the boto3 default region if unset."
        )
    )
    algorithm: str = Field(
        default="AES-256-GCM",
        description="Encryption algorithm (currently only AES-256-GCM supported)"
    )
    key_rotation_enabled: bool = Field(
        default=False,
        description="Enable key rotation support"
    )
    
    @field_validator("key_source")
    @classmethod
    def validate_key_source(cls, v: str) -> str:
        """Validate key source is supported."""
        valid_sources = ["env", "file", "generated", "kms"]
        if v not in valid_sources:
            raise ValueError(
                f"Invalid key_source: {v}. Must be one of {valid_sources}"
            )
        return v
    
    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate algorithm is supported."""
        if v != "AES-256-GCM":
            raise ValueError(
                f"Unsupported algorithm: {v}. Currently only AES-256-GCM is supported"
            )
        return v
    
    @classmethod
    def from_env(cls) -> "EncryptionConfig":
        """Load encryption configuration from environment variables."""
        return cls(
            enabled=os.getenv("UAEF_ENCRYPTION_ENABLED", "true").lower() == "true",
            key_source=os.getenv("UAEF_ENCRYPTION_KEY_SOURCE", "env"),
            key_path=os.getenv("UAEF_ENCRYPTION_KEY_PATH"),
            key_env_var=os.getenv("UAEF_ENCRYPTION_KEY_ENV_VAR", "UAEF_ENCRYPTION_KEY"),
            kms_key_id=os.getenv("UAEF_ENCRYPTION_KMS_KEY_ID"),
            kms_region=os.getenv("UAEF_ENCRYPTION_KMS_REGION"),
            algorithm=os.getenv("UAEF_ENCRYPTION_ALGORITHM", "AES-256-GCM"),
            key_rotation_enabled=os.getenv("UAEF_KEY_ROTATION_ENABLED", "false").lower() == "true",
        )


class EncryptionError(Exception):
    """Base exception for encryption errors."""
    pass


class KeyLoadError(EncryptionError):
    """Exception raised when encryption key cannot be loaded."""
    pass


class EncryptionOperationError(EncryptionError):
    """Exception raised when encryption/decryption operation fails."""
    pass


class DataEncryptor:
    """
    AES-256-GCM encryption for data at rest.
    
    This class provides transparent encryption and decryption of sensitive data
    using AES-256 in GCM (Galois/Counter Mode) which provides both confidentiality
    and authenticity.
    
    Features:
    - AES-256-GCM encryption (industry standard)
    - Customer-managed encryption keys (CMEK)
    - Key rotation support
    - Transparent encryption/decryption for dictionaries and objects
    
    Example:
        >>> config = EncryptionConfig(enabled=True, key_source="env")
        >>> encryptor = DataEncryptor(config)
        >>> encrypted = encryptor.encrypt("sensitive data")
        >>> decrypted = encryptor.decrypt(encrypted)
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize data encryptor.
        
        Args:
            config: Encryption configuration (uses defaults if None)
            
        Raises:
            KeyLoadError: If encryption key cannot be loaded
        """
        self.config = config or EncryptionConfig.from_env()
        self._key: Optional[bytes] = None
        self._key_id: str = "default"
        #: KMS-wrapped (encrypted) form of ``self._key``, set only when
        #: ``key_source == "kms"``. Embedded as an additive "wrapped_key"
        #: field in every encrypted payload (format version stays "1" —
        #: existing ciphertext without this field decrypts unchanged) so
        #: decryption can unwrap the data key via KMS without needing the
        #: original in-memory plaintext key — this is what makes
        #: KMS-sourced ciphertext decryptable across process restarts and by
        #: any encryptor instance holding the same KMS key permission, not
        #: just the one that encrypted it.
        self._kms_wrapped_key: Optional[bytes] = None
        #: Retired (rotated-out) keys, keyed by their key_id, kept in memory
        #: so data encrypted before a rotate_key() call remains decryptable
        #: (security review L-02 — rotate_key() previously only switched to
        #: the new key with no path back to the old one, so decrypt() always
        #: raised for anything encrypted before a rotation).
        self._old_keys: Dict[str, bytes] = {}
        
        if self.config.enabled:
            self._load_key()
    
    def _load_key(self) -> None:
        """
        Load encryption key based on configuration.
        
        Raises:
            KeyLoadError: If key cannot be loaded
        """
        try:
            if self.config.key_source == "env":
                self._load_key_from_env()
            elif self.config.key_source == "file":
                self._load_key_from_file()
            elif self.config.key_source == "generated":
                self._generate_key()
            elif self.config.key_source == "kms":
                self._load_key_from_kms()
            else:
                raise KeyLoadError(f"Unsupported key source: {self.config.key_source}")
        except Exception as e:
            raise KeyLoadError(f"Failed to load encryption key: {str(e)}") from e
    
    def _load_key_from_env(self) -> None:
        """
        Load encryption key from environment variable.
        
        Raises:
            KeyLoadError: If environment variable is not set or invalid
        """
        key_str = os.getenv(self.config.key_env_var)
        if not key_str:
            raise KeyLoadError(
                f"Encryption key not found in environment variable: {self.config.key_env_var}. "
                f"Set {self.config.key_env_var} or disable encryption."
            )
        
        try:
            # Decode base64-encoded key
            self._key = base64.b64decode(key_str)
            
            # Validate key length (256 bits = 32 bytes)
            if len(self._key) != 32:
                raise KeyLoadError(
                    f"Invalid key length: {len(self._key)} bytes. "
                    f"AES-256 requires 32 bytes (256 bits)."
                )
        except Exception as e:
            raise KeyLoadError(f"Failed to decode encryption key: {str(e)}") from e
    
    def _load_key_from_file(self) -> None:
        """
        Load encryption key from file.
        
        Raises:
            KeyLoadError: If file doesn't exist or is invalid
        """
        if not self.config.key_path:
            raise KeyLoadError("key_path must be specified when key_source='file'")
        
        try:
            with open(self.config.key_path, "rb") as f:
                key_data = f.read()
            
            # Try to decode as base64 first
            try:
                self._key = base64.b64decode(key_data)
            except Exception:
                # If not base64, use raw bytes
                self._key = key_data
            
            # Validate key length
            if len(self._key) != 32:
                raise KeyLoadError(
                    f"Invalid key length: {len(self._key)} bytes. "
                    f"AES-256 requires 32 bytes (256 bits)."
                )
        except FileNotFoundError:
            raise KeyLoadError(f"Encryption key file not found: {self.config.key_path}")
        except Exception as e:
            raise KeyLoadError(f"Failed to load key from file: {str(e)}") from e

    def _load_key_from_kms(self) -> None:
        """
        Generate an AES-256 data key via AWS KMS envelope encryption.

        Requests a new plaintext data key plus its KMS-encrypted ("wrapped")
        form from ``kms:GenerateDataKey``. The plaintext key is used exactly
        like any other key_source for the AES-256-GCM operations below and is
        never persisted; the wrapped form is embedded in every ciphertext
        this encryptor produces so a *different* encryptor instance (a new
        process, a different host) can recover the same plaintext key via
        ``kms:Decrypt`` on the wrapped blob, using its own IAM permission on
        the KMS key rather than needing the original plaintext key passed
        around out-of-band.

        Raises:
            KeyLoadError: If kms_key_id is not configured or the KMS call fails
        """
        if not self.config.kms_key_id:
            raise KeyLoadError("kms_key_id must be specified when key_source='kms'")

        try:
            import boto3
        except ImportError as e:
            raise KeyLoadError(
                "boto3 is required for key_source='kms' but is not installed"
            ) from e

        try:
            kms_kwargs: Dict[str, Any] = {}
            if self.config.kms_region:
                kms_kwargs["region_name"] = self.config.kms_region
            client = boto3.client("kms", **kms_kwargs)
            response = client.generate_data_key(
                KeyId=self.config.kms_key_id,
                KeySpec="AES_256",
            )
            self._key = response["Plaintext"]
            self._kms_wrapped_key = response["CiphertextBlob"]
            # key_id here identifies which KMS-wrapped key was used, distinct
            # from the KMS key ARN itself, so rotate_key()/decrypt() logic
            # stays uniform across key sources.
            self._key_id = "kms"
        except Exception as e:
            raise KeyLoadError(f"Failed to generate data key from KMS: {str(e)}") from e
    
    def _generate_key(self) -> None:
        """
        Generate a new encryption key.
        
        This should only be used for testing or initial setup.
        In production, use customer-managed keys.
        """
        self._key = os.urandom(32)  # 256 bits
        
        # Do NOT log or print the generated key material — this class exposes
        # no public accessor for it, so the key exists only for the lifetime of
        # this process. Data encrypted with a generated key is undecryptable
        # after the process exits. key_source='generated' is intended for
        # tests/ephemeral use only; production callers should use key_source
        # 'env' or 'file' with a customer-managed key.
        logger.warning(
            "Generated a new in-memory encryption key (key_source='generated'). "
            "This key is not persisted or logged and will be lost when this "
            "process exits, making previously encrypted data unrecoverable. "
            "Use key_source='env' or 'file' with a customer-managed key for any "
            "data that must outlive this process."
        )
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Base64-encoded encrypted data with format: version:key_id:iv:ciphertext:tag
            
        Raises:
            EncryptionOperationError: If encryption fails
            ValueError: If data is invalid
        """
        if not self.config.enabled:
            # If encryption is disabled, return data as-is (base64 encoded for consistency)
            if isinstance(data, str):
                data = data.encode('utf-8')
            return base64.b64encode(data).decode('utf-8')
        
        if not data:
            raise ValueError("Data cannot be empty")
        
        try:
            # Convert string to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Generate random IV (96 bits for GCM)
            iv = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self._key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
            
            # Get authentication tag
            tag = encryptor.tag
            
            # Format: version:key_id:iv:ciphertext:tag (all base64 encoded)
            encrypted_data = {
                "version": "1",
                "key_id": self._key_id,
                "iv": base64.b64encode(iv).decode('utf-8'),
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "tag": base64.b64encode(tag).decode('utf-8'),
            }
            # KMS envelope encryption (security review L-02): embed the
            # KMS-wrapped data key alongside the ciphertext it protects, so
            # any encryptor instance with kms:Decrypt on this KMS key can
            # recover the plaintext key and decrypt — not just the process
            # that originally called generate_data_key. Absent for
            # non-KMS key sources; older ciphertext without this field
            # decrypts exactly as before (see decrypt()'s .get() below).
            if self._kms_wrapped_key is not None:
                encrypted_data["wrapped_key"] = base64.b64encode(
                    self._kms_wrapped_key
                ).decode('utf-8')
            
            # Encode as JSON and then base64 for storage
            json_str = json.dumps(encrypted_data)
            return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
            
        except Exception as e:
            raise EncryptionOperationError(f"Encryption failed: {str(e)}") from e
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data encrypted with AES-256-GCM.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data as string
            
        Raises:
            EncryptionOperationError: If decryption fails
            ValueError: If encrypted data is invalid
        """
        if not encrypted_data:
            raise ValueError("Encrypted data cannot be empty")
        
        if not self.config.enabled:
            # If encryption is disabled, decode base64 directly
            try:
                return base64.b64decode(encrypted_data).decode('utf-8')
            except Exception as e:
                raise EncryptionOperationError(f"Failed to decode data: {str(e)}") from e
        
        try:
            # Decode base64 and parse JSON
            json_str = base64.b64decode(encrypted_data).decode('utf-8')
            data_dict = json.loads(json_str)
            
            # Extract components
            version = data_dict.get("version")
            key_id = data_dict.get("key_id")
            iv = base64.b64decode(data_dict["iv"])
            ciphertext = base64.b64decode(data_dict["ciphertext"])
            tag = base64.b64decode(data_dict["tag"])
            wrapped_key_b64 = data_dict.get("wrapped_key")
            
            # Validate version
            if version != "1":
                raise EncryptionOperationError(f"Unsupported encryption version: {version}")
            
            # Resolve which plaintext key to decrypt with (security review
            # L-02 — multi-key rotation): try, in order, the KMS-wrapped key
            # embedded in this ciphertext (if present — always correct for
            # KMS, since KMS can unwrap any data key it originally wrapped
            # regardless of which encryptor instance / rotation generation
            # produced it), the current in-memory key, then any retired key
            # this instance still remembers from a prior rotate_key() call.
            # A previous version of this method treated any key_id mismatch
            # under key_rotation_enabled as a hard failure ("Implement key
            # rotation logic") — this replaces that stub with the resolution
            # this docstring already promised.
            decrypt_key: Optional[bytes] = None
            if wrapped_key_b64 is not None:
                decrypt_key = self._unwrap_kms_key(base64.b64decode(wrapped_key_b64))
            if decrypt_key is None:
                if key_id == self._key_id:
                    decrypt_key = self._key
                elif key_id in self._old_keys:
                    decrypt_key = self._old_keys[key_id]
                elif not self.config.key_rotation_enabled:
                    # Rotation not enabled — preserve the original strict
                    # behavior (only the single current key is ever tried).
                    decrypt_key = self._key
                else:
                    raise EncryptionOperationError(
                        f"No usable key found to decrypt data encrypted with "
                        f"key '{key_id}'. Current key is '{self._key_id}' and "
                        f"{len(self._old_keys)} retired key(s) are known. Call "
                        f"rotate_key(..., retain_old_key=True) when rotating "
                        f"if old ciphertext must remain decryptable."
                    )
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(decrypt_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext.decode('utf-8')
            
        except json.JSONDecodeError as e:
            raise EncryptionOperationError(f"Invalid encrypted data format: {str(e)}") from e
        except KeyError as e:
            raise EncryptionOperationError(f"Missing required field in encrypted data: {str(e)}") from e
        except Exception as e:
            raise EncryptionOperationError(f"Decryption failed: {str(e)}") from e
    
    def _unwrap_kms_key(self, wrapped_key: bytes) -> Optional[bytes]:
        """Unwrap a KMS-wrapped data key via ``kms:Decrypt``.

        Returns ``None`` (rather than raising) when this encryptor was not
        configured for KMS or the unwrap fails for a reason other than a
        missing dependency, so callers fall back to the plaintext-key
        resolution path in :meth:`decrypt` — a ciphertext carrying a
        ``wrapped_key`` field is always assumed to be KMS in origin, but a
        best-effort fallback avoids a hard crash if, for example, boto3 is
        unavailable in a context that otherwise doesn't need it.
        """
        if self.config.key_source != "kms":
            return None
        try:
            import boto3
        except ImportError:
            return None
        try:
            kms_kwargs: Dict[str, Any] = {}
            if self.config.kms_region:
                kms_kwargs["region_name"] = self.config.kms_region
            client = boto3.client("kms", **kms_kwargs)
            response = client.decrypt(CiphertextBlob=wrapped_key)
            return response["Plaintext"]
        except Exception:
            return None

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """
        Encrypt a dictionary by converting to JSON and encrypting.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Base64-encoded encrypted data
            
        Raises:
            EncryptionOperationError: If encryption fails
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        try:
            json_str = json.dumps(data)
            return self.encrypt(json_str)
        except Exception as e:
            raise EncryptionOperationError(f"Failed to encrypt dictionary: {str(e)}") from e
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt data and parse as dictionary.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted dictionary
            
        Raises:
            EncryptionOperationError: If decryption fails
            ValueError: If encrypted data is invalid
        """
        try:
            json_str = self.decrypt(encrypted_data)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise EncryptionOperationError(f"Decrypted data is not valid JSON: {str(e)}") from e
        except Exception as e:
            raise EncryptionOperationError(f"Failed to decrypt dictionary: {str(e)}") from e
    
    def rotate_key(
        self,
        new_key: bytes,
        new_key_id: str,
        *,
        retain_old_key: bool = True,
    ) -> None:
        """
        Rotate to a new encryption key.
        
        New calls to :meth:`encrypt` use the new key immediately. Existing
        ciphertext does NOT need to be re-encrypted for :meth:`decrypt` to
        keep working: with ``retain_old_key=True`` (the default), the
        outgoing key is kept in memory under its own ``key_id`` and
        :meth:`decrypt` transparently selects the correct key based on the
        ``key_id`` recorded in each ciphertext (security review L-02 — this
        previously switched the key with no path back to the old one, so
        anything encrypted before a rotation became permanently
        undecryptable by this instance; see :meth:`decrypt` for the
        resolution logic).

        Retained old keys live only in this process's memory — they are
        never persisted, and a new process (e.g. after a Lambda cold start)
        will not have them unless re-supplied via whatever mechanism
        originally provided them (this method doesn't prescribe one; it's
        the caller's responsibility to call ``rotate_key`` with the
        appropriate historical key material on process start if continuity
        across restarts is required for non-KMS key sources). KMS-sourced
        ciphertext does not have this limitation, since the wrapped key
        travels with the ciphertext itself and any instance with KMS
        permission can unwrap it.
        
        Args:
            new_key: New encryption key (32 bytes for AES-256)
            new_key_id: Identifier for the new key
            retain_old_key: If True (default), keep the outgoing key
                available for decrypting ciphertext produced before this
                rotation. Set False only if you are certain no un-migrated
                ciphertext under the old key remains.
            
        Raises:
            ValueError: If new key is invalid
        """
        if len(new_key) != 32:
            raise ValueError(
                f"Invalid key length: {len(new_key)} bytes. "
                f"AES-256 requires 32 bytes (256 bits)."
            )
        
        if retain_old_key and self._key is not None:
            self._old_keys[self._key_id] = self._key

        self._key = new_key
        self._key_id = new_key_id
        # A rotation replaces the plaintext key wholesale; any previously
        # loaded KMS-wrapped form no longer corresponds to self._key, so
        # clear it. New encrypt() calls after this point use new_key
        # directly (non-KMS), which is correct for a manually-supplied
        # rotation key.
        self._kms_wrapped_key = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self.config.enabled
    
    @property
    def key_id(self) -> str:
        """Get current key ID."""
        return self._key_id


# Global encryptor instance
_encryptor: Optional[DataEncryptor] = None


def get_encryptor() -> DataEncryptor:
    """
    Get the global data encryptor.
    
    Loads from environment configuration on first call.
    
    Returns:
        DataEncryptor instance
    """
    global _encryptor
    if _encryptor is None:
        _encryptor = DataEncryptor()
    return _encryptor


def set_encryptor(encryptor: DataEncryptor) -> None:
    """
    Set the global data encryptor.
    
    Args:
        encryptor: DataEncryptor instance to set as global
    """
    global _encryptor
    _encryptor = encryptor


def reset_encryptor() -> None:
    """Reset the global encryptor to None."""
    global _encryptor
    _encryptor = None
