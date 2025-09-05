import os
from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Optional, List
import secrets
import hashlib


class SecurityUtils:
    """Utility class for security-related operations"""
    
    def __init__(self):
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        # Generate a default API key if not provided in environment
        self.valid_api_key = os.getenv("API_KEY", self._generate_default_api_key())
    
    def _generate_default_api_key(self) -> str:
        """Generate a default API key for development"""
        return "sk-rag-" + secrets.token_urlsafe(32)
    
    def get_api_key(self) -> str:
        """Get the valid API key"""
        return self.valid_api_key
    
    def validate_api_key(self, api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))) -> str:
        """
        Validate API key from request headers
        
        Args:
            api_key: API key from request header
            
        Returns:
            str: Validated API key
            
        Raises:
            HTTPException: If API key is invalid or missing
        """
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key is required. Please provide X-API-Key header."
            )
        
        if not self._is_valid_api_key(api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key provided."
            )
        
        return api_key
    
    def _is_valid_api_key(self, provided_key: str) -> bool:
        """
        Check if the provided API key is valid
        
        Args:
            provided_key: API key to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return secrets.compare_digest(provided_key, self.valid_api_key)
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        Hash sensitive data for logging purposes
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            str: Hashed data
        """
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal attacks
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Remove path components and keep only the filename
        import os.path
        sanitized = os.path.basename(filename)
        
        # Remove any remaining dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        return sanitized
    
    def validate_file_type(self, filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate file type based on extension
        
        Args:
            filename: Name of the file to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            bool: True if file type is allowed
        """
        if allowed_extensions is None:
            allowed_extensions = ['.xlsx', '.xls']
        
        file_extension = os.path.splitext(filename.lower())[1]
        return file_extension in allowed_extensions


# Global security instance
security = SecurityUtils()


def get_api_key_dependency():
    """Dependency function for API key validation"""
    return Depends(security.validate_api_key)