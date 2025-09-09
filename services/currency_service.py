import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from security.logging_utils import logger_utils


class CurrencyService:
    """Service for handling currency conversions with real-time exchange rates"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)  # Cache rates for 30 minutes
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
        self.fallback_url = "https://open.er-api.com/v6/latest"
        
    def get_exchange_rate(self, from_currency: str = "INR", to_currency: str = "USD") -> Optional[float]:
        """
        Get exchange rate between two currencies
        
        Args:
            from_currency: Source currency code (default: INR)
            to_currency: Target currency code (default: USD)
            
        Returns:
            Optional[float]: Exchange rate or None if failed
        """
        cache_key = f"{from_currency}_{to_currency}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger_utils.log_application_event(
                "cache_hit", 
                f"Using cached exchange rate for {from_currency} to {to_currency}",
                {"from_currency": from_currency, "to_currency": to_currency, "rate": self.cache[cache_key]["rate"]}
            )
            return self.cache[cache_key]["rate"]
        
        # Try to fetch fresh rate
        rate = self._fetch_exchange_rate(from_currency, to_currency)
        
        if rate is not None:
            # Cache the rate
            self.cache[cache_key] = {
                "rate": rate,
                "timestamp": datetime.now(),
                "from_currency": from_currency,
                "to_currency": to_currency
            }
            
            logger_utils.log_application_event(
                "exchange_rate_fetched",
                f"Successfully fetched exchange rate for {from_currency} to {to_currency}",
                {"from_currency": from_currency, "to_currency": to_currency, "rate": rate}
            )
        else:
            logger_utils.log_error(
                "exchange_rate_fetch_failed",
                f"Failed to fetch exchange rate for {from_currency} to {to_currency}",
                {"from_currency": from_currency, "to_currency": to_currency}
            )
        
        return rate
    
    def _fetch_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Fetch exchange rate from external API
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Optional[float]: Exchange rate or None if failed
        """
        # Try primary API
        rate = self._try_api_endpoint(self.base_url, from_currency, to_currency)
        
        if rate is None:
            # Try fallback API
            rate = self._try_api_endpoint(self.fallback_url, from_currency, to_currency)
        
        if rate is None:
            # Use hardcoded fallback rates for common conversions
            rate = self._get_fallback_rate(from_currency, to_currency)
        
        return rate
    
    def _try_api_endpoint(self, base_url: str, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Try to fetch rate from a specific API endpoint
        
        Args:
            base_url: Base URL of the API
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Optional[float]: Exchange rate or None if failed
        """
        try:
            url = f"{base_url}/{from_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "rates" in data and to_currency in data["rates"]:
                return float(data["rates"][to_currency])
            
        except Exception as e:
            logger_utils.log_error(
                "api_request_failed",
                f"Failed to fetch from {base_url}: {str(e)}",
                {"url": base_url, "from_currency": from_currency, "to_currency": to_currency}
            )
        
        return None
    
    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Get fallback exchange rates for common currency pairs
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Optional[float]: Fallback exchange rate
        """
        # Hardcoded fallback rates (updated periodically)
        fallback_rates = {
            "INR_USD": 0.012,  # 1 INR = 0.012 USD (approximate)
            "USD_INR": 83.0,   # 1 USD = 83 INR (approximate)
            "EUR_USD": 1.08,   # 1 EUR = 1.08 USD (approximate)
            "USD_EUR": 0.93,   # 1 USD = 0.93 EUR (approximate)
            "GBP_USD": 1.26,   # 1 GBP = 1.26 USD (approximate)
            "USD_GBP": 0.79,   # 1 USD = 0.79 GBP (approximate)
        }
        
        rate_key = f"{from_currency}_{to_currency}"
        
        if rate_key in fallback_rates:
            logger_utils.log_application_event(
                "fallback_rate_used",
                f"Using fallback rate for {from_currency} to {to_currency}",
                {"from_currency": from_currency, "to_currency": to_currency, "rate": fallback_rates[rate_key]}
            )
            return fallback_rates[rate_key]
        
        # If direct conversion not available, try inverse
        inverse_key = f"{to_currency}_{from_currency}"
        if inverse_key in fallback_rates:
            inverse_rate = 1.0 / fallback_rates[inverse_key]
            logger_utils.log_application_event(
                "inverse_fallback_rate_used",
                f"Using inverse fallback rate for {from_currency} to {to_currency}",
                {"from_currency": from_currency, "to_currency": to_currency, "rate": inverse_rate}
            )
            return inverse_rate
        
        return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached rate is still valid
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            bool: True if cache is valid
        """
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]["timestamp"]
        return datetime.now() - cached_time < self.cache_duration
    
    def convert_currency(self, amount: float, from_currency: str = "INR", to_currency: str = "USD") -> Optional[float]:
        """
        Convert amount from one currency to another
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Optional[float]: Converted amount or None if conversion failed
        """
        if from_currency == to_currency:
            return amount
        
        exchange_rate = self.get_exchange_rate(from_currency, to_currency)
        
        if exchange_rate is not None:
            converted_amount = amount * exchange_rate
            logger_utils.log_application_event(
                "currency_converted",
                f"Converted {amount} {from_currency} to {converted_amount:.2f} {to_currency}",
                {
                    "original_amount": amount,
                    "converted_amount": converted_amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": exchange_rate
                }
            )
            return converted_amount
        
        return None
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the currency cache
        
        Returns:
            Dict[str, any]: Cache statistics
        """
        valid_entries = sum(1 for key in self.cache.keys() if self._is_cache_valid(key))
        
        return {
            "total_cached_rates": len(self.cache),
            "valid_cached_rates": valid_entries,
            "cache_duration_minutes": self.cache_duration.total_seconds() / 60,
            "cached_pairs": list(self.cache.keys())
        }
    
    def clear_cache(self):
        """Clear the currency cache"""
        self.cache.clear()
        logger_utils.log_application_event(
            "cache_cleared",
            "Currency cache cleared",
            {"action": "cache_clear"}
        )


# Global currency service instance
currency_service = CurrencyService()