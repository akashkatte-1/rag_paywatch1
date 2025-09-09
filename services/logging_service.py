from datetime import datetime
from typing import Dict, Any, Optional, List
from models.schemas import LogEntry, LogType, LogResponse
from security.logging_utils import logger_utils
import re

class LoggingService:
    """Service for managing application logging operations"""
    
    def __init__(self):
        self.logger_utils = logger_utils
    
    def log_query_processing(self, query: str, response: str, processing_time: float, 
                           sources_used: int = 0, user_id: str = "api_user"):
        """
        Log query processing events
        
        Args:
            query: User query
            response: Generated response
            processing_time: Time taken to process
            sources_used: Number of sources used
            user_id: User identifier
        """
        self.logger_utils.log_query(
            query=query,
            response=response,
            processing_time=processing_time,
            sources_used=sources_used,
            user_id=user_id
        )
    
    def log_file_upload(self, filename: str, file_size: int, records_processed: int, 
                       vector_count: int, success: bool = True, error_message: Optional[str] = None):
        """
        Log file upload events
        
        Args:
            filename: Name of uploaded file
            file_size: Size of the file
            records_processed: Number of records processed
            vector_count: Number of vectors created
            success: Whether upload was successful
            error_message: Error message if failed
        """
        self.logger_utils.log_upload(
            filename=filename,
            file_size=file_size,
            records_processed=records_processed,
            vector_count=vector_count,
            success=success,
            error_message=error_message
        )
    
    def log_error_event(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log error events
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        self.logger_utils.log_error(
            error_type=error_type,
            error_message=error_message,
            context=context or {}
        )
    @staticmethod
    def extract_metadata_from_filename(filename: str) -> dict:
        # Example: Android(0-2).xlsx
        match = re.match(r"([A-Za-z]+)\((\d+)-(\d+)\)", filename)
        if match:
            tech = match.group(1)
            exp_min = int(match.group(2))
            exp_max = int(match.group(3))
            return {"technology": tech, "exp_min": exp_min, "exp_max": exp_max, "filename": filename}
        return {"technology": None, "exp_min": None, "exp_max": None, "filename": filename}
    def log_application_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log general application events
        
        Args:
            event_type: Type of event
            message: Event message
            metadata: Additional metadata
        """
        self.logger_utils.log_application_event(
            event_type=event_type,
            message=message,
            metadata=metadata
        )
    def log_error(self, error_type: str, message: str, metadata: dict):
        """Log error events for monitoring and debugging."""
        self.log_application_event(error_type, message, metadata)
    def get_logs_by_date(self, date: str, log_type: str = "all") -> LogResponse:
        """
        Retrieve logs for a specific date and type
        
        Args:
            date: Date in YYYYMMDD format
            log_type: Type of logs to retrieve
            
        Returns:
            LogResponse: Response containing logs
        """
        try:
            # Validate date format
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYYMMDD format.")
        
        # Validate log type
        if log_type not in ["all", "queries", "uploads", "errors", "app"]:
            raise ValueError("Invalid log type. Use: all, queries, uploads, errors, or app")
        
        logs = self.logger_utils.get_logs(date, log_type)
        
        return LogResponse(
            date=date,
            log_type=log_type,
            logs=logs,
            total_count=len(logs)
        )
    
    def get_log_statistics(self, date: str) -> Dict[str, int]:
        """
        Get log statistics for a specific date
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            Dict[str, int]: Log statistics
        """
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYYMMDD format.")
        
        return self.logger_utils.get_log_stats(date)
    
    def get_recent_errors(self, limit: int = 10) -> List[LogEntry]:
        """
        Get recent error logs
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List[LogEntry]: Recent error logs
        """
        today = datetime.now().strftime("%Y%m%d")
        logs = self.logger_utils.get_logs(today, "errors")
        
        # Return only the most recent errors up to the limit
        return logs[:limit]
    
    def get_query_performance_stats(self, date: str) -> Dict[str, Any]:
        """
        Get query performance statistics for a specific date
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            Dict[str, Any]: Performance statistics
        """
        logs = self.logger_utils.get_logs(date, "queries")
        
        if not logs:
            return {
                "total_queries": 0,
                "average_processing_time": 0,
                "min_processing_time": 0,
                "max_processing_time": 0,
                "total_sources_used": 0
            }
        
        processing_times = []
        total_sources = 0
        
        for log in logs:
            if log.metadata and "processing_time" in log.metadata:
                processing_times.append(log.metadata["processing_time"])
            if log.metadata and "sources_used" in log.metadata:
                total_sources += log.metadata["sources_used"]
        
        return {
            "total_queries": len(logs),
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "total_sources_used": total_sources
        }


# Global logging service instance
logging_service = LoggingService()