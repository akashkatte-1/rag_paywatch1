import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from models.schemas import LogEntry, LogType


class LoggingUtils:
    """Utility class for advanced logging operations"""
    
    def __init__(self, logs_directory: str = "logs"):
        self.logs_directory = Path(logs_directory)
        self.logs_directory.mkdir(exist_ok=True)
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different loggers for different types of events"""
        # Main application logger
        self.app_logger = logging.getLogger("rag_app")
        self.app_logger.setLevel(logging.INFO)
        
        # Query logger
        self.query_logger = logging.getLogger("rag_queries")
        self.query_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger("rag_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # Upload logger
        self.upload_logger = logging.getLogger("rag_uploads")
        self.upload_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for logger in [self.app_logger, self.query_logger, self.error_logger, self.upload_logger]:
            logger.handlers.clear()
        
        # Setup handlers
        self._setup_file_handlers()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # App handler
        app_handler = logging.FileHandler(self.logs_directory / "app.log")
        app_handler.setFormatter(formatter)
        self.app_logger.addHandler(app_handler)
        
        # Query handler
        query_handler = logging.FileHandler(self.logs_directory / "queries.log")
        query_handler.setFormatter(formatter)
        self.query_logger.addHandler(query_handler)
        
        # Error handler
        error_handler = logging.FileHandler(self.logs_directory / "errors.log")
        error_handler.setFormatter(formatter)
        self.error_logger.addHandler(error_handler)
        
        # Upload handler
        upload_handler = logging.FileHandler(self.logs_directory / "uploads.log")
        upload_handler.setFormatter(formatter)
        self.upload_logger.addHandler(upload_handler)
    
    def log_query(self, query: str, response: str, processing_time: float, 
                  sources_used: int = 0, user_id: str = "api_user"):
        """
        Log query processing events
        
        Args:
            query: User query
            response: Generated response
            processing_time: Time taken to process the query
            sources_used: Number of sources used
            user_id: User identifier
        """
        log_data = {
            "query": query[:100] + "..." if len(query) > 100 else query,
            "response_length": len(response),
            "processing_time": processing_time,
            "sources_used": sources_used,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_logger.info(json.dumps(log_data))
        self._write_structured_log("queries", "query_processed", log_data)
    
    def log_upload(self, filename: str, file_size: int, records_processed: int, 
                   vector_count: int, success: bool = True, error_message: str = None):
        """
        Log file upload events
        
        Args:
            filename: Name of uploaded file
            file_size: Size of the file in bytes
            records_processed: Number of records processed
            vector_count: Number of vectors created
            success: Whether upload was successful
            error_message: Error message if upload failed
        """
        log_data = {
            "filename": filename,
            "file_size": file_size,
            "records_processed": records_processed,
            "vector_count": vector_count,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            self.upload_logger.info(json.dumps(log_data))
        else:
            self.error_logger.error(json.dumps(log_data))
        
        self._write_structured_log("uploads", "file_upload", log_data)
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """
        Log error events
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context information
        """
        log_data = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.error_logger.error(json.dumps(log_data))
        self._write_structured_log("errors", "error_occurred", log_data)
    
    def log_application_event(self, event_type: str, message: str, metadata: Dict[str, Any] = None):
        """
        Log general application events
        
        Args:
            event_type: Type of event
            message: Event message
            metadata: Additional metadata
        """
        log_data = {
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.app_logger.info(json.dumps(log_data))
        self._write_structured_log("app", event_type, log_data)
    
    def _write_structured_log(self, log_type: str, event: str, data: Dict[str, Any]):
        """
        Write structured log entry to date-based file
        
        Args:
            log_type: Type of log
            event: Event name
            data: Log data
        """
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.logs_directory / f"{today}_{log_type}.json"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "log_type": log_type,
            "event": event,
            "data": data
        }
        
        # Append to daily log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_logs(self, date: str, log_type: str = "all") -> List[LogEntry]:
        """
        Retrieve logs for a specific date and type
        
        Args:
            date: Date in YYYYMMDD format
            log_type: Type of logs to retrieve
            
        Returns:
            List[LogEntry]: List of log entries
        """
        logs = []
        
        if log_type == "all":
            log_types = ["queries", "uploads", "errors", "app"]
        else:
            log_types = [log_type]
        
        for lt in log_types:
            log_file = self.logs_directory / f"{date}_{lt}.json"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                log_data = json.loads(line.strip())
                                logs.append(LogEntry(
                                    timestamp=datetime.fromisoformat(log_data["timestamp"]),
                                    log_type=log_data["log_type"],
                                    message=log_data.get("event", ""),
                                    metadata=log_data.get("data", {})
                                ))
                except Exception as e:
                    self.log_error("log_retrieval_error", str(e), {"date": date, "log_type": lt})
        
        # Sort logs by timestamp
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs
    
    def get_log_stats(self, date: str) -> Dict[str, int]:
        """
        Get statistics for logs on a specific date
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            Dict[str, int]: Statistics for different log types
        """
        stats = {"queries": 0, "uploads": 0, "errors": 0, "app": 0, "total": 0}
        
        for log_type in ["queries", "uploads", "errors", "app"]:
            log_file = self.logs_directory / f"{date}_{log_type}.json"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        count = sum(1 for line in f if line.strip())
                        stats[log_type] = count
                        stats["total"] += count
                except Exception:
                    pass
        
        return stats


# Global logging instance
logger_utils = LoggingUtils()