from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QueryRequest(BaseModel):
    """Schema for user query requests"""
    question: str = Field(..., description="Natural language question about the data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the average CTC for candidates with Python skills?"
            }
        }


class QueryResponse(BaseModel):
    """Schema for query responses"""
    answer: str = Field(..., description="Generated answer from the RAG system")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    sources_used: int = Field(..., description="Number of documents retrieved for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the data, the average CTC for candidates with Python skills is $45,000 USD per year.",
                "processing_time": 1.23,
                "sources_used": 5
            }
        }


class UploadResponse(BaseModel):
    """Schema for file upload responses"""
    message: str = Field(..., description="Upload status message")
    filename: str = Field(..., description="Name of the uploaded file")
    records_processed: int = Field(..., description="Number of records processed")
    vector_count: int = Field(..., description="Number of vectors stored in FAISS")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "File uploaded and processed successfully",
                "filename": "employee_data.xlsx",
                "records_processed": 150,
                "vector_count": 150
            }
        }


class LogType(str, Enum):
    """Enum for different log types"""
    QUERIES = "queries"
    ERRORS = "errors"
    UPLOADS = "uploads"
    ALL = "all"


class LogEntry(BaseModel):
    """Schema for log entries"""
    timestamp: datetime
    log_type: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "log_type": "queries",
                "message": "Query processed successfully",
                "metadata": {
                    "query": "What is the average salary?",
                    "processing_time": 1.2,
                    "user_id": "api_user"
                }
            }
        }


class LogResponse(BaseModel):
    """Schema for log retrieval responses"""
    date: str = Field(..., description="Date of logs (YYYYMMDD format)")
    log_type: str = Field(..., description="Type of logs retrieved")
    logs: List[LogEntry] = Field(..., description="List of log entries")
    total_count: int = Field(..., description="Total number of log entries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "20240115",
                "log_type": "all",
                "logs": [],
                "total_count": 25
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid file format. Only .xlsx and .xls files are supported.",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class CandidateData(BaseModel):
    """Schema for candidate data structure"""
    name: str
    location: str
    ctc_inr: float = Field(..., description="CTC in Indian Rupees (LPA)")
    skills: str
    experience: str = Field(..., description="Experience in years/months format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "location": "Bangalore",
                "ctc_inr": 12.5,
                "skills": "Python, Django, React, AWS",
                "experience": "3y 6m"
            }
        }


class ExchangeRateRequest(BaseModel):
    """Schema for exchange rate requests"""
    from_currency: str = Field(default="INR", description="Source currency code")
    to_currency: str = Field(default="USD", description="Target currency code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_currency": "INR",
                "to_currency": "USD"
            }
        }


class ExchangeRateResponse(BaseModel):
    """Schema for exchange rate responses"""
    from_currency: str
    to_currency: str
    rate: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_currency": "INR",
                "to_currency": "USD",
                "rate": 0.012,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }