from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional

from models.schemas import QueryRequest, QueryResponse, UploadResponse, LogResponse, ErrorResponse
from services.rag_service import rag_service
from services.logging_service import logging_service
from security.security_utils import security


# Initialize FastAPI app
app = FastAPI(
    title="RAG Application Backend",
    description="A comprehensive RAG system for HR data analysis with FAISS vector database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logging_service.log_application_event(
        "app_startup",
        "RAG Application started successfully",
        {"api_key": security.get_api_key()[:10] + "..."}
    )
    print(f"🚀 RAG Application started successfully!")
    print(f"🔑 API Key: {security.get_api_key()}")
    print(f"📊 Vector storage initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logging_service.log_application_event(
        "app_shutdown",
        "RAG Application shutting down",
        {}
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Application Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload-excel/",
            "query": "/query/",
            "logs": "/logs/"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_count": rag_service.index.ntotal if rag_service.index else 0,
        "documents_count": len(rag_service.documents)
    }


@app.post("/upload-excel/", response_model=UploadResponse)
async def upload_excel_file(
    file: UploadFile = File(...),
    api_key: str = Depends(security.validate_api_key)
):
    """
    Upload and process Excel file for RAG system
    
    Args:
        file: Excel file (.xlsx or .xls)
        api_key: API key for authentication
        
    Returns:
        UploadResponse: Upload processing results
        
    Raises:
        HTTPException: If file validation fails or processing error occurs
    """
    try:
        # Validate file type
        if not file.filename or not security.validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only .xlsx and .xls files are supported."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process the file
        result = rag_service.process_excel_file(file_content, file.filename)
        
        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=security.sanitize_filename(file.filename or "unknown_file"),
            records_processed=result["records_processed"],
            vector_count=result["vector_count"]
        )
        
    except ValueError as e:
        logging_service.log_error(
            "validation_error",
            str(e),
            {"filename": file.filename, "user": "api_user"}
        )
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging_service.log_error(
            "upload_processing_error",
            str(e),
            {"filename": file.filename, "user": "api_user"}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during file processing: {str(e)}"
        )


@app.post("/query/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    api_key: str = Depends(security.validate_api_key)
):
    """
    Process natural language query using RAG system
    
    Args:
        request: Query request containing the question
        api_key: API key for authentication
        
    Returns:
        QueryResponse: Generated answer with metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Query question cannot be empty"
            )
        
        # Check if data is loaded
        if rag_service.dataframe is None:
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload an Excel file first."
            )
        
        # Process the query
        response = rag_service.query(request.question)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging_service.log_error(
            "query_processing_error",
            str(e),
            {"query": request.question, "user": "api_user"}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during query processing: {str(e)}"
        )


@app.get("/logs/", response_model=LogResponse)
async def get_logs(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    log_type: str = Query("all", description="Type of logs: all, queries, uploads, errors"),
    api_key: str = Depends(security.validate_api_key)
):
    """
    Retrieve logs for monitoring and debugging
    
    Args:
        date: Date in YYYYMMDD format
        log_type: Type of logs to retrieve
        api_key: API key for authentication
        
    Returns:
        LogResponse: Log entries for the specified date and type
        
    Raises:
        HTTPException: If date format is invalid or retrieval fails
    """
    try:
        log_response = logging_service.get_logs_by_date(date, log_type)
        return log_response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging_service.log_error(
            "log_retrieval_error",
            str(e),
            {"date": date, "log_type": log_type, "user": "api_user"}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during log retrieval: {str(e)}"
        )


@app.get("/stats/")
async def get_application_stats(
    api_key: str = Depends(security.validate_api_key)
):
    """
    Get application statistics
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Dict: Application statistics
    """
    try:
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        
        # Get today's log stats
        log_stats = logging_service.get_log_statistics(today)
        
        # Get performance stats
        performance_stats = logging_service.get_query_performance_stats(today)
        
        # Get vector store stats
        vector_stats = {
            "total_vectors": rag_service.index.ntotal if rag_service.index else 0,
            "total_documents": len(rag_service.documents),
            "dataframe_loaded": rag_service.dataframe is not None,
            "dataframe_rows": len(rag_service.dataframe) if rag_service.dataframe is not None else 0
        }
        
        return {
            "date": today,
            "log_statistics": log_stats,
            "performance_statistics": performance_stats,
            "vector_store_statistics": vector_stats
        }
        
    except Exception as e:
        logging_service.log_error(
            "stats_retrieval_error",
            str(e),
            {"user": "api_user"}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during stats retrieval: {str(e)}"
        )


@app.exception_handler(401)
async def unauthorized_handler(request, exc):
    """Handle unauthorized access"""
    return JSONResponse(
        status_code=401,
        content=ErrorResponse(
            error="Unauthorized",
            detail="Invalid or missing API key. Please provide a valid X-API-Key header."
        ).dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    logging_service.log_error(
        "internal_server_error",
        str(exc),
        {"request_url": str(request.url), "method": request.method}
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An internal server error occurred. Please try again later."
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        
        reload=True,
        log_level="info"
    )