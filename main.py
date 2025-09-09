from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional

from models.schemas import QueryRequest, QueryResponse, UploadResponse, LogResponse, ErrorResponse
from services.rag_service import rag_service
from services.logging_service import logging_service
from security.security_utils import security

# ------------------------
# Initialize FastAPI App
# ------------------------
app = FastAPI(
    title="RAG Application Backend",
    description="A comprehensive RAG system for HR data analysis with ChromaDB vector database",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Startup and Shutdown Events
# ------------------------
@app.on_event("startup")
async def startup_event():
    logging_service.log_application_event(
        "app_startup",
        "RAG Application started successfully",
        {"api_key": security.get_api_key()[:10] + "..."}
    )
    print(f"🚀 RAG Application started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    logging_service.log_application_event(
        "app_shutdown",
        "RAG Application shutting down",
        {}
    )

# ------------------------
# Root and Health
# ------------------------
@app.get("/")
async def root():
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
    return {
        "status": "healthy",
        "vectors_stored": rag_service.collection.count() if rag_service.collection else 0,
        "documents_count": len(rag_service.dataframe) if rag_service.dataframe is not None else 0
    }

# ------------------------
# Upload Excel Endpoint
# ------------------------
@app.post("/upload-excel/", response_model=UploadResponse)
async def upload_excel_file(file: UploadFile = File(...), api_key: str = Depends(security.validate_api_key)):
    try:
        if not file.filename or not security.validate_file_type(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format. Only .xlsx and .xls files are supported.")

        file_content = await file.read()
        result = rag_service.process_excel_file(file_content, file.filename)

        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=security.sanitize_filename(file.filename or "unknown_file"),
            records_processed=result["records_processed"],
            vectors_stored=result["vectors_stored"]  # ✅ updated
        )

    except Exception as e:
        logging_service.log_error("upload_processing_error", str(e), {"filename": file.filename})
        raise HTTPException(status_code=500, detail=f"Internal server error during file processing: {str(e)}")

# ------------------------
# Query Endpoint
# ------------------------
@app.post("/query/", response_model=QueryResponse)
async def process_query(request: QueryRequest, api_key: str = Depends(security.validate_api_key)):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Query question cannot be empty")

        if rag_service.dataframe is None:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload an Excel file first.")

        response = rag_service.query(request.question)
        return response

    except Exception as e:
        logging_service.log_error("query_processing_error", str(e), {"query": request.question})
        raise HTTPException(status_code=500, detail=f"Internal server error during query processing: {str(e)}")

# ------------------------
# Logs Endpoint
# ------------------------
@app.get("/logs/", response_model=LogResponse)
async def get_logs(date: str = Query(..., description="Date in YYYYMMDD format"),
                   log_type: str = Query("all", description="Type of logs: all, queries, uploads, errors"),
                   api_key: str = Depends(security.validate_api_key)):
    try:
        return logging_service.get_logs_by_date(date, log_type)
    except Exception as e:
        logging_service.log_error("log_retrieval_error", str(e), {"date": date, "log_type": log_type})
        raise HTTPException(status_code=500, detail=f"Internal server error during log retrieval: {str(e)}")

# ------------------------
# Exception Handlers
# ------------------------
@app.exception_handler(401)
async def unauthorized_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content=ErrorResponse(
            error="Unauthorized",
            detail="Invalid or missing API key. Please provide a valid X-API-Key header."
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logging_service.log_error("internal_server_error", str(exc), {"request_url": str(request.url)})
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
