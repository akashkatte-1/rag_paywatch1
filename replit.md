# Overview

This is a comprehensive RAG (Retrieval-Augmented Generation) application backend built with FastAPI for HR data analysis. The system specializes in processing Excel files containing candidate data (names, experience, locations, CTC/salary information) and providing conversational AI capabilities for querying this data. It features semantic search using FAISS vector database, automated currency conversion from INR to USD, and comprehensive logging for monitoring and analytics.

**Status**: Fully implemented and running on port 5000. All core functionality is operational including file upload, vector search, natural language querying, and real-time currency conversion.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
**FastAPI with Python**: Chosen for its high performance, automatic API documentation, and excellent async support. The application uses a modular service-oriented architecture with clear separation of concerns across models, services, and security layers.

## Vector Database and Search
**FAISS (Facebook AI Similarity Search)**: Implemented for efficient semantic similarity search on candidate data. Documents are embedded using OpenAI's text-embedding-3-small model and stored in a persistent FAISS index. This enables natural language querying of HR data with semantic understanding rather than simple keyword matching.

## AI and Language Model Integration
**OpenAI GPT-4 with Function Calling**: The system uses GPT-4 as the core conversational agent with a sophisticated tool-calling architecture. The agent has access to specialized tools for:
- Semantic document retrieval from vector store
- CTC value aggregation and analysis
- Experience-based location filtering
- Real-time currency conversion

## Data Processing Pipeline
**Pandas-based Excel Processing**: Excel files (.xlsx/.xls) are processed using pandas with validation rules ensuring data quality (CTC ≥ (2/3) × Experience in years). The system converts various experience formats to decimal years and validates salary data against experience thresholds.

## Security Architecture
**API Key Authentication**: All endpoints are protected using X-API-Key header authentication. The security system generates default development keys and validates all incoming requests through FastAPI dependency injection.

## Logging and Monitoring
**Comprehensive Event Logging**: Multi-tiered logging system tracking:
- Application lifecycle events (startup/shutdown)
- Query processing with performance metrics
- File upload operations with metadata
- Error tracking with detailed context
- Logs are stored in JSON format with date-based organization

## Data Persistence
**Multi-layer Storage Strategy**:
- FAISS binary index for vector embeddings
- Pickle serialization for document metadata
- Original DataFrame persistence for direct data operations
- Cache-based storage for exchange rates (30-minute TTL)

## Currency Conversion Service
**Real-time Exchange Rate Integration**: Automated currency conversion system that fetches live exchange rates from external APIs. All CTC values are assumed to be in INR and converted to USD for user queries, with intelligent caching to minimize API calls.

# External Dependencies

## AI Services
- **OpenAI API**: GPT-4 for language generation and text-embedding-3-small for document embeddings
- **FAISS**: Facebook's similarity search library for vector operations

## Currency APIs
- **Primary**: exchangerate-api.com for real-time exchange rates
- **Fallback**: open.er-api.com as backup currency service

## Python Libraries
- **FastAPI**: Web framework for API development
- **Pandas**: Data manipulation and Excel file processing
- **NumPy**: Numerical computations for vector operations
- **Pydantic**: Data validation and serialization
- **Requests**: HTTP client for external API calls

## Development Tools
- **CORS Middleware**: Configured for cross-origin requests
- **Uvicorn**: ASGI server for running the FastAPI application
- **Python Pickle**: Serialization for persistent data storage

## Infrastructure Requirements
- **File System**: Local storage for vector indices, documents, and logs
- **Environment Variables**: API key management for OpenAI and optional custom API keys
- **JSON Logging**: Structured logging with timestamp-based file organization