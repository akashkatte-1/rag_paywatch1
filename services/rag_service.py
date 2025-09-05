from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import faiss
import openai
import os
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from io import BytesIO

from models.schemas import CandidateData, QueryResponse
from services.currency_service import currency_service
from services.logging_service import logging_service


class RAGService:
    """RAG service with FAISS vector database and OpenAI integration"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o"
        self.embedding_dimension = 1536
        
        # FAISS index and data storage
        self.index = None
        self.documents = []
        self.dataframe = None
        self.metadata_list = []
        # Persistent storage paths
        self.storage_dir = Path("vector_storage")
        self.storage_dir.mkdir(exist_ok=True)
        self.index_path = self.storage_dir / "index.bin"
        self.documents_path = self.storage_dir / "documents.pkl"
        self.dataframe_path = self.storage_dir / "dataframe.pkl"
        
        # Load existing data if available
        self._load_persistent_data()
        
        # Tools for the agent
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Retrieve relevant documents from the vector store based on semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_ctc_values",
                    "description": "Get all CTC values from the dataset for calculations like average, sum, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_conditions": {
                                "type": "object",
                                "description": "Optional filters to apply (e.g., {'Location': 'Bangalore', 'Skills': 'Python'})"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_Exp_locations",
                    "description": "Get locations of candidates with specific Experience levels",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Experience": {
                                "type": "string",
                                "description": "Experience level to search for (e.g., '2y 6m', '3 years')"
                            }
                        },
                        "required": ["Experience"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "description": "Get real-time exchange rate between two currencies",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_currency": {
                                "type": "string",
                                "description": "Source currency code (default: INR)"
                            },
                            "to_currency": {
                                "type": "string",
                                "description": "Target currency code (default: USD)"
                            }
                        }
                    }
                }
            }
        ]
    
    def _load_persistent_data(self):
        """Load persistent FAISS index and data"""
        try:
            if self.index_path.exists() and self.documents_path.exists() and self.dataframe_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load documents
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load dataframe
                with open(self.dataframe_path, 'rb') as f:
                    self.dataframe = pickle.load(f)
                
                # Ensure CTC_USD column exists
                if self.dataframe is not None and 'CTC_USD' not in self.dataframe.columns:
                    rate = currency_service.get_exchange_rate("INR", "USD") or 0.012
                    self.dataframe['CTC_USD'] = self.dataframe['CTC'] * 100000 * rate

                logging_service.log_application_event(
                    "data_loaded",
                    "Successfully loaded persistent vector data",
                    {
                        "index_vectors": self.index.ntotal if self.index else 0,
                        "documents_count": len(self.documents),
                        "dataframe_rows": len(self.dataframe) if self.dataframe is not None else 0
                    }
                )
        except Exception as e:
            logging_service.log_error(
                "data_load_error",
                f"Failed to load persistent data: {str(e)}",
                {"error_type": type(e).__name__}
            )
            self._initialize_empty_storage()
    
    def _initialize_empty_storage(self):
        """Initialize empty storage structures"""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.documents = []
        self.dataframe = None
    
    def _save_persistent_data(self):
        """Save FAISS index and data to persistent storage"""
        try:
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save dataframe
            if self.dataframe is not None:
                with open(self.dataframe_path, 'wb') as f:
                    pickle.dump(self.dataframe, f)
            
            logging_service.log_application_event(
                "data_saved",
                "Successfully saved persistent vector data",
                {
                    "index_vectors": self.index.ntotal if self.index else 0,
                    "documents_count": len(self.documents)
                }
            )
        except Exception as e:
            logging_service.log_error(
                "data_save_error",
                f"Failed to save persistent data: {str(e)}",
                {"error_type": type(e).__name__}
            )
    
    def process_excel_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process Excel file and create vector embeddings
        """
        try:
            # Read Excel file
            df = pd.read_excel(BytesIO(file_content))
            metadata = self.extract_metadata_from_filename(filename)

            # Remove Name column immediately if present
            if 'Name' in df.columns:
                df = df.drop(columns=['Name'])

            # Clean and validate data
            df, invalid_records = self._clean_and_validate_dataframe(df)

            # Create documents for embedding
            documents = self._create_documents(df)

            # Generate embeddings
            embeddings = self._generate_embeddings(documents)

            # Ensure FAISS index is initialized
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dimension)

            # Update FAISS index
            self._update_vector_index(embeddings, documents)

            # Store dataframe
            self.dataframe = df

            # Save persistent data
            self._save_persistent_data()

            # Log upload and invalid records
            logging_service.log_file_upload(
                filename=filename,
                file_size=len(file_content),
                records_processed=len(df),
                vector_count=len(documents),
                success=True
            )
            if not invalid_records.empty:
                logging_service.log_application_event(
                    "invalid_records_excluded",
                    f"Excluded {len(invalid_records)} invalid records during preprocessing.",
                    {"invalid_count": len(invalid_records), "invalid_samples": invalid_records.head(3).to_dict('records')}
                )

            return {
                "records_processed": len(df),
                "vector_count": len(documents),
                "columns": list(df.columns),
                "sample_data": df.head(3).to_dict('records'),
                "invalid_records": invalid_records.head(3).to_dict('records')
            }

        except Exception as e:
            logging_service.log_file_upload(
                filename=filename,
                file_size=len(file_content),
                records_processed=0,
                vector_count=0,
                success=False,
                error_message=str(e)
            )
            raise e
    def _parse_experience(self, exp_str: str) -> float:
        """
        Parse experience string to decimal years.
        Supported formats: '4 yrs 3 months', '5 years', '3.5', '2 yrs 6 m', '4yrs'
        """
        exp_str = str(exp_str).lower().replace("years", "yrs").replace("year", "yr").replace("months", "m").replace("month", "m")
        match = re.match(r"(?P<years>\d+(\.\d+)?)\s*(yrs?|y)?(\s*(?P<months>\d+)\s*m)?", exp_str)
        if match:
            years = float(match.group("years"))
            months = float(match.group("months") or 0)
            return years + (months / 12)
        try:
            return float(exp_str)
        except Exception:
            return 0.0
    def _clean_and_validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and validate the dataframe:
        - Remove rows with missing critical data
        - Parse experience to decimal years
        - Validate CTC against experience
        - Convert CTC to USD
        - Exclude invalid records
        Returns: (valid_df, invalid_df)
        """
        # Standardize column names
        column_mapping = {
            'CTC (in INR LPA)': 'CTC',
            'Exp(Experience)': 'Exp',
            'Experience': 'Exp'
        }
        df = df.rename(columns=column_mapping)

        # Remove rows with missing critical data
        df = df.dropna(subset=['CTC', 'Skills', 'Exp', 'Location'])

        # Parse experience to decimal years
        df['Exp_years'] = df['Exp'].apply(self._parse_experience)

        # Convert CTC to numeric
        df['CTC'] = pd.to_numeric(df['CTC'], errors='coerce')
        df = df.dropna(subset=['CTC'])

        # Validation: CTC ≥ (2/3) × Experience
        df['valid'] = df.apply(lambda row: row['CTC'] >= (2/3) * row['Exp_years'], axis=1)
        invalid_records = df[~df['valid']].copy()
        valid_df = df[df['valid']].copy()

        # Convert CTC to USD using currency_service
        rate = currency_service.get_exchange_rate("INR", "USD") or 0.012
        valid_df['CTC_USD'] = valid_df['CTC'] * 100000 * rate  # Convert LPA to INR, then to USD

        # Clean strings
        string_columns = ['Location', 'Skills', 'Exp']
        for col in string_columns:
            if col in valid_df.columns:
                valid_df[col] = valid_df[col].astype(str).str.strip()

        return valid_df, invalid_records

    def _create_documents(self, df: pd.DataFrame) -> List[str]:
        """Create text documents from dataframe rows (without Name, CTC in USD)."""
        documents = []
        for _, row in df.iterrows():
            doc = f"""
            Location: {row['Location']}
            CTC: {row['CTC_USD']:.2f} USD per annum
            Skills: {row['Skills']}
            Experience: {row['Exp_years']:.2f} years

            Summary: Candidate in {row['Location']} with {row['Exp_years']:.2f} years experience, skills in {row['Skills']}, earns {row['CTC_USD']:.2f} USD/year.
            """.strip()
            documents.append(doc)
        return documents
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using OpenAI"""
        embeddings = []
        
        # Process in batches to handle API limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _update_vector_index(self, embeddings: np.ndarray, documents: List[str]):
        """Update FAISS index with new embeddings"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize new index if needed
        if self.index is None or self.index.ntotal == 0:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents = []
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents using semantic search"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        # Return relevant documents
        relevant_docs = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        return relevant_docs
    
    def get_all_ctc_values(self, filter_conditions: Optional[Dict] = None) -> List[float]:
        """Get all CTC values in USD with optional filtering."""
        if self.dataframe is None:
            return []
        df = self.dataframe.copy()
        if filter_conditions:
            for column, value in filter_conditions.items():
                if column in df.columns:
                    if isinstance(value, str):
                        df = df[df[column].str.contains(value, case=False, na=False)]
                    else:
                        df = df[df[column] == value]
        return df['CTC_USD'].tolist()
    
    def get_Exp_locations(self, Experience: str) -> List[str]:
        """Get locations of candidates with specific Experience"""
        if self.dataframe is None:
            return []
        
        # Normalize Experience format
        Exp_pattern = self._normalize_Experience(Experience)
        
        df = self.dataframe.copy()
        matching_rows = df[df['Exp'].str.contains(Exp_pattern, case=False, na=False)]
        
        return matching_rows['Location'].unique().tolist()
    
    def _normalize_Experience(self, Experience: str) -> str:
        """Normalize Experience string for matching"""
        # Convert various formats to searchable pattern
        Exp = Experience.lower().strip()
        
        # Extract numbers and units
        years_match = re.search(r'(\d+)\s*(?:y|year|yrs?)', Exp)
        months_match = re.search(r'(\d+)\s*(?:m|month|months?)', Exp)
        
        if years_match:
            years = years_match.group(1)
            if months_match:
                months = months_match.group(1)
                return f"{years}.*{months}"
            else:
                return f"{years}"
        
        return Exp
    
    def get_exchange_rate_tool(self, from_currency: str = "INR", to_currency: str = "USD") -> float:
        """Tool function for getting exchange rates"""
        rate = currency_service.get_exchange_rate(from_currency, to_currency)
        return rate if rate is not None else 0.012  # fallback rate
    def get_ctc_statistics(self, filter_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Returns a dictionary of statistics for the CTC column from the validated dataframe.
        Supports optional filtering.
        """
        if self.dataframe is None or self.dataframe.empty:
            return {}

        df = self.dataframe.copy()
        if filter_conditions:
            for column, value in filter_conditions.items():
                if column in df.columns:
                    if isinstance(value, str):
                        df = df[df[column].str.contains(value, case=False, na=False)]
                    else:
                        df = df[df[column] == value]

        ctc_series = df['CTC_USD'] if 'CTC_USD' in df.columns else df['CTC']
        stats = {
            'mean': round(ctc_series.mean(), 2),
            'mode': [round(m, 2) for m in ctc_series.mode().tolist()],
            'median': round(ctc_series.median(), 2),
            'highest': round(ctc_series.max(), 2),
            'lowest': round(ctc_series.min(), 2),
            'average': round(ctc_series.mean(), 2),  # Same as mean
            'count': int(ctc_series.count()),
            'sum': round(ctc_series.sum(), 2)
        }
        return stats
    def _execute_function_call(self, function_name: str, arguments: Dict) -> Any:
        """Execute function calls from the LLM"""
        if function_name == "retrieve_documents":
            query = arguments.get("query", "")
            k = arguments.get("k", 2900)
            return self.retrieve_documents(query, k)
        
        elif function_name == "get_all_ctc_values":
            filter_conditions = arguments.get("filter_conditions")
            return self.get_all_ctc_values(filter_conditions)
        
        elif function_name == "get_Exp_locations":
            Experience = arguments.get("Experience", "")
            return self.get_Exp_locations(Experience)
        
        elif function_name == "get_exchange_rate":
            from_currency = arguments.get("from_currency", "INR")
            to_currency = arguments.get("to_currency", "USD")
            return self.get_exchange_rate_tool(from_currency, to_currency)
        
        return None
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

    def query(self, question: str) -> QueryResponse:    
            """Process natural language query using RAG"""
            start_time = time.time()
            
            try:
                # System prompt with instructions
                system_prompt = """You are an Expert data analyst assistant 
                IMPORTANT INSTRUCTIONS:
                1. All CTC values in the source data are in Indian Rupees (INR) - LPA (Lakhs Per Annum)
                2. For ANY monetary value queries, you MUST:
                - First get the INR values using your tools
                - Then use get_exchange_rate to convert from INR to USD
                - Present the final answer ONLY in USD
                3. Use semantic search to find relevant candidates
                4. Use data aggregation tools for calculations
                5. Be precise with numbers and calculations
                6. Provide clear, actionable insights
                7. Unless the user explicitly asks for "top N" or "bottom N", always use all eligible data for calculations (e.g., average, min, max, median,highest,lowest, etc.).
                Available tools:
                - retrieve_documents
                - get_all_ctc_values
                - get_Exp_locations
                - get_exchange_rate: Convert currency (INR to USD)
                """
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Call OpenAI with function calling
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                message = response.choices[0].message
                sources_used = 0
                
                # Handle function calls
                while message.tool_calls:
                    messages.append(message)
                    
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = eval(tool_call.function.arguments)
                        
                        # Execute function
                        function_result = self._execute_function_call(function_name, function_args)
                        
                        if function_name == "retrieve_documents":
                            sources_used = len(function_result) if function_result else 0
                        
                        # Add function result to messages
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_result)
                        })
                    
                    # Get next response
                    response = self.openai_client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.1
                    )
                    message = response.choices[0].message
                
                answer = message.content
                processing_time = time.time() - start_time
                
                # Log the query
                logging_service.log_query_processing(
                    query=question,
                    response=answer,
                    processing_time=processing_time,
                    sources_used=sources_used
                )
                
                return QueryResponse(
                    answer=answer,
                    processing_time=processing_time,
                    sources_used=sources_used
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"Error processing query: {str(e)}"
                
                logging_service.log_error(
                    "query_processing_error",
                    error_msg,
                    {"query": question, "processing_time": processing_time}
                )
                
                return QueryResponse(
                    answer=error_msg,
                    processing_time=processing_time,
                    sources_used=0
                )


# Global RAG service instance
rag_service = RAGService()