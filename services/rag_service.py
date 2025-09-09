from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import pickle
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import openai
import chromadb
from chromadb.config import Settings

from models.schemas import QueryResponse
from services.currency_service import currency_service
from services.logging_service import logging_service


class RAGService:
    """RAG service with **ChromaDB** vector database and OpenAI integration."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o"

        # Persistent storage
        self.storage_dir = Path("vector_storage")
        self.storage_dir.mkdir(exist_ok=True)
        self.dataframe_path = self.storage_dir / "dataframe.pkl"

        # ChromaDB
        self.chroma_path = str(self.storage_dir / "chroma_db")
        self.chroma = chromadb.PersistentClient(path=self.chroma_path, settings=Settings(allow_reset=False))
        self.collection = self._get_or_create_collection("candidates")

        self.dataframe: Optional[pd.DataFrame] = None
        self._load_persistent_data()

        # Tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_top_k",
                    "description": "Return top/bottom K rows by CTC_USD with optional filters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "k": {"type": "integer"},
                            "order": {"type": "string", "enum": ["desc", "asc"]},
                            "filter_conditions": {"type": "object"},
                        },
                        "required": ["k", "order"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ctc_statistics",
                    "description": "Compute statistics over CTC_USD with optional filters",
                    "parameters": {"type": "object", "properties": {"filter_conditions": {"type": "object"}}},
                },
            },
        ]

    # ---------------- Core setup ---------------- #
    def _get_or_create_collection(self, name: str):
        try:
            return self.chroma.get_collection(name)
        except Exception:
            return self.chroma.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _load_persistent_data(self):
        try:
            if self.dataframe_path.exists():
                with open(self.dataframe_path, "rb") as f:
                    self.dataframe = pickle.load(f)
            else:
                self.dataframe = None
        except Exception as e:
            logging_service.log_error("data_load_error", str(e), {})
            self.dataframe = None

    def _save_persistent_data(self):
        try:
            if self.dataframe is not None:
                with open(self.dataframe_path, "wb") as f:
                    pickle.dump(self.dataframe, f)
        except Exception as e:
            logging_service.log_error("data_save_error", str(e), {})

    # ---------------- File ingestion ---------------- #
    def process_excel_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an Excel file: clean, convert CTC, generate embeddings, and store in ChromaDB.
        Supports multiple files without overwriting previous data.
        Returns records processed and vectors stored.
        """
        # ---------------- Load Excel ---------------- #
        df = pd.read_excel(BytesIO(file_content))

        # Drop Name column if present
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        # Clean & validate dataframe
        valid_df, _ = self._clean_and_validate_dataframe(df)
        valid_df = valid_df.reset_index(drop=False).rename(columns={"index": "row_index"})
        valid_df["source_file"] = filename

        # Append to existing dataframe
        if self.dataframe is None or self.dataframe.empty:
            self.dataframe = valid_df.copy()
        else:
            self.dataframe = pd.concat([self.dataframe, valid_df], ignore_index=True)

        self._save_persistent_data()

        # ---------------- Generate batch embeddings ---------------- #
        texts = [
            str(row["Skills"]) + " " + str(row["CTC"]) + " " + str(row["Exp_years"])
            for _, row in valid_df.iterrows()
        ]
        embeddings_response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [d["embedding"] for d in embeddings_response["data"]]

        # ---------------- Prepare IDs and metadata ---------------- #
        ids = [f"{filename}_{row['row_index']}" for _, row in valid_df.iterrows()]
        metadatas = [row.to_dict() for _, row in valid_df.iterrows()]
        documents = [str(row.to_dict()) for _, row in valid_df.iterrows()]

        # ---------------- Add to ChromaDB ---------------- #
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
        except Exception as e:
            logging_service.log_error("chroma_add_error", str(e), {"file": filename})
            raise Exception(f"Error adding embeddings to ChromaDB: {e}")

        return {
            "records_processed": len(valid_df),
            "vectors_stored": len(valid_df)
        }


    def _parse_experience(self, exp_str: str) -> float:
        exp_str = str(exp_str).lower().replace("years", "yrs").replace("year", "yr").replace("months", "m")
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
        column_mapping = {"CTC (in INR LPA)": "CTC", "Exp(Experience)": "Exp", "Experience": "Exp"}
        df = df.rename(columns=column_mapping)
        df = df.dropna(subset=["CTC", "Skills", "Exp", "Location"])
        df["Exp_years"] = df["Exp"].apply(self._parse_experience)
        df["CTC"] = pd.to_numeric(df["CTC"], errors="coerce")
        df = df.dropna(subset=["CTC"])
        rate = currency_service.get_exchange_rate("INR", "USD") or 0.012
        df["CTC_USD"] = df["CTC"] * 100000 * rate
        return df, pd.DataFrame()

    # ---------------- Query helpers ---------------- #
    def get_ctc_statistics(self, filter_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        if self.dataframe is None or self.dataframe.empty:
            return {}
        df = self.dataframe.copy()
        if filter_conditions:
            for col, val in filter_conditions.items():
                df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
        if df.empty:
            return {}
        ctc = df["CTC_USD"]
        return {
            "highest": round(float(ctc.max()), 2),
            "lowest": round(float(ctc.min()), 2),
            "average": round(float(ctc.mean()), 2),
            "median": round(float(ctc.median()), 2),
        }

    def get_top_k(self, k: int, order: str = "desc", filter_conditions: Optional[Dict] = None):
        if self.dataframe is None or self.dataframe.empty:
            return []
        df = self.dataframe.copy()
        if filter_conditions:
            for col, val in filter_conditions.items():
                df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
        if df.empty:
            return []
        ascending = order == "asc"
        df_sorted = df.sort_values(by="CTC_USD", ascending=ascending).head(k)
        return df_sorted[["CTC_USD", "source_file"]].to_dict("records")

    # ---------------- Main query ---------------- #
    def _execute_function_call(self, fn: str, args: Dict) -> Any:
        if fn == "get_top_k":
            return self.get_top_k(int(args.get("k", 5)), args.get("order", "desc"), args.get("filter_conditions"))
        elif fn == "get_ctc_statistics":
            return self.get_ctc_statistics(args.get("filter_conditions"))
        return None

    def query(self, question: str) -> QueryResponse:
        start = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": question}],
                tools=self.tools,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if message.tool_calls:
                all_results, source_files = [], set()
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = eval(tool_call.function.arguments)
                    fn_result = self._execute_function_call(fn_name, fn_args)
                    all_results.append(fn_result)
                    if isinstance(fn_result, list):
                        for row in fn_result:
                            if "source_file" in row:
                                source_files.add(row["source_file"])

                # Summarize into natural answer
                summary_prompt = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes database query results."},
                    {"role": "user", "content": f"Question: {question}\nResults: {all_results}\nSummarize briefly."},
                ]
                summary = self.openai_client.chat.completions.create(model=self.llm_model, messages=summary_prompt)
                final_answer = summary.choices[0].message.content

                sources_used = len(all_results)
            else:
                final_answer, sources_used, source_files = message.content or "No answer generated.", 0, set()

            return QueryResponse(
                answer=final_answer,
                processing_time=round(time.time() - start, 2),
                sources_used=sources_used,
                source_files=list(source_files),
            )
        except Exception as e:
            raise Exception(f"Error processing query: {e}")


# Singleton
rag_service = RAGService()
