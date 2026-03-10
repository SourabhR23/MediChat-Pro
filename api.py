"""
MediChat Pro - FastAPI Backend
Serves all business logic as REST endpoints consumed by the Streamlit frontend.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import time
import logging
import re
from io import BytesIO

# ── App-level imports (same as original main.py) ──────────────────────────────
from app.pdf_utils import extract_text_from_pdf, clean_text
from app.vectorstore_utils import (
    create_faiss_index,
    retrieve_relevant_docs,
    clear_chroma_collection,
    ensure_collection_exists,
)
from app.s3_utils import (
    process_uploaded_files_with_s3,
    list_documents_in_s3,
    process_all_s3_documents_for_vector_storage,
)
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from app.email_utils import (
    send_medical_analytics,
    validate_email,
    send_support_ticket,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medichat_api")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediChat Pro API",
    description="FastAPI backend powering the MediChat Pro medical document assistant.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state (replace with Redis/DB in production) ─────────────────────
_state: Dict[str, Any] = {
    "vectorstore": None,
    "chat_model": None,
    "document_count": 0,
}

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    insights: Dict[str, Any]
    response_time: float

class EmailReportRequest(BaseModel):
    user_query: str
    ai_response: str
    document_insights: Dict[str, Any]
    receiver_email: str
    chat_history: List[Dict[str, Any]] = []

class SupportTicketRequest(BaseModel):
    user_query: str
    ai_response: str
    user_email: Optional[str] = None
    chat_history: List[Dict[str, Any]] = []

class SessionSaveRequest(BaseModel):
    receiver_email: str
    document_count: int
    message_count: int
    recent_messages: List[Dict[str, Any]] = []

class InitResponse(BaseModel):
    success: bool
    document_count: int
    message: str


# ── Patient medical history insight helpers ───────────────────────────────────

# Common name prefixes to help extract patient names from queries / chunks
_NAME_PREFIXES = re.compile(
    r"\b(?:patient|mr\.?|mrs\.?|ms\.?|dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)

# Medical record fields expected in a patient history PDF
_RECORD_FIELDS = [
    "name", "age", "diagnosis", "medication", "allergy", "allergies",
    "history", "dosage", "prescription", "symptom", "blood pressure",
    "heart rate", "temperature", "weight", "height", "gender",
]

# High-risk fields — must never be silently missing
_CRITICAL_FIELDS = ["allergy", "allergies", "medication", "diagnosis"]

# Query intent → keywords map
_INTENT_MAP: Dict[str, List[str]] = {
    "allergy":    ["allergy", "allergic", "allergies", "reaction"],
    "medication": ["medication", "drug", "prescribed", "dosage", "medicine", "tablet"],
    "history":    ["history", "previous", "past", "diagnosed", "background"],
    "vitals":     ["blood pressure", "heart rate", "temperature", "pulse", "weight"],
    "identity":   ["name", "age", "gender", "years old", "patient"],
}


def _extract_names_from_text(text: str) -> List[str]:
    """Extract likely patient names from a block of text."""
    matches = _NAME_PREFIXES.findall(text)
    # Also catch bare capitalised two-word names (e.g. "John Doe")
    bare = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
    return list({m.strip() for m in matches + bare})


def _detect_intent(query_lower: str) -> List[str]:
    return [
        intent for intent, terms in _INTENT_MAP.items()
        if any(t in query_lower for t in terms)
    ]


def generate_document_insights(
    user_query: str,
    ai_response: str,
    relevant_docs: List[Any],
    response_time: float,
    total_documents: int = 0,
) -> Dict[str, Any]:
    """
    Patient-history-aware RAG insight engine.

    Computes:
      1. Patient Match Score       — right patient's chunks retrieved?
      2. Field Coverage Score      — how many record fields present?
      3. Entity Faithfulness       — medical entities from chunks echoed in response?
      4. Critical Field Safety     — allergy / medication always surfaced?
      5. Cross-Patient Contamination Risk — other patients' data leaking in?
      6. Age / Name Consistency    — patient details match between chunk and response?
      7. Intent Coverage           — did the response address what was actually asked?
      8. Avg Chunk Relevance       — token-overlap relevance per chunk
      9. Confidence Score          — weighted composite of all above
    """
    query_lower    = user_query.lower()
    response_lower = ai_response.lower()
    num_docs       = len(relevant_docs)
    all_chunk_text = " ".join(d.page_content for d in relevant_docs)
    chunk_lower    = all_chunk_text.lower()

    # ── 1. Patient Match Score ────────────────────────────────────────────────
    query_names = _extract_names_from_text(user_query)
    if query_names:
        chunks_with_patient = sum(
            1 for doc in relevant_docs
            if any(n.lower() in doc.page_content.lower() for n in query_names)
        )
        patient_match_score = chunks_with_patient / max(num_docs, 1)
        patient_match_pct   = f"{patient_match_score:.0%}"
    else:
        patient_match_score = None   # query didn't name a patient
        patient_match_pct   = "N/A (no name in query)"

    # ── 2. Field Coverage Score ───────────────────────────────────────────────
    fields_found   = [f for f in _RECORD_FIELDS if f in chunk_lower]
    field_coverage = len(fields_found) / len(_RECORD_FIELDS)
    field_coverage_pct = f"{field_coverage:.0%}"

    # ── 3. Entity Faithfulness ────────────────────────────────────────────────
    # Extract multi-word medical entities (≥2 chars, capitalised or known terms)
    context_entities = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})?\b", all_chunk_text))
    response_entities = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})?\b", ai_response))
    if context_entities:
        faithfulness = len(context_entities & response_entities) / len(context_entities)
    else:
        faithfulness = 0.0
    faithfulness_pct = f"{faithfulness:.0%}"

    # ── 4. Critical Field Safety ──────────────────────────────────────────────
    # Check: if a critical field appears in context, does it also appear in response?
    critical_present_in_context  = [f for f in _CRITICAL_FIELDS if f in chunk_lower]
    critical_present_in_response = [f for f in critical_present_in_context if f in response_lower]
    if critical_present_in_context:
        safety_score = len(critical_present_in_response) / len(critical_present_in_context)
        safety_label = "✅ Safe" if safety_score == 1.0 else (
                       "⚠️ Partial" if safety_score >= 0.5 else "🚨 Risk")
        missing_critical = [
            f for f in critical_present_in_context
            if f not in critical_present_in_response
        ]
    else:
        safety_score    = 1.0
        safety_label    = "✅ Safe (not applicable)"
        missing_critical = []

    # ── 5. Cross-Patient Contamination Risk ───────────────────────────────────
    names_in_chunks = _extract_names_from_text(all_chunk_text)
    unique_patients = len(set(names_in_chunks))
    if unique_patients > 2:
        contamination_risk = "🚨 High"
    elif unique_patients == 2:
        contamination_risk = "⚠️ Medium"
    else:
        contamination_risk = "✅ Low"

    # ── 6. Age / Name Consistency ─────────────────────────────────────────────
    age_in_chunk    = re.search(r"\b(\d{1,3})\s*(?:years?|yr)", all_chunk_text, re.IGNORECASE)
    age_in_response = re.search(r"\b(\d{1,3})\s*(?:years?|yr)", ai_response,    re.IGNORECASE)
    if age_in_chunk and age_in_response:
        age_consistent = age_in_chunk.group(1) == age_in_response.group(1)
        age_consistency_label = "✅ Match" if age_consistent else "❌ Mismatch"
    else:
        age_consistent        = None
        age_consistency_label = "N/A"

    # ── 7. Intent Coverage ────────────────────────────────────────────────────
    detected_intents  = _detect_intent(query_lower)
    addressed_intents = [
        i for i in detected_intents
        if any(t in response_lower for t in _INTENT_MAP[i])
    ]
    intent_score = (
        len(addressed_intents) / len(detected_intents)
        if detected_intents else 1.0
    )
    intent_coverage_pct = f"{intent_score:.0%}"
    unaddressed_intents = [i for i in detected_intents if i not in addressed_intents]

    # ── 8. Avg Chunk Relevance ────────────────────────────────────────────────
    query_tokens = set(re.findall(r"\b\w{4,}\b", query_lower))
    chunk_scores = []
    for doc in relevant_docs:
        ct = doc.page_content.lower()
        if query_tokens:
            hits = sum(1 for tok in query_tokens if tok in ct)
            chunk_scores.append(hits / len(query_tokens))
        else:
            chunk_scores.append(0.0)
    avg_chunk_relevance     = (sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 0.0
    avg_chunk_relevance_pct = f"{avg_chunk_relevance:.0%}"

    # ── 9. Retrieval Coverage label ───────────────────────────────────────────
    if num_docs >= 7:
        retrieval_coverage = "Comprehensive"
    elif num_docs >= 3:
        retrieval_coverage = "Moderate"
    else:
        retrieval_coverage = "Limited"

    # ── 10. Composite Confidence Score ───────────────────────────────────────
    pm  = (patient_match_score if patient_match_score is not None else 0.5) * 25  # 25 pts
    fc  = field_coverage   * 20                                                    # 20 pts
    ef  = faithfulness     * 15                                                    # 15 pts
    ss  = safety_score     * 20                                                    # 20 pts
    ic  = intent_score     * 10                                                    # 10 pts
    acr = avg_chunk_relevance * 10                                                 # 10 pts
    confidence = pm + fc + ef + ss + ic + acr

    # ── Query complexity (kept for context) ──────────────────────────────────
    word_count = len(user_query.split())
    query_complexity = "Complex" if word_count > 15 else ("Medium" if word_count > 7 else "Simple")

    # ── Summary coverage string ───────────────────────────────────────────────
    doc_coverage = (
        f"Retrieved {num_docs} chunks — {retrieval_coverage.lower()} coverage. "
        f"Fields found: {', '.join(fields_found[:6]) or 'none'}."
    )

    logger.info(
        f"Insights | patient_match={patient_match_pct} field_cov={field_coverage_pct} "
        f"faithfulness={faithfulness_pct} safety={safety_label} "
        f"contamination={contamination_risk} intent={intent_coverage_pct} "
        f"confidence={confidence:.1f}%"
    )

    return {
        # Core metrics
        "total_documents":          total_documents,
        "total_chunks":             num_docs,
        "relevant_docs_count":      num_docs,
        "response_time":            f"{response_time:.2f}",
        "query_complexity":         query_complexity,
        "query_word_count":         word_count,

        # Patient-specific metrics
        "patient_match_score":      patient_match_pct,
        "field_coverage":           field_coverage_pct,
        "fields_found":             fields_found,
        "entity_faithfulness":      faithfulness_pct,
        "critical_field_safety":    safety_label,
        "missing_critical_fields":  missing_critical,
        "contamination_risk":       contamination_risk,
        "unique_patients_in_chunks": unique_patients,
        "age_name_consistency":     age_consistency_label,

        # Response quality
        "intent_coverage":          intent_coverage_pct,
        "detected_intents":         detected_intents,
        "unaddressed_intents":      unaddressed_intents,
        "avg_chunk_relevance":      avg_chunk_relevance_pct,
        "retrieval_coverage":       retrieval_coverage,

        # Composite
        "confidence_score":         f"{min(confidence, 100):.1f}%",
        "document_coverage":        doc_coverage,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_ready": _state["vectorstore"] is not None}


@app.post("/init", response_model=InitResponse)
def init_from_chromadb():
    """Auto-initialise from existing ChromaDB collection."""
    try:
        collection = ensure_collection_exists()
        if collection and collection.count() > 0:
            _state["vectorstore"] = collection
            _state["chat_model"] = get_chat_model(EURI_API_KEY)
            _state["document_count"] = collection.count()
            logger.info(f"Initialized with {collection.count()} docs from ChromaDB")
            return InitResponse(
                success=True,
                document_count=collection.count(),
                message=f"Initialized with {collection.count()} existing documents.",
            )
        return InitResponse(success=False, document_count=0, message="No documents in ChromaDB.")
    except Exception as e:
        logger.exception("Init failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload PDFs → extract text → store in S3 → embed into ChromaDB.
    Returns S3 results + chunk count.
    """
    try:
        # Wrap UploadFile objects so process_uploaded_files_with_s3 can .read() + .name
        class _FileLike:
            def __init__(self, upload: UploadFile, content: bytes):
                self.name = upload.filename
                self._content = content
            def read(self):
                return self._content

        wrapped = []
        for f in files:
            content = await f.read()
            wrapped.append(_FileLike(f, content))

        s3_results = process_uploaded_files_with_s3(wrapped, extract_text_from_pdf)

        all_texts = s3_results.get("all_texts", [])
        if not all_texts:
            raise HTTPException(status_code=422, detail="No text extracted from uploaded files.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for text in all_texts:
            chunks.extend(splitter.split_text(text))

        logger.info(f"Embedding {len(chunks)} chunks from {len(files)} files")
        vectorstore = create_faiss_index(chunks)
        _state["vectorstore"] = vectorstore
        _state["chat_model"] = get_chat_model(EURI_API_KEY)
        _state["document_count"] = len(files)

        return {
            "success": True,
            "files_processed": len(files),
            "chunks_created": len(chunks),
            "s3_uploaded": len(s3_results.get("uploaded_to_s3", [])),
            "s3_existing": len(s3_results.get("already_in_s3", [])),
            "s3_failed": len(s3_results.get("failed_uploads", [])),
            "details": s3_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/process-s3")
def process_s3_documents():
    """Download all S3 documents and rebuild ChromaDB index."""
    try:
        all_texts = process_all_s3_documents_for_vector_storage(extract_text_from_pdf)
        if not all_texts:
            return {"success": False, "message": "No documents found in S3."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for text in all_texts:
            chunks.extend(splitter.split_text(text))

        vectorstore = create_faiss_index(chunks)
        _state["vectorstore"] = vectorstore
        _state["chat_model"] = get_chat_model(EURI_API_KEY)
        _state["document_count"] = len(all_texts)

        return {
            "success": True,
            "documents_processed": len(all_texts),
            "chunks_created": len(chunks),
        }
    except Exception as e:
        logger.exception("S3 processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/s3-list")
def list_s3():
    """List all PDFs stored in the S3 bucket."""
    try:
        docs = list_documents_in_s3()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        logger.exception("S3 list failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/clear")
def clear_documents():
    """Wipe the ChromaDB collection."""
    try:
        success = clear_chroma_collection()
        if success:
            _state["vectorstore"] = None
            _state["chat_model"] = None
            _state["document_count"] = 0
        return {"success": success}
    except Exception as e:
        logger.exception("Clear failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Core RAG chat endpoint.
    Retrieves relevant chunks → builds prompt → calls LLM → returns response + insights.
    """
    if _state["vectorstore"] is None or _state["chat_model"] is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Upload documents or initialise from ChromaDB first.",
        )

    start = time.time()

    try:
        relevant_docs = retrieve_relevant_docs(req.prompt, k=10)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant specializing in patient medical history records.
Each document contains structured patient data: name, age, gender, diagnosis, medications, allergies, medical history, and vitals.

CRITICAL SAFETY RULES (never violate these):
1. ALLERGIES: If allergy information exists in the documents, ALWAYS include it prominently. If allergy data is NOT present in the retrieved context, explicitly state: "⚠️ Allergy information not found in retrieved records — do not assume none exist."
2. PATIENT IDENTITY: Only answer about the specific patient asked. Never mix information from different patients' records.
3. MEDICATIONS: Always include dosage and frequency if present. Never invent or assume medication details not in the documents.
4. MISSING DATA: If a specific field (e.g. blood pressure, diagnosis) is not in the retrieved chunks, say so explicitly — do not omit or imply it is normal.

RESPONSE RULES:
5. Structure your answer with clear field labels: **Name**, **Age**, **Diagnosis**, **Medications**, **Allergies**, **History**, **Vitals**
6. If multiple patients appear in context, clearly separate them and flag this as unusual
7. Always end with: "⚠️ This is for informational purposes only. Consult a qualified healthcare professional."
8. Keep responses factual and grounded strictly in the provided documents

Patient Records Context (Chunks Retrieved: {len(relevant_docs)}):
{context}

User Question: {req.prompt}

Provide a structured, accurate response based strictly on the above patient records:"""

        response_text = ask_chat_model(_state["chat_model"], system_prompt)
        response_time = time.time() - start

        insights = generate_document_insights(
            user_query=req.prompt,
            ai_response=response_text,
            relevant_docs=relevant_docs,
            response_time=response_time,
            total_documents=_state["document_count"],
        )

        logger.info(
            f"Chat | query_words={insights['query_word_count']} "
            f"chunks={insights['relevant_docs_count']} "
            f"confidence={insights['confidence_score']} "
            f"time={response_time:.2f}s"
        )

        return ChatResponse(
            response=response_text,
            insights=insights,
            response_time=response_time,
        )

    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Return current vectorstore / document state."""
    return {
        "vectorstore_ready": _state["vectorstore"] is not None,
        "document_count": _state["document_count"],
        "chat_model_ready": _state["chat_model"] is not None,
    }


@app.post("/email/send-report")
def send_report(req: EmailReportRequest):
    """Send a medical analytics email report."""
    if not validate_email(req.receiver_email):
        raise HTTPException(status_code=422, detail="Invalid email address.")
    try:
        success = send_medical_analytics(
            user_query=req.user_query,
            ai_response=req.ai_response,
            document_insights=req.document_insights,
            receiver_email=req.receiver_email,
            chat_history=req.chat_history,
        )
        return {"success": success}
    except Exception as e:
        logger.exception("Email report failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/support-ticket")
def create_ticket(req: SupportTicketRequest):
    """Create and email a support ticket."""
    try:
        success = send_support_ticket(
            user_query=req.user_query,
            ai_response=req.ai_response,
            user_email=req.user_email,
            chat_history=req.chat_history,
        )
        return {"success": success}
    except Exception as e:
        logger.exception("Support ticket failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/save-session")
def save_session(req: SessionSaveRequest):
    """Email a session-save summary."""
    if not validate_email(req.receiver_email):
        raise HTTPException(status_code=422, detail="Invalid email address.")
    try:
        insights = {
            "total_documents": req.document_count,
            "total_chunks": req.message_count,
            "confidence_score": "100%",
            "response_time": "0.00",
            "query_complexity": "Session Save",
            "document_coverage": (
                f"Session saved with {req.document_count} documents "
                f"and {req.message_count} messages."
            ),
            "medical_keywords": ["session", "save", "backup"],
        }
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        success = send_medical_analytics(
            user_query="Chat command: save session",
            ai_response=(
                f"Session saved at {ts}. "
                f"Documents: {req.document_count}, Messages: {req.message_count}"
            ),
            document_insights=insights,
            receiver_email=req.receiver_email,
            chat_history=req.recent_messages,
        )
        return {"success": success, "timestamp": ts}
    except Exception as e:
        logger.exception("Session save failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/test")
def test_email(receiver_email: str):
    """Send a test email to verify SMTP config."""
    if not validate_email(receiver_email):
        raise HTTPException(status_code=422, detail="Invalid email address.")
    test_insights = {
        "total_documents": 1,
        "total_chunks": 5,
        "relevant_docs_count": 2,
        "confidence_score": "95.0%",
        "response_time": "1.50",
        "query_complexity": "Test",
        "document_coverage": "Test email to verify configuration.",
        "medical_keywords": ["test", "email", "configuration"],
    }
    success = send_medical_analytics(
        user_query="Test email configuration",
        ai_response="This is a test email. If received, configuration is successful!",
        document_insights=test_insights,
        receiver_email=receiver_email,
        chat_history=[],
    )
    return {"success": success}