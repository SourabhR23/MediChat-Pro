"""
MediChat Pro v2 — Streamlit Frontend
Consumes the FastAPI backend (api.py) via HTTP.
Run backend first:  uvicorn api:app --reload --port 8000
Run frontend:       streamlit run main2.py
"""

import streamlit as st
import requests
import time
import re
import os
from typing import List, Dict, Any

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat Pro — Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (mirrors + upgrades original main.py style) ───────────────────
st.markdown("""
<style>
    /* ── fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── header ── */
    .hero {
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #e63946;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero p { font-size: 1.05rem; color: #6b7280; margin: 0; }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] { background: #0f1117 !important; color: #f1f5f9; }
    section[data-testid="stSidebar"] h3 { color: #e63946; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stTextInput input { background: #1e293b; color: #f1f5f9; border-color: #334155; }
    section[data-testid="stSidebar"] h3 {
        color: #e63946;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }

    /* ── buttons ── */
    .stButton > button {
        background: #e63946 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
        transition: background 0.2s ease;
    }
    .stButton > button:hover { background: #c1121f !important; }
    .stButton > button:disabled { background: #d1d5db !important; color: #9ca3af !important; }

    /* ── chat messages ── */
    .chat-wrap { margin-bottom: 1rem; }
    .chat-user {
        background: #1e293b;
        color: #f1f5f9;
        padding: 0.9rem 1.1rem;
        border-radius: 14px 14px 4px 14px;
        max-width: 78%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-assistant {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        padding: 0.9rem 1.1rem;
        border-radius: 14px 14px 14px 4px;
        max-width: 86%;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-ts {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 0.3rem;
        text-align: right;
    }

    /* ── insight card ── */
    .insight-card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }
    .insight-card h4 {
        color: #e63946;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.7rem;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 0.5rem;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .insight-item {
        background: #f9fafb;
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
    }
    .insight-label { font-size: 0.68rem; color: #6b7280; font-weight: 500; margin-bottom: 2px; }
    .insight-value { font-size: 0.9rem; font-weight: 700; color: #111827; }

    /* ── status badges ── */
    .badge-green {
        display: inline-block;
        background: #16a34a; color: #ffffff;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-red {
        display: inline-block;
        background: #fee2e2; color: #991b1b;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-yellow {
        display: inline-block;
        background: #fef3c7; color: #92400e;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600;
    }

    /* ── dividers ── */
    hr { border: none; border-top: 1px solid #f0f0f0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "vectorstore_ready": False,
        "document_count": 0,
        "receiver_email": "",
        "last_insights": {},
        "api_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend. Is `uvicorn api:app` running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def _post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend. Is `uvicorn api:app` running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def _delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def _validate_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def _sync_status():
    """Pull current backend state into session_state."""
    data = _get("/status")
    if data:
        st.session_state.vectorstore_ready = data.get("vectorstore_ready", False)
        st.session_state.document_count = data.get("document_count", 0)
        st.session_state.api_ok = True


# ── Insight card renderer (patient-history-aware, native Streamlit) ───────────

def _render_insights(ins: Dict[str, Any], key_suffix: str = ""):
    """
    Renders patient-history RAG insights using only native Streamlit components
    inside an st.expander — avoids all unsafe_allow_html rendering issues.
    """
    if not ins:
        return

    with st.expander("📊 RAG Quality Insights — Patient History", expanded=False):

        # ── Section 1: Retrieval Quality ──────────────────────────────────────
        st.markdown("**🔍 Retrieval Quality**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence Score",    ins.get("confidence_score", "—"))
        c2.metric("Patient Match Score", ins.get("patient_match_score", "—"))
        c3.metric("Field Coverage",      ins.get("field_coverage", "—"))

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Chunk Relevance", ins.get("avg_chunk_relevance", "—"))
        c5.metric("Chunks Retrieved",    ins.get("relevant_docs_count", "—"))
        c6.metric("Retrieval Coverage",  ins.get("retrieval_coverage", "—"))

        st.divider()

        # ── Section 2: Patient Safety Checks ─────────────────────────────────
        st.markdown("**🛡️ Patient Safety Checks**")

        safety  = ins.get("critical_field_safety", "N/A")
        cont    = ins.get("contamination_risk", "N/A")
        age_con = ins.get("age_name_consistency", "N/A")

        s1, s2, s3 = st.columns(3)

        # Critical Field Safety — coloured indicator
        with s1:
            colour = "🟢" if "Safe" in safety else ("🟡" if "Partial" in safety else "🔴")
            st.metric("Critical Field Safety", f"{colour} {safety}")

        # Contamination Risk
        with s2:
            colour = "🟢" if "Low" in cont else ("🟡" if "Medium" in cont else "🔴")
            st.metric("Contamination Risk", f"{colour} {cont}")

        # Age / Name Consistency
        with s3:
            colour = "🟢" if "Match" in age_con else ("⚪" if "N/A" in age_con else "🔴")
            st.metric("Age/Name Consistency", f"{colour} {age_con}")

        # Missing critical fields warning
        missing = ins.get("missing_critical_fields", [])
        if missing:
            st.warning(f"🚨 Missing critical fields in response: **{', '.join(missing)}**")
        else:
            st.success("✅ All critical fields (allergy, medication, diagnosis) accounted for.")

        # Contamination detail
        unique_pts = ins.get("unique_patients_in_chunks", 0)
        if unique_pts > 2:
            st.error(f"🚨 {unique_pts} different patient names detected in retrieved chunks — possible data mixing.")
        elif unique_pts == 2:
            st.warning(f"⚠️ 2 patient names detected in chunks — verify correct patient.")

        st.divider()

        # ── Section 3: Response Quality ───────────────────────────────────────
        st.markdown("**🎯 Response Quality**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Entity Faithfulness", ins.get("entity_faithfulness", "—"))
        r2.metric("Intent Coverage",     ins.get("intent_coverage", "—"))
        r3.metric("Response Time",       f"{ins.get('response_time', '—')}s")

        # Intent detail
        intents     = ins.get("detected_intents", [])
        unaddressed = ins.get("unaddressed_intents", [])
        if intents:
            st.caption(f"**Query intents detected:** {', '.join(intents)}")
        if unaddressed:
            st.warning(f"⚠️ Unaddressed intents: **{', '.join(unaddressed)}**")

        st.divider()

        # ── Section 4: Detail summary ─────────────────────────────────────────
        st.markdown("**📋 Detail Summary**")
        fields = ins.get("fields_found", [])
        st.caption(f"**Fields found in context:** {', '.join(fields) if fields else 'None detected'}")
        st.caption(f"**Coverage note:** {ins.get('document_coverage', '')}")


# ── Auto-init on first load ───────────────────────────────────────────────────
if not st.session_state.api_ok:
    health = _get("/health")
    if health:
        st.session_state.api_ok = True
        if health.get("vectorstore_ready"):
            _sync_status()
        else:
            # Try auto-init from ChromaDB
            result = _post("/init")
            if result and result.get("success"):
                st.session_state.vectorstore_ready = True
                st.session_state.document_count = result.get("document_count", 0)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Backend status ────────────────────────────────────────────────────────
    st.markdown("### 🔌 Backend")
    if st.session_state.api_ok:
        st.markdown('<span class="badge-green">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-red">● API Offline</span>', unsafe_allow_html=True)

    if st.session_state.vectorstore_ready:
        st.markdown(
            f'<span class="badge-green">● Vector DB Ready ({st.session_state.document_count} docs)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge-yellow">● No Documents Loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Document upload ───────────────────────────────────────────────────────
    st.markdown("### 📄 Document Upload")
    st.caption("Upload one or more medical PDF files.")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Uploading & embedding documents…"):
                files_payload = [
                    ("files", (f.name, f.read(), "application/pdf"))
                    for f in uploaded_files
                ]
                try:
                    r = requests.post(
                        f"{API_BASE}/documents/upload",
                        files=files_payload,
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.success(
                        f"✅ {data['files_processed']} file(s) processed — "
                        f"{data['chunks_created']} chunks created."
                    )
                    if data.get("s3_uploaded"):
                        st.info(f"☁️ {data['s3_uploaded']} file(s) uploaded to S3.")
                    if data.get("s3_existing"):
                        st.info(f"📂 {data['s3_existing']} file(s) already in S3.")
                    if data.get("s3_failed"):
                        st.warning(f"⚠️ {data['s3_failed']} S3 upload(s) failed.")
                    _sync_status()
                    st.balloons()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")

    # ── Manual init ───────────────────────────────────────────────────────────
    st.markdown("### 🔄 ChromaDB")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Init from DB"):
            with st.spinner("Initializing…"):
                result = _post("/init")
                if result and result.get("success"):
                    st.success(f"✅ {result['document_count']} docs loaded.")
                    _sync_status()
                elif result:
                    st.error(result.get("message", "No documents."))

    with col_b:
        if st.button("🗑 Clear DB"):
            with st.spinner("Clearing…"):
                result = _delete("/documents/clear")
                if result and result.get("success"):
                    st.success("Cleared!")
                    st.session_state.vectorstore_ready = False
                    st.session_state.document_count = 0

    st.markdown("---")

    # ── S3 management ─────────────────────────────────────────────────────────
    st.markdown("### ☁️ S3 Documents")
    if st.button("📋 List S3 Files"):
        with st.spinner("Fetching…"):
            data = _get("/documents/s3-list")
            if data:
                docs = data.get("documents", [])
                if docs:
                    st.success(f"{len(docs)} file(s) in S3")
                    for d in docs:
                        with st.expander(f"📄 {d['filename']}"):
                            st.write(f"**Key:** `{d['key']}`")
                            st.write(f"**Size:** {d['size']} bytes")
                            st.write(f"**Modified:** {d['last_modified']}")
                else:
                    st.info("No files in S3.")

    if st.button("🔄 Process All S3"):
        with st.spinner("Downloading & embedding…"):
            result = _post("/documents/process-s3")
            if result and result.get("success"):
                st.success(
                    f"✅ {result['documents_processed']} docs → {result['chunks_created']} chunks"
                )
                _sync_status()
            elif result:
                st.warning(result.get("message", "Nothing processed."))

    st.markdown("---")

    # ── Email configuration ───────────────────────────────────────────────────
    st.markdown("### 📧 Email Configuration")
    email_input = st.text_input(
        "Recipient Email",
        value=st.session_state.receiver_email,
        placeholder="user@example.com",
    )
    if email_input:
        st.session_state.receiver_email = email_input
        if _validate_email(email_input):
            st.markdown('<span class="badge-green">✓ Valid email</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-red">✗ Invalid format</span>', unsafe_allow_html=True)
    else:
        st.caption("Enter an email to enable reports.")

    if st.session_state.receiver_email and _validate_email(st.session_state.receiver_email):
        if st.button("📨 Send Test Email"):
            with st.spinner("Sending…"):
                result = _post(
                    f"/email/test",
                    params={"receiver_email": st.session_state.receiver_email},
                )
                if result and result.get("success"):
                    st.success("Test email sent!")
                else:
                    st.error("Failed to send test email.")

    st.markdown("---")

    # ── Quick actions ─────────────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Actions")
    qa1, qa2 = st.columns(2)

    with qa1:
        email_valid = bool(
            st.session_state.receiver_email
            and _validate_email(st.session_state.receiver_email)
        )
        if st.button(
            "📊 Send Report",
            disabled=not email_valid,
            help="Send full session analysis to email",
        ):
            with st.spinner("Generating report…"):
                result = _post(
                    "/email/send-report",
                    json={
                        "user_query": "Session analysis report",
                        "ai_response": "Complete session analysis.",
                        "document_insights": st.session_state.get("last_insights", {}),
                        "receiver_email": st.session_state.receiver_email,
                        "chat_history": st.session_state.messages[-10:],
                    },
                )
                if result and result.get("success"):
                    st.success(f"Report sent to {st.session_state.receiver_email}!")
                else:
                    st.error("Failed to send report.")

    with qa2:
        if st.button("🎫 Support Ticket"):
            with st.spinner("Creating ticket…"):
                result = _post(
                    "/email/support-ticket",
                    json={
                        "user_query": "Support request via sidebar",
                        "ai_response": "User requested support.",
                        "user_email": st.session_state.receiver_email or None,
                        "chat_history": st.session_state.messages[-5:],
                    },
                )
                if result and result.get("success"):
                    st.success("Ticket created!")
                else:
                    st.error("Failed to create ticket.")

    if st.button("💾 Save Session", disabled=not email_valid):
        with st.spinner("Saving…"):
            result = _post(
                "/email/save-session",
                json={
                    "receiver_email": st.session_state.receiver_email,
                    "document_count": st.session_state.document_count,
                    "message_count": len(st.session_state.messages),
                    "recent_messages": st.session_state.messages[-5:],
                },
            )
            if result and result.get("success"):
                st.success(f"Session saved at {result.get('timestamp', '')}!")
            else:
                st.error("Failed to save session.")

    st.markdown("---")

    # ── Session analytics ─────────────────────────────────────────────────────
    st.markdown("### 📈 Session Analytics")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Docs", st.session_state.document_count)
    mc2.metric("Messages", len(st.session_state.messages))
    mc3.metric("DB", "✅" if st.session_state.vectorstore_ready else "❌")

    if st.session_state.messages:
        st.caption("Recent messages:")
        for msg in st.session_state.messages[-3:]:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            preview = msg["content"][:80] + "…" if len(msg["content"]) > 80 else msg["content"]
            st.caption(f"{role_icon} {preview}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>🏥 MediChat Pro</h1>
    <p>Your Intelligent Medical Document Assistant — Powered by FastAPI + ChromaDB + Euri AI</p>
</div>
""", unsafe_allow_html=True)


# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    ts = msg.get("timestamp", "")
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
            st.caption(ts)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            st.caption(ts)
            # Show insights if stored
            if msg.get("insights"):
                _render_insights(msg["insights"])


# ── Helper: append both sides of a command exchange to history ───────────────
def _add_to_history(user_msg: str, assistant_msg: str, ts: str):
    st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": ts})
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg, "timestamp": ts})


# ── Chat input + command routing ──────────────────────────────────────────────
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

if prompt := st.chat_input(
    "Ask about your medical documents… or type 'send report to email@example.com'"
):
    prompt_lower = prompt.lower()
    command_handled = False
    ts_now = time.strftime("%H:%M")

    # ── Command: send report ──────────────────────────────────────────────────
    is_report_cmd = (
        any(kw in prompt_lower for kw in [
            "send report", "send analysis", "send analytics",
            "email report", "send repot",
        ])
        and "@" in prompt
    )
    if is_report_cmd:
        email_match = EMAIL_PATTERN.search(prompt)
        email_addr = email_match.group() if email_match else ""
        if _validate_email(email_addr):
            with st.chat_message("assistant"):
                with st.spinner("Sending report…"):
                    result = _post(
                        "/email/send-report",
                        json={
                            "user_query": prompt,
                            "ai_response": f"Report sent to {email_addr} via chat command.",
                            "document_insights": st.session_state.get("last_insights", {}),
                            "receiver_email": email_addr,
                            "chat_history": st.session_state.messages[-10:],
                        },
                    )
                    resp = (
                        f"✅ Analysis report sent to **{email_addr}**!"
                        if result and result.get("success")
                        else f"❌ Failed to send report to {email_addr}."
                    )
                    st.markdown(resp)
            _add_to_history(prompt, resp, ts_now)
        else:
            _add_to_history(prompt, f"❌ Invalid email address: `{email_addr}`", ts_now)
        command_handled = True

    # ── Command: support ticket ───────────────────────────────────────────────
    elif any(kw in prompt_lower for kw in [
        "support ticket", "create ticket", "open ticket", "generate ticket"
    ]):
        with st.chat_message("assistant"):
            with st.spinner("Creating support ticket…"):
                result = _post(
                    "/email/support-ticket",
                    json={
                        "user_query": prompt,
                        "ai_response": "Support ticket via chat.",
                        "user_email": st.session_state.receiver_email or None,
                        "chat_history": st.session_state.messages[-5:],
                    },
                )
                resp = (
                    "✅ Support ticket created and sent!"
                    if result and result.get("success")
                    else "❌ Failed to create support ticket."
                )
                st.markdown(resp)
        _add_to_history(prompt, resp, ts_now)
        command_handled = True

    # ── Command: process S3 ───────────────────────────────────────────────────
    elif "process s3" in prompt_lower or "process all s3" in prompt_lower:
        with st.chat_message("assistant"):
            with st.spinner("Processing S3 documents…"):
                result = _post("/documents/process-s3")
                if result and result.get("success"):
                    resp = (
                        f"✅ Processed {result['documents_processed']} S3 docs → "
                        f"{result['chunks_created']} chunks. Ready to chat!"
                    )
                    _sync_status()
                else:
                    resp = "❌ No documents could be processed from S3."
                st.markdown(resp)
        _add_to_history(prompt, resp, ts_now)
        command_handled = True

    # ── Command: save session ─────────────────────────────────────────────────
    elif "save session" in prompt_lower or "save the session" in prompt_lower:
        if st.session_state.receiver_email and _validate_email(st.session_state.receiver_email):
            with st.chat_message("assistant"):
                with st.spinner("Saving session…"):
                    result = _post(
                        "/email/save-session",
                        json={
                            "receiver_email": st.session_state.receiver_email,
                            "document_count": st.session_state.document_count,
                            "message_count": len(st.session_state.messages),
                            "recent_messages": st.session_state.messages[-5:],
                        },
                    )
                    resp = (
                        f"✅ Session saved at {result.get('timestamp','')} — summary emailed!"
                        if result and result.get("success")
                        else "❌ Failed to save session."
                    )
                    st.markdown(resp)
        else:
            resp = "❌ Please enter a valid recipient email in the sidebar first."
            with st.chat_message("assistant"):
                st.markdown(resp)
        _add_to_history(prompt, resp, ts_now)
        command_handled = True

    # ── Normal RAG chat ───────────────────────────────────────────────────────
    if not command_handled:
        # Show user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": ts_now,
        })
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(ts_now)

        if not st.session_state.vectorstore_ready:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ No documents are loaded yet. "
                    "Upload PDFs using the sidebar or click **Init from DB** to load existing data."
                )
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching documents & generating response…"):
                    result = _post("/chat", json={
                        "prompt": prompt,
                        "chat_history": st.session_state.messages[-10:],
                    })

                if result:
                    response_text = result["response"]
                    insights = result.get("insights", {})
                    st.session_state.last_insights = insights

                    st.markdown(response_text)
                    ts_resp = time.strftime("%H:%M")
                    st.caption(ts_resp)

                    # Inline insights expander (built into _render_insights)
                    _render_insights(insights)

                    # Action buttons per response
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        email_ok = bool(
                            st.session_state.receiver_email
                            and _validate_email(st.session_state.receiver_email)
                        )
                        if st.button(
                            "📧 Send Report",
                            key=f"rpt_{len(st.session_state.messages)}",
                            disabled=not email_ok,
                        ):
                            with st.spinner("Sending…"):
                                _post("/email/send-report", json={
                                    "user_query": prompt,
                                    "ai_response": response_text,
                                    "document_insights": insights,
                                    "receiver_email": st.session_state.receiver_email,
                                    "chat_history": st.session_state.messages[-10:],
                                })
                                st.success("Report sent!")

                    with b2:
                        if st.button("🎫 Ticket", key=f"tkt_{len(st.session_state.messages)}"):
                            with st.spinner("Creating…"):
                                _post("/email/support-ticket", json={
                                    "user_query": prompt,
                                    "ai_response": response_text,
                                    "user_email": st.session_state.receiver_email or None,
                                    "chat_history": st.session_state.messages[-10:],
                                })
                                st.success("Ticket created!")

                    with b3:
                        if st.button("🔍 Raw JSON", key=f"json_{len(st.session_state.messages)}"):
                            st.json(insights)

                    with b4:
                        if st.button("💾 Save", key=f"save_{len(st.session_state.messages)}"):
                            st.success("Message saved to history!")

                    # Persist to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "insights": insights,
                        "timestamp": ts_resp,
                    })
                else:
                    st.error("❌ Failed to get a response from the backend.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.82rem; padding-bottom:1rem;">
    🤖 Powered by <strong>FastAPI · Euri AI · ChromaDB · HuggingFace</strong>
    &nbsp;|&nbsp; 🏥 MediChat Pro v2.0
</div>
""", unsafe_allow_html=True)