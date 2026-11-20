import os

import requests
import streamlit as st

from db import log_event, get_events, get_unanswered_count, init_db

st.set_page_config(
    page_title="TheCarePal Support Assistant",
    page_icon="💙",
    layout="wide"
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DEFAULT_API_URL = os.getenv("CAREPAL_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60  # seconds

# Ensure the SQLite escalation DB exists
init_db()

# Phrases returned by app.py / chat.py when nothing useful was found.
# Used to decide whether to offer the human-handoff form.
NOT_FOUND_MARKERS = (
    "i couldn't find this information",
    "i couldn't find reliable information",
    "sorry, i couldn't retrieve the information",
    "search service is not ready",
    "please provide a valid question",
)


# -------------------------------------------------
# RAG INTEGRATION (calls the FastAPI backend: POST /search)
# -------------------------------------------------

def get_rag_response(question: str, api_url: str) -> dict:
    """
    Calls the FastAPI backend and normalises the response to:
    {
        "answer": str,
        "sources": list[str],
        "chunks": list[{"text": str, "score": float}],
        "found": bool
    }

    The current backend returns only {"answer": "..."}; "sources" and
    "chunks" are picked up automatically if you add them to SearchResponse.
    """
    response = requests.post(
        f"{api_url.rstrip('/')}/search",
        json={"query": question},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []
    chunks = data.get("chunks") or []
    signal = data.get("signal")  # 'NOT_FOUND' | 'ESCALATE' | None

    # An answer is "found" only when we have real content and no escalation signal
    found = bool(answer) and signal not in ("NOT_FOUND", "ESCALATE") and not any(
        marker in answer.lower() for marker in NOT_FOUND_MARKERS
    )

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "found": found,
        "signal": signal,
        "raw_keys": list(data.keys()),
    }


def check_backend(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def save_support_request(name: str, phone: str, question: str) -> None:
    """Persist a human-handoff request to SQLite for admin review."""
    log_event(question=question, answer="", signal="ESCALATE", name=name, phone=phone)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "support_form" not in st.session_state:
    st.session_state.support_form = False

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("⚙️ Configuration")

api_url = st.sidebar.text_input("Backend URL", value=DEFAULT_API_URL)

if check_backend(api_url):
    st.sidebar.success("Backend connected")
else:
    st.sidebar.error("Backend not reachable")

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieved Chunks")

# Placeholder: filled at the END of the script, after the new question
# has been processed, so chunks always match the latest answer.
chunks_box = st.sidebar.empty()

st.sidebar.markdown("---")

if st.sidebar.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.retrieved_chunks = []
    st.session_state.support_form = False
    st.session_state.last_question = ""
    st.rerun()

# -------------------------------------------------
# SIDEBAR: ADMIN VIEW (simple password gate)
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Admin")

if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

if not st.session_state.admin_auth:
    admin_pw = st.sidebar.text_input("Admin password", type="password", key="admin_pw_input")
    if st.sidebar.button("Unlock admin", key="admin_unlock"):
        if admin_pw == "carepal2026":  # change in production
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.sidebar.error("Incorrect password")
else:
    st.sidebar.success("Admin unlocked")
    if st.sidebar.button("Lock admin", key="admin_lock"):
        st.session_state.admin_auth = False
        st.rerun()

    unanswered = get_unanswered_count()
    st.sidebar.metric("Unanswered / Escalated", unanswered)

    if st.sidebar.button("Refresh admin data", key="admin_refresh"):
        st.rerun()

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------

st.title("💙 TheCarePal Support Assistant")

st.caption(
    "Ask questions about our home care services, "
    "elderly care, baby care and other support services."
)

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.markdown(f"- {source}")


# -------------------------------------------------
# USER QUESTION
# -------------------------------------------------

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching our knowledge base..."):

            try:
                result = get_rag_response(question, api_url)

                answer = result["answer"]
                sources = result["sources"]
                chunks = result["chunks"]
                found = result["found"]
                signal = result.get("signal")

                st.session_state.retrieved_chunks = chunks
                st.session_state.raw_keys = result["raw_keys"]
                st.session_state.last_question = question

                # --- ESCALATE signal: show answer + support form ---
                if signal == "ESCALATE":
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.info("Would you like me to connect you with our support team?")
                    st.session_state.support_form = True

                # --- NOT_FOUND signal: show fallback + support form ---
                elif signal == "NOT_FOUND" or not found:
                    fallback = answer or (
                        "I couldn't find reliable information about "
                        "this question in our available resources."
                    )
                    handoff = (
                        "Would you like me to connect you "
                        "with our support team?"
                    )
                    st.warning(fallback)
                    st.write(handoff)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{fallback}\n\n{handoff}"
                    })
                    st.session_state.support_form = True

                # --- Normal answer ---
                else:
                    st.markdown(answer)

                    if sources:
                        st.markdown("**Sources:**")
                        for source in sources:
                            st.markdown(f"- {source}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot reach the backend at {api_url}. "
                    "Make sure the FastAPI server is running."
                )

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")


# -------------------------------------------------
# HUMAN HANDOFF
# -------------------------------------------------

if st.session_state.support_form:

    st.markdown("---")
    st.subheader("📞 Connect with our Support Team")

    st.write(
        "Please share your contact details so our team "
        "can follow up with you."
    )

    with st.form("support_contact_form"):

        name = st.text_input("Your Name")
        phone = st.text_input("Phone Number")

        consent = st.checkbox(
            "I agree to share my contact details "
            "with TheCarePal support team for follow-up."
        )

        submitted = st.form_submit_button(
            "Submit Request",
            type="primary"
        )

        if submitted:

            if not name.strip():
                st.error("Please enter your name.")

            elif not phone.strip():
                st.error("Please enter your phone number.")

            elif not consent:
                st.error("Please provide your consent.")

            else:
                try:
                    save_support_request(
                        name.strip(),
                        phone.strip(),
                        st.session_state.last_question,
                    )
                    st.success(
                        "Your request has been submitted successfully!"
                    )
                    st.session_state.support_form = False
                except Exception as e:
                    st.error(f"Could not save your request: {e}")


# -------------------------------------------------
# RENDER RETRIEVED CHUNKS (sidebar)
# -------------------------------------------------

with chunks_box.container():
    if st.session_state.retrieved_chunks:
        for rank, chunk in enumerate(
            st.session_state.retrieved_chunks, start=1
        ):
            with st.expander(
                f"Chunk #{rank} | Score: {chunk.get('score', 0):.4f}"
            ):
                st.write(chunk.get("text", ""))
    else:
        st.caption("No retrieved chunks yet.")

    # Debug: shows which fields the backend actually returned
    if st.session_state.get("raw_keys") is not None:
        st.caption(f"Backend returned fields: {st.session_state.raw_keys}")


# -------------------------------------------------
# ADMIN DASHBOARD (main area, shown only when unlocked)
# -------------------------------------------------
if st.session_state.get("admin_auth"):
    st.markdown("---")
    st.subheader("📋 Admin — Escalation & Support Requests")

    events = get_events(limit=200)

    if not events:
        st.info("No escalation or support requests logged yet.")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        not_found_count = sum(1 for e in events if e["signal"] == "NOT_FOUND")
        escalate_count = sum(1 for e in events if e["signal"] == "ESCALATE")
        with_contact = sum(1 for e in events if e["name"] or e["phone"])

        col1.metric("Total events", len(events))
        col2.metric("Not found (NOT_FOUND)", not_found_count)
        col3.metric("Human escalation (ESCALATE)", escalate_count)
        st.caption(f"With contact details provided: {with_contact}")

        st.markdown("### Recent events")
        for ev in events:
            signal_label = {
                "NOT_FOUND": "🔍 Not found",
                "ESCALATE": "📞 Human escalation",
            }.get(ev["signal"], ev["signal"] or "—")

            with st.expander(
                f"{signal_label} — {ev['created_at'][:16]} — "
                f"{ev['question'][:60]}..."
            ):
                st.markdown(f"**Question:** {ev['question']}")
                st.markdown(f"**Answer:** {ev['answer'] or '_(none)_'}")
                st.markdown(f"**Signal:** `{ev['signal']}`")
                if ev["name"]:
                    st.markdown(f"**Name:** {ev['name']}")
                if ev["phone"]:
                    st.markdown(f"**Phone:** {ev['phone']}")
                st.markdown(f"**Created at:** {ev['created_at']}")