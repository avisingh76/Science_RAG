"""
app.py  —  Streamlit chat UI (User Accounts + JWT Auth).

Changes:
  - Login / Register page shown when not logged in
  - JWT token stored in session state, sent with every request
  - Logout button in sidebar
  - All API calls use Authorization: Bearer <token> header instead of X-API-Key
  - /ingest still uses X-API-Key (admin only) from .env
"""

import os
import streamlit as st
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

API_URL     = os.getenv("API_URL", "http://localhost:8000")
APP_API_KEY = os.getenv("APP_API_KEY", "")   # admin key for ingest only

st.set_page_config(page_title="Science RAG Assistant", page_icon="🔬", layout="wide")

# ── Session state init ─────────────────────────────────────────────────────
for key, default in {
    "token":      None,
    "username":   None,
    "email":      None,
    "session_id": None,
    "messages":   [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def admin_headers() -> dict:
    return {"X-API-Key": APP_API_KEY}


# ── Cached helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_sources(token: str):
    r = requests.get(f"{API_URL}/sources",
                     headers={"Authorization": f"Bearer {token}"}, timeout=3)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def fetch_sessions(token: str):
    r = requests.get(f"{API_URL}/sessions",
                     headers={"Authorization": f"Bearer {token}"}, timeout=3)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def fetch_history(session_id: str, token: str):
    encoded = quote(session_id, safe="")
    r = requests.get(f"{API_URL}/history/{encoded}",
                     headers={"Authorization": f"Bearer {token}"}, timeout=5)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════
# LOGIN / REGISTER PAGE
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.token:
    st.title("🔬 Science Assistant")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── Login ──────────────────────────────────────────────────────────────
    with tab_login:
        st.subheader("Login to your account")
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, type="primary"):
            if not email or not password:
                st.error("Please enter email and password.")
            else:
                try:
                    r = requests.post(
                        f"{API_URL}/login",
                        json={"email": email, "password": password},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state.token    = data["access_token"]
                        st.session_state.username = data["username"]
                        st.session_state.email    = data["email"]
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Login failed."))
                except Exception as e:
                    st.error(f"Could not connect to server: {e}")

    # ── Register ───────────────────────────────────────────────────────────
    with tab_register:
        st.subheader("Create a new account")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email    = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password (min 6 chars)", type="password", key="reg_password")

        if st.button("Register", use_container_width=True, type="primary"):
            if not reg_username or not reg_email or not reg_password:
                st.error("Please fill in all fields.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    r = requests.post(
                        f"{API_URL}/register",
                        json={"email": reg_email, "username": reg_username,
                              "password": reg_password},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state.token    = data["access_token"]
                        st.session_state.username = data["username"]
                        st.session_state.email    = data["email"]
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Registration failed."))
                except Exception as e:
                    st.error(f"Could not connect to server: {e}")

    st.stop()   # Don't render the rest of the app until logged in


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP (logged in)
# ══════════════════════════════════════════════════════════════════════════
st.title("🔬 Science Assistant")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:

    # User info + logout
    st.markdown(f"👤 **{st.session_state.username}**  \n"
                f"<small>{st.session_state.email}</small>",
                unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        for key in ["token", "username", "email", "session_id", "messages"]:
            st.session_state[key] = None if key != "messages" else []
        fetch_sessions.clear()
        fetch_sources.clear()
        st.rerun()

    st.divider()

    # 1. Ingested Books
    st.markdown("### 📚 Ingested Books")
    try:
        sources = fetch_sources(st.session_state.token)
        if not sources:
            st.caption("No books ingested yet.")
        else:
            for s in sources:
                st.markdown(
                    f"📄 **{s['filename']}**  \n"
                    f"<small>{s['num_pages']} pages · {s['num_chunks']} chunks</small>",
                    unsafe_allow_html=True,
                )
    except Exception:
        st.caption("Could not load book list.")

    st.divider()

    # 2. Add New Book (admin only — uses APP_API_KEY)
    st.markdown("### ➕ Add a New Book")
    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("📥 Ingest Book", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ingest",
                        headers=admin_headers(),
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        timeout=300,
                    )
                    r.raise_for_status()
                    st.success(f"✅ '{uploaded_file.name}' ingested!")
                    fetch_sources.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ingest failed: {e}")

    st.divider()

    # 3. New Chat
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages   = []
        st.rerun()

    if st.session_state.session_id:
        if st.button("🗑️ Delete Current Chat", use_container_width=True):
            try:
                encoded = quote(st.session_state.session_id, safe="")
                requests.delete(
                    f"{API_URL}/sessions/{encoded}",
                    headers=auth_headers(), timeout=3,
                )
                fetch_sessions.clear()
            except Exception:
                pass
            st.session_state.session_id = None
            st.session_state.messages   = []
            st.rerun()

    st.divider()

    # 4. Previous Chats
    st.markdown("### 🕑 Previous Chats")
    try:
        sessions = fetch_sessions(st.session_state.token)
        if not sessions:
            st.caption("No previous chats yet.")
        else:
            for s in sessions[:10]:
                sid   = s["session_id"]
                label = f"💬 {sid}  ({s['created_at'][:10]})"
                if st.button(label, use_container_width=True, key=sid):
                    try:
                        history = fetch_history(sid, st.session_state.token)
                        st.session_state.session_id = sid
                        st.session_state.messages   = [
                            {"role": m["role"], "content": m["content"],
                             "sources": m["sources"]}
                            for m in history
                        ]
                        st.rerun()
                    except requests.exceptions.HTTPError as e:
                        if e.response is not None and e.response.status_code == 404:
                            try:
                                encoded = quote(sid, safe="")
                                requests.delete(
                                    f"{API_URL}/sessions/{encoded}",
                                    headers=auth_headers(), timeout=3,
                                )
                                fetch_sessions.clear()
                                st.rerun()
                            except Exception:
                                pass
                        else:
                            st.error(f"Could not load chat: {e}")
                    except Exception as e:
                        st.error(f"Could not load chat: {e}")
    except Exception:
        st.caption("Could not load previous chats.")


# ── Chat messages ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📄 " + " · ".join(msg["sources"]))

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about any of your science books...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"message": user_input,
                          "session_id": st.session_state.session_id},
                    headers=auth_headers(),
                    timeout=60,
                )
                r.raise_for_status()
                data    = r.json()

                st.session_state.session_id = data["session_id"]
                answer  = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.caption("📄 " + " · ".join(sources))

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })
                fetch_sessions.clear()

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    st.error("Session expired. Please log in again.")
                    st.session_state.token = None
                    st.rerun()
                else:
                    st.error(f"❌ Error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")