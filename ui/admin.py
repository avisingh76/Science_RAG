"""
admin.py  —  Admin Dashboard for Science RAG.

Run with:
  streamlit run ui/admin.py --server.port 8502

Access: http://localhost:8502
Requires APP_API_KEY from .env to login.
"""

import os
import streamlit as st
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

API_URL     = os.getenv("API_URL", "http://localhost:8000")
APP_API_KEY = os.getenv("APP_API_KEY", "")
HEADERS     = {"X-API-Key": APP_API_KEY}

st.set_page_config(
    page_title="Science RAG — Admin",
    page_icon="⚙️",
    layout="wide",
)

# ── Admin login ────────────────────────────────────────────────────────────
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    st.title("⚙️ Admin Dashboard")
    st.markdown("---")
    st.subheader("Enter Admin API Key")
    key_input = st.text_input("API Key", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if key_input == APP_API_KEY:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Invalid API Key.")
    st.stop()


# ── Cached helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def fetch_stats():
    r = requests.get(f"{API_URL}/admin/stats", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def fetch_users():
    r = requests.get(f"{API_URL}/admin/users", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def fetch_sessions():
    r = requests.get(f"{API_URL}/admin/sessions", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def fetch_books():
    r = requests.get(f"{API_URL}/sources", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_logout = st.columns([6, 1])
with col_title:
    st.title("⚙️ Admin Dashboard")
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_overview, tab_users, tab_sessions, tab_books = st.tabs([
    "📊 Overview", "👥 Users", "💬 Sessions", "📚 Books"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("System Overview")

    try:
        stats = fetch_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("👥 Total Users",    stats["total_users"])
        c2.metric("💬 Total Sessions", stats["total_sessions"])
        c3.metric("📨 Total Messages", stats["total_messages"])
        c4.metric("📚 Books Ingested", stats["total_books"])
        c5.metric("🧩 Total Chunks",   stats["total_chunks"])
    except Exception as e:
        st.error(f"Could not load stats: {e}")

    st.markdown("---")

    # Quick refresh
    if st.button("🔄 Refresh Stats"):
        fetch_stats.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: USERS
# ══════════════════════════════════════════════════════════════════════════
with tab_users:
    st.subheader("Registered Users")

    if st.button("🔄 Refresh", key="refresh_users"):
        fetch_users.clear()
        st.rerun()

    try:
        users = fetch_users()
        if not users:
            st.info("No users registered yet.")
        else:
            st.markdown(f"**Total: {len(users)} users**")
            st.markdown("---")

            # Table header
            h1, h2, h3, h4, h5 = st.columns([1, 2, 2, 2, 1])
            h1.markdown("**ID**")
            h2.markdown("**Username**")
            h3.markdown("**Email**")
            h4.markdown("**Registered**")
            h5.markdown("**Sessions**")
            st.markdown("---")

            for u in users:
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
                c1.write(u["user_id"])
                c2.write(u["username"])
                c3.write(u["email"])
                c4.write(u["created_at"][:10])
                c5.write(u["session_count"])

    except Exception as e:
        st.error(f"Could not load users: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: SESSIONS
# ══════════════════════════════════════════════════════════════════════════
with tab_sessions:
    st.subheader("All Sessions")

    if st.button("🔄 Refresh", key="refresh_sessions"):
        fetch_sessions.clear()
        st.rerun()

    try:
        sessions = fetch_sessions()
        if not sessions:
            st.info("No sessions yet.")
        else:
            st.markdown(f"**Total: {len(sessions)} sessions** (showing latest 100)")
            st.markdown("---")

            # Search filter
            search = st.text_input("🔍 Filter by username or session ID", "")

            filtered = [
                s for s in sessions
                if not search
                or search.lower() in (s.get("username") or "").lower()
                or search.lower() in s["session_id"].lower()
            ]

            # Table header
            h1, h2, h3, h4 = st.columns([3, 2, 2, 1])
            h1.markdown("**Session**")
            h2.markdown("**User**")
            h3.markdown("**Created**")
            h4.markdown("**Messages**")
            st.markdown("---")

            for s in filtered:
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                c1.write(s["session_id"])
                c2.write(s.get("username") or "—")
                c3.write(s["created_at"][:10] if s["created_at"] else "—")
                c4.write(s["message_count"])

    except Exception as e:
        st.error(f"Could not load sessions: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: BOOKS
# ══════════════════════════════════════════════════════════════════════════
with tab_books:
    st.subheader("Ingested Books")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_books"):
            fetch_books.clear()
            st.rerun()

    try:
        books = fetch_books()
        if not books:
            st.info("No books ingested yet.")
        else:
            st.markdown(f"**Total: {len(books)} books**")
            st.markdown("---")

            for b in books:
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"📄 **{b['filename']}**  \n"
                        f"<small>{b['num_pages']} pages · {b['num_chunks']} chunks · "
                        f"ingested {b['ingested_at'][:10]}</small>",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete", key=f"del_{b['filename']}", use_container_width=True):
                        st.session_state[f"confirm_{b['filename']}"] = True

                # Confirmation step before delete
                if st.session_state.get(f"confirm_{b['filename']}"):
                    st.warning(
                        f"⚠️ Delete **{b['filename']}**? "
                        "This will remove it from the vector store permanently."
                    )
                    yes_col, no_col = st.columns(2)
                    with yes_col:
                        if st.button("✅ Yes, delete", key=f"yes_{b['filename']}",
                                     use_container_width=True):
                            try:
                                encoded = quote(b["filename"], safe="")
                                r = requests.delete(
                                    f"{API_URL}/admin/books/{encoded}",
                                    headers=HEADERS, timeout=60,
                                )
                                r.raise_for_status()
                                st.success(f"✅ '{b['filename']}' deleted.")
                                fetch_books.clear()
                                st.session_state.pop(f"confirm_{b['filename']}", None)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                    with no_col:
                        if st.button("❌ Cancel", key=f"no_{b['filename']}",
                                     use_container_width=True):
                            st.session_state.pop(f"confirm_{b['filename']}", None)
                            st.rerun()

                st.markdown("---")

    except Exception as e:
        st.error(f"Could not load books: {e}")

    # Upload new book
    st.subheader("➕ Add a New Book")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"],
                                label_visibility="collapsed", key="admin_upload")
    if uploaded:
        if st.button("📥 Ingest Book", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded.name}'... this may take a minute."):
                try:
                    r = requests.post(
                        f"{API_URL}/ingest",
                        headers=HEADERS,
                        files={"file": (uploaded.name, uploaded, "application/pdf")},
                        timeout=300,
                    )
                    r.raise_for_status()
                    st.success(f"✅ '{uploaded.name}' ingested successfully!")
                    fetch_books.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ingest failed: {e}")