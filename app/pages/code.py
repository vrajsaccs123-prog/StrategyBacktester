import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

import re

import streamlit as st

HOSTED = Path("/mount/src").exists()

import shared
from engine.book import check_book_strategy, new_strategy_source
from strategies.registry import discover

shared.boot()
st.title("Code")
st.write(
    "A strategy file is the filter, the buy, the sell, the stop, the size, and the costs. "
    "Saving reloads the bank. The chart draws whatever indicator_frame returns."
)
st.caption("A file joins the bank only when it has those stages and can draw its indicators.")

files = shared.editor_files()
names = [path.name for path in files]


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"strategy_{slug}" if slug else "strategy"
    return slug


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_") if part)


with st.expander("New strategy"):
    title = st.text_input("Name", placeholder="Weekly trend", disabled=HOSTED)
    if st.button("Create strategy", type="primary", disabled=HOSTED):
        cleaned = title.strip()
        if not cleaned:
            st.error("Give the strategy a name.")
        else:
            slug = _slug(cleaned)
            path = shared.STRATEGIES / f"{slug}.py"
            if path.exists():
                st.error(f"{path.name} already exists.")
            else:
                source = new_strategy_source(_class_name(slug), slug, cleaned)
                path.write_text(source, encoding="utf-8")
                st.session_state["editor_path"] = path.name
                st.session_state["editor_text"] = source
                st.rerun()

if not names and "editor_text" not in st.session_state:
    st.info("Create a strategy to start a file.")
    st.stop()

if names:
    current = st.session_state.get("editor_path")
    if current not in names:
        current = names[0]
        st.session_state["editor_path"] = current
        st.session_state["editor_text"] = (shared.STRATEGIES / current).read_text(encoding="utf-8")
    picked = st.selectbox("File", names, index=names.index(st.session_state["editor_path"]))
    if picked != st.session_state["editor_path"]:
        st.session_state["editor_path"] = picked
        st.session_state["editor_text"] = (shared.STRATEGIES / picked).read_text(encoding="utf-8")
        st.rerun()

if HOSTED:
    st.caption("This shared copy is read-only. Strategy files are changed on your own machine.")

st.text_area("Source", height=640, key="editor_text", label_visibility="collapsed", disabled=HOSTED)

save_col, delete_col = st.columns(2)
if save_col.button("Save", type="primary", disabled=HOSTED):
    filename = st.session_state.get("editor_path")
    if not filename:
        st.error("Create a strategy before saving.")
    else:
        path = shared.STRATEGIES / filename
        path.write_text(st.session_state.get("editor_text") or "", encoding="utf-8")
        stem = path.stem
        found = [item for item in discover() if item.__class__.__module__ == f"strategies.{stem}"]
        messages = shared.load_errors()
        for message in messages:
            st.error(message)
        if not found and not messages:
            st.warning("The file saved, and it does not define a strategy class with a key.")
        for item in found:
            problems = check_book_strategy(item)
            if problems:
                for problem in problems:
                    st.warning(problem)
            else:
                st.success(f"{item.name} is in the bank.")
                shared.select_strategy(item)

confirm = delete_col.checkbox("Delete this file", disabled=HOSTED)
if delete_col.button("Delete", disabled=HOSTED):
    if not confirm:
        st.caption("Tick the box, then delete.")
    else:
        filename = st.session_state.get("editor_path")
        path = shared.STRATEGIES / filename if filename else None
        if path and path.exists() and path.parent == shared.STRATEGIES and path.suffix == ".py":
            path.unlink()
            st.session_state.pop("editor_path", None)
            st.session_state.pop("editor_text", None)
            st.rerun()
