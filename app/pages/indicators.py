import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

import streamlit as st

import shared
from engine.catalog import catalog_notes
from engine.indicator_study import categories, group_intro, lessons

shared.boot()
st.title("Indicators")
st.write(
    "Every indicator stored in the catalog. Search or pick a group, then read what it is, "
    "how the calculation works, and why a trader would use it."
)
st.caption(catalog_notes())

records = lessons()
by_id = {item.id: item for item in records}
groups = categories()

query = st.text_input("Search", placeholder="Name, idea, or formula term")
group = st.selectbox("Group", ["All"] + groups)
mode = st.segmented_control(
    "Reading",
    ["One", "Group"],
    default="One",
    key="indicator_reading",
    required=True,
)

needle = query.strip().lower()


def matches(item) -> bool:
    if group != "All" and item.category != group:
        return False
    if not needle:
        return True
    blob = " ".join(
        [
            item.id,
            item.name,
            item.category,
            item.what,
            item.how,
            item.why,
            item.formula,
            item.input_text,
            item.param_text,
        ]
    ).lower()
    return needle in blob


def rank(item) -> int:
    if not needle:
        return 0
    name = item.name.lower()
    ident = item.id.lower()
    if needle == ident or needle == name:
        return 0
    if ident.startswith(needle) or name.startswith(needle):
        return 1
    if needle in ident or needle in name:
        return 2
    return 3


shown = sorted((item for item in records if matches(item)), key=rank)
st.caption(f"{len(shown)} of {len(records)} indicators")

if not shown:
    st.info("Nothing in the catalog matches that search.")
    st.stop()


def opening(text: str) -> str:
    stop = text.find(". ")
    if stop == -1:
        return text
    return text[: stop + 1]


def source(item) -> str:
    if item.calculated and item.code_id == item.id:
        return "Calculated from one symbol's open, high, low, close, and volume."
    if item.calculated:
        return f"Calculated from one symbol's bars. In code the type is `{item.code_id}`."
    return item.limit


def render(item, *, title: bool) -> None:
    if title:
        st.subheader(item.name)
    st.caption(f"`{item.id}` · {item.category}")
    st.markdown("**What it is.** " + item.what)
    st.markdown("**How it works.** " + item.how)
    st.markdown("**Why you would use it.** " + item.why)
    st.markdown(f"**Inputs.** {item.input_text}")
    st.markdown(f"**Defaults.** {item.param_text}")
    st.markdown("**Outputs.** " + ", ".join(item.outputs))
    st.markdown(f"**Formula.** `{item.formula}`")
    st.caption(source(item))


def show_intro(category: str) -> None:
    intro = group_intro(category)
    if intro:
        st.write(intro)


if mode == "Group":
    if group != "All":
        show_intro(group)
    current = None
    for item in shown:
        if group == "All" and item.category != current:
            current = item.category
            st.subheader(current)
            show_intro(current)
        with st.expander(f"{item.name}  ·  {item.id}", expanded=len(shown) == 1):
            render(item, title=False)
    st.stop()

ids = [item.id for item in shown]
labels = [f"{item.name}  ·  {item.id}" for item in shown]
label_for = dict(zip(ids, labels))
choice_key = "study_choice"

previous_query = st.session_state.get("study_query")
if previous_query != needle:
    st.session_state["study_query"] = needle
    if needle and previous_query is not None:
        st.session_state["study_jump"] = ids[0]

jump = st.session_state.pop("study_jump", None)
if jump in ids:
    st.session_state["study_id"] = jump
    st.session_state[choice_key] = label_for[jump]

if st.session_state.get(choice_key) not in labels:
    preferred = st.session_state.get("study_id")
    if preferred not in ids:
        requested = st.query_params.get("indicator")
        preferred = requested if requested in ids else ids[0]
    st.session_state["study_id"] = preferred
    st.session_state[choice_key] = label_for[preferred]

picked = st.selectbox("Indicator", labels, key=choice_key)
item = shown[labels.index(picked)]
st.session_state["study_id"] = item.id
if st.query_params.get("indicator") != item.id:
    st.query_params["indicator"] = item.id

place = ids.index(item.id)
previous, position, nxt = st.columns([1, 3, 1])
if previous.button("Previous", disabled=place == 0, width="stretch"):
    st.session_state["study_jump"] = ids[place - 1]
    st.rerun()
position.markdown(
    f"<div style='text-align:center;padding-top:0.35rem;color:#57534e'>{place + 1} of {len(shown)} · {item.category}</div>",
    unsafe_allow_html=True,
)
if nxt.button("Next", disabled=place == len(ids) - 1, width="stretch"):
    st.session_state["study_jump"] = ids[place + 1]
    st.rerun()

if group != "All":
    show_intro(group)

reading, _rest = st.columns([1.45, 1])
with reading:
    render(item, title=True)

with st.expander(f"Names in this list ({len(shown)})"):
    st.markdown("\n".join(f"- **{entry.name}** (`{entry.id}`). {opening(entry.what)}" for entry in shown))
