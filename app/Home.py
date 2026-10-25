import streamlit as st

st.set_page_config(page_title="Strategy desk", layout="wide", initial_sidebar_state="expanded")

navigation = st.navigation(
    [
        st.Page("pages/bank.py", title="Bank", default=True),
        st.Page("pages/code.py", title="Code"),
        st.Page("pages/indicators.py", title="Indicators"),
    ]
)
navigation.run()
