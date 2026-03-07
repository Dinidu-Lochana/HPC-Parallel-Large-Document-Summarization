import streamlit as st
from summarizer import summarize_document

st.title("📄 AI Document Summarizer")

st.write("Upload a PDF and enter the topic for more relevant summaries.")

uploaded_file = st.file_uploader(
    "Upload PDF file",
    type=["pdf"]
)

topic = st.text_input("Enter the topic of this document", value="General")

if uploaded_file is not None:

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("Generate Summary"):

        with st.spinner("Summarizing document..."):

            summary = summarize_document(uploaded_file, topic=topic)

        st.subheader("Summary")
        st.write(summary)