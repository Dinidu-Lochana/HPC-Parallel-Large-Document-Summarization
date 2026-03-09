import streamlit as st
from summarizer import summarize_document

st.title("📄 AI Document Summarizer")
st.write("Upload a PDF and enter the topic for more relevant summaries.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF file",
    type=["pdf"]
)

# Topic input
topic = st.text_input("Enter the topic of this document", value="Enter Document Title")

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("Generate Summary"):

        st.subheader("Summary")

        # Placeholder for live streaming
        placeholder = st.empty()
        summary_text = ""

        # Progress bar
        progress_bar = st.progress(0)

        # Start streaming
        with st.spinner("Summarizing document..."):
            for token in summarize_document(uploaded_file, topic=topic):
                summary_text += token
                # Update placeholder with current summary
                placeholder.markdown(summary_text)

                # Update progress bar 
                progress = min(int(len(summary_text) / 5000 * 100), 100)  
                progress_bar.progress(progress)

        # Set progress bar to 100% when done
        progress_bar.progress(100)