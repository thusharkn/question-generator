import streamlit as st
import tempfile
from utils import extract_text_from_pdf
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Concept Question Generator")

st.title("Generate Questions:")

uploaded_file = st.file_uploader("Upload the PDF", type="pdf")
num_q = st.slider("Number of questions", 1, 2, 10)

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

qgen = load_model()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
        content = extract_text_from_pdf(pdf_path)

    if st.button("Generate Questions"):
        with st.spinner("Generating..."):
            prompt = f"Generate {num_q} relevant questions from the following educational content:\n{content[:2000]}"
            result = qgen(prompt, max_length=512, do_sample=True)[0]['generated_text']
            st.subheader("Generated Questions")
            st.text_area("Output", result, height=300)
