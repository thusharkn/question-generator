import streamlit as st
import tempfile
from utils import extract_text_from_pdf
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Concept Question Generator")

st.title("Generate Questions:")

uploaded_file = st.file_uploader("Upload the PDF", type="pdf")
num_q = st.slider("Number of questions", 1, 10, 5)

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, temperature=0.9)

qgen = None
content = ""

if uploaded_file:
    with st.spinner("Loading model... (one-time setup)"):
        qgen = load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
        content = extract_text_from_pdf(pdf_path)

if uploaded_file and qgen:
    if st.button("Generate Questions"):
        with st.spinner("Generating..."):
            questions = []
            context = content[:2000]  # Limit to first chunk
            for i in range(num_q):
                prompt = f"Based on the following content, generate 1 relevant academic question:\n\n{context}"
                output = qgen(prompt, max_length=128, do_sample=True)[0]['generated_text']
                questions.append(f"{i+1}. {output.strip()}")
            result = "\n".join(questions)
            st.subheader("Generated Questions")
            st.text_area("Output", result, height=300)
