import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from PyPDF2 import PdfReader

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    return model, tokenizer



model, tokenizer = load_model()

# Function to handle question answering
def answer_question(question, text, model, tokenizer, max_len=512, stride=256):
    all_answers = []
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=max_len, stride=stride)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    for i in range(0, input_ids.shape[1], stride):
        input_chunk = input_ids[:, i:i+max_len]
        token_type_chunk = token_type_ids[:, i:i+max_len]

        if input_chunk.shape[1] < max_len:
            padding_length = max_len - input_chunk.shape[1]
            input_chunk = torch.cat([input_chunk, torch.tensor([[tokenizer.pad_token_id] * padding_length])], dim=-1)
            token_type_chunk = torch.cat([token_type_chunk, torch.tensor([[0] * padding_length])], dim=-1)

        with torch.no_grad():
            outputs = model(input_chunk, token_type_ids=token_type_chunk)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        all_tokens = tokenizer.convert_ids_to_tokens(input_chunk[0])
        answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])
        all_answers.append(answer)

    return ' '.join(all_answers)

# Streamlit app layout
st.title('PDF Question-Answering System')

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Text input for the question
question = st.text_input("Enter your question:")

# Submit button
if st.button('Submit'):
    if uploaded_file is not None and question:
        # Read the PDF file and extract text
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Get the answer from the model
        answer = answer_question(question, text, model, tokenizer)

        # Display the answer
        st.write("**Answer:**", answer)
    else:
        st.write("Please upload a PDF file and enter a question.")
