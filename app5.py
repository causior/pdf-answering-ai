import streamlit as st
from PyPDF2 import PdfReader
import pickle
import torch
# Placeholder for your 'answer_question' function
def answer_question(question, text):
    
    pass


contexts = [
    "Nokia C12 Android 12 (Go Edition) Smartphone, All-Day Battery, 4GB RAM (2GB RAM + 2GB Virtual RAM) + 64GB Capacity | Light Mint",
    "Nokia G21 Android Smartphone, Dual SIM, 3-Day Battery Life, 6GB RAM + 128GB Storage, 50MP Triple AI Camera | Nordic Blue",
    "realme narzo 50i Prime (Dark Blue 4GB RAM+64GB Storage) Octa-core Processor | 5000 mAh Battery",
    "realme narzo N53 (Feather Gold, 4GB+64GB) 33W Segment Fastest Charging | Slimmest Phone in Segment | 90 Hz Smooth Display",
    "realme narzo N55 (Prime Blue, 4GB+64GB) 33W Segment Fastest Charging | Super High-res 64MP Primary AI Camera",
    "Redmi 9A Sport (Carbon Black, 2GB RAM, 32GB Storage) | 2GHz Octa-core Helio G25 Processor | 5000 mAh Battery",
    "Redmi 11 Prime 5G (Thunder Black, 4GB RAM, 64GB Storage) | Prime Design | MTK Dimensity 700 | 50 MP Dual Cam | 5000mAh | 7 Band 5G",
    "Redmi 12C (Royal Blue, 4GB RAM, 64GB Storage) | High Performance Mediatek Helio G85 | Big 17cm(6.71) HD+ Display with 5000mAh(typ) Battery",
    "Redmi A1 (Light Green, 2GB RAM 32GB ROM) | Segment Best AI Dual Cam | 5000mAh Battery | Leather Texture Design | Android 12",
    "Samsung Galaxy M04 Light Green, 4GB RAM, 64GB Storage | Upto 8GB RAM with RAM Plus | MediaTek Helio P35 Octa-core Processor | 5000 mAh Battery | 13MP Dual Camera",
    "Samsung Galaxy M13 (Midnight Blue, 4GB, 64GB Storage) | 6000mAh Battery | Upto 8GB RAM with RAM Plus",
    "Tecno Camon 20 (Serenity Blue, 8GB RAM,256GB Storage)|16GB Expandable RAM | 64MP RGBW Rear Camera|6.67 FHD+ Big AMOLED with in-Display Fingerprint Sensor",
    "Tecno POP 7 Pro (Uyuni Blue, 3GB RAM,64GB Storage) | Type C Port | 12MP Dual Camera | Up to 6GB RAM with Memory Fusion"
]

questions_answers = [
    {
        "context_index": 0,
        "question": "What is the operating system of the Nokia C12 smartphone?",
        "answer": "Android 12 (Go Edition)"
    },
    {
        "context_index": 0,
        "question": "How much RAM does the Nokia C12 have?",
        "answer": "4GB"
    },
    {
        "context_index": 0,
        "question": "Does the Nokia C12 have virtual RAM?",
        "answer": "(2GB RAM + 2GB Virtual RAM)"
    },
    {
        "context_index": 0,
        "question": "What is the total capacity of the Nokia C12?",
        "answer": "64GB"
    },
    {
        "context_index": 0,
        "question": "What is the color option available for the Nokia C12?",
        "answer": "Light Mint"
    },
    {
        "context_index": 1,
        "question": "What is the model name of the Nokia smartphone?",
        "answer": "Nokia G21"
    },
    {
        "context_index": 1,
        "question": "What is the operating system of the Nokia G21?",
        "answer": "Android"
    },
    {
        "context_index": 1,
        "question": "How many SIM cards does the Nokia G21 support?",
        "answer": "Dual SIM"
    },
    {
        "context_index": 1,
        "question": "How long is the battery life of the Nokia G21?",
        "answer": "3-Day Battery Life"
    },
    {
        "context_index": 1,
        "question": "How much RAM does the Nokia G21 have?",
        "answer": "6GB"
    },
    {
        "context_index": 1,
        "question": "What is the storage capacity of the Nokia G21?",
        "answer": "128GB"
    },
    {
        "context_index": 1,
        "question": "What is the resolution of the main camera on the Nokia G21?",
        "answer": "50MP"
    },
    {
        "context_index": 1,
        "question": "How many AI cameras does the Nokia G21 have?",
        "answer": "Triple AI Camera"
    },
    {
        "context_index": 1,
        "question": "What is the color option available for the Nokia G21?",
        "answer": "Nordic Blue"
    },
    {
        "context_index": 2,
        "question": "What is the model name of the Realme smartphone?",
        "answer": "Realme narzo 50i Prime"
    },
    {
        "context_index": 2,
        "question": "What is the color option available for the Realme narzo 50i Prime?",
        "answer": "Dark Blue"
    },
    {
        "context_index": 2,
        "question": "How much RAM does the Realme narzo 50i Prime have?",
        "answer": "4GB"
    },
    {
        "context_index": 2,
        "question": "What is the storage capacity of the Realme narzo 50i Prime?",
        "answer": "64GB"
    },
    {
        "context_index": 2,
        "question": "What type of processor does the Realme narzo 50i Prime have?",
        "answer": "Octa-core Processor"
    },
    {
        "context_index": 2,
        "question": "What is the battery capacity of the Realme narzo 50i Prime?",
        "answer": "5000 mAh"
    },
    {
        "context_index": 3,
        "question": "What is the model name of the Realme smartphone?",
        "answer": "Realme narzo N53"
    },
    {
        "context_index": 3,
        "question": "What is the color option available for the Realme narzo N53?",
        "answer": "Feather Gold"
    },
    {
        "context_index": 3,
        "question": "How much RAM does the Realme narzo N53 have?",
        "answer": "4GB"
    },
    {
        "context_index": 3,
        "question": "What is the storage capacity of the Realme narzo N53?",
        "answer": "64GB"
    },
    {
        "context_index": 3,
        "question": "What is the charging speed of the Realme narzo N53?",
        "answer": "33W Segment Fastest Charging"
    },
    {
        "context_index": 3,
        "question": "What is the special feature of the Realme narzo N53 in terms of phone thickness?",
        "answer": "Slimmest Phone in Segment"
    },
    {
        "context_index": 3,
        "question": "What is the refresh rate of the display on the Realme narzo N53?",
        "answer": "90 Hz"
    },
    {
        "context_index": 4,
        "question": "What is the model name of the Realme smartphone?",
        "answer": "Realme narzo N55"
    },
    {
        "context_index": 4,
        "question": "What is the color option available for the Realme narzo N55?",
        "answer": "Prime Blue"
    },
    {
        "context_index": 4,
        "question": "How much RAM does the Realme narzo N55 have?",
        "answer": "4GB"
    },
    {
        "context_index": 4,
        "question": "What is the storage capacity of the Realme narzo N55?",
        "answer": "64GB"
    },
    {
        "context_index": 4,
        "question": "What is the charging speed of the Realme narzo N55?",
        "answer": "33W Segment Fastest Charging"
    },
    {
        "context_index": 4,
        "question": "What is the resolution of the primary AI camera on the Realme narzo N55?",
        "answer": "Super High-res 64MP"
    },
    {
        "context_index": 5,
        "question": "What is the model name of the Redmi smartphone?",
        "answer": "Redmi 9A Sport"
    },
    {
        "context_index": 5,
        "question": "What is the color option available for the Redmi 9A Sport?",
        "answer": "Carbon Black"
    },
    {
        "context_index": 5,
        "question": "How much RAM does the Redmi 9A Sport have?",
        "answer": "2GB"
    },
    {
        "context_index": 5,
        "question": "What is the storage capacity of the Redmi 9A Sport?",
        "answer": "32GB"
    },
    {
        "context_index": 5,
        "question": "What is the processor of the Redmi 9A Sport?",
        "answer": "2GHz Octa-core Helio G25 Processor"
    },
    {
        "context_index": 5,
        "question": "What is the battery capacity of the Redmi 9A Sport?",
        "answer": "5000 mAh"
    },
    {
        "context_index": 6,
        "question": "What is the model name of the Redmi smartphone?",
        "answer": "Redmi 11 Prime 5G"
    },
    {
        "context_index": 6,
        "question": "What is the color option available for the Redmi 11 Prime 5G?",
        "answer": "Thunder Black"
    },
    {
        "context_index": 6,
        "question": "How much RAM does the Redmi 11 Prime 5G have?",
        "answer": "4GB"
    },
    {
        "context_index": 6,
        "question": "What is the storage capacity of the Redmi 11 Prime 5G?",
        "answer": "64GB"
    },
    {
        "context_index": 6,
        "question": "What is the special feature of the Redmi 11 Prime 5G in terms of design?",
        "answer": "Prime Design"
    },
    {
        "context_index": 6,
        "question": "What is the processor of the Redmi 11 Prime 5G?",
        "answer": "MTK Dimensity 700"
    },
    {
        "context_index": 6,
        "question": "What is the resolution of the dual camera on the Redmi 11 Prime 5G?",
        "answer": "50 MP"
    },
    {
        "context_index": 6,
        "question": "What is the battery capacity of the Redmi 11 Prime 5G?",
        "answer": "5000mAh"
    },
    {
        "context_index": 6,
        "question": "How many 5G bands does the Redmi 11 Prime 5G support?",
        "answer": "7 Band 5G"
    },
    {
        "context_index": 7,
        "question": "What is the model name of the Redmi smartphone?",
        "answer": "Redmi 12C"
    },
    {
        "context_index": 7,
        "question": "What is the color option available for the Redmi 12C?",
        "answer": "Royal Blue"
    },
    {
        "context_index": 7,
        "question": "How much RAM does the Redmi 12C have?",
        "answer": "4GB"
    },
    {
        "context_index": 7,
        "question": "What is the storage capacity of the Redmi 12C?",
        "answer": "64GB"
    },
    {
        "context_index": 7,
        "question": "What is the processor of the Redmi 12C?",
        "answer": "High Performance Mediatek Helio G85"
    },
    {
        "context_index": 7,
        "question": "What is the size of the display on the Redmi 12C?",
        "answer": "Big 17cm(6.71) HD+ Display"
    },
    {
        "context_index": 7,
        "question": "What is the battery capacity of the Redmi 12C?",
        "answer": "5000mAh(typ) Battery"
    },
    {
        "context_index": 8,
        "question": "What is the model name of the Redmi smartphone?",
        "answer": "Redmi A1"
    },
    {
        "context_index": 8,
        "question": "What is the color option available for the Redmi A1?",
        "answer": "Light Green"
    },
    {
        "context_index": 8,
        "question": "How much RAM does the Redmi A1 have?",
        "answer": "2GB"
    },
    {
        "context_index": 8,
        "question": "What is the storage capacity of the Redmi A1?",
        "answer": "32GB"
    },
    {
        "context_index": 8,
        "question": "What is the special feature of the camera on the Redmi A1?",
        "answer": "Segment Best AI Dual Cam"
    },
    {
        "context_index": 8,
        "question": "What is the battery capacity of the Redmi A1?",
        "answer": "5000mAh Battery"
    },
    {
        "context_index": 8,
        "question": "What is the design feature of the Redmi A1?",
        "answer": "Leather Texture Design"
    },
    {
        "context_index": 8,
        "question": "What is the operating system of the Redmi A1?",
        "answer": "Android 12"
    },
    {
        "context_index": 9,
        "question": "What is the model name of the Samsung smartphone?",
        "answer": "Samsung Galaxy M04"
    },
    {
        "context_index": 9,
        "question": "What is the color option available for the Samsung Galaxy M04?",
        "answer": "Light Green"
    },
    {
        "context_index": 9,
        "question": "How much RAM does the Samsung Galaxy M04 have?",
        "answer": "4GB"
    },
    {
        "context_index": 9,
        "question": "What is the storage capacity of the Samsung Galaxy M04?",
        "answer": "64GB"
    },
    {
        "context_index": 9,
        "question": "How much RAM can the Samsung Galaxy M04 have with RAM Plus?",
        "answer": "Upto 8GB RAM with RAM Plus"
    },
    {
        "context_index": 9,
        "question": "What is the processor of the Samsung Galaxy M04?",
        "answer": "MediaTek Helio P35 Octa-core Processor"
    },
    {
        "context_index": 9,
        "question": "What is the battery capacity of the Samsung Galaxy M04?",
        "answer": "5000 mAh"
    },
    {
        "context_index": 9,
        "question": "What is the resolution of the dual camera on the Samsung Galaxy M04?",
        "answer": "13MP"
    },
    {
        "context_index": 10,
        "question": "What is the model name of the Samsung smartphone?",
        "answer": "Samsung Galaxy M13"
    },
    {
        "context_index": 10,
        "question": "What is the color option available for the Samsung Galaxy M13?",
        "answer": "Midnight Blue"
    },
    {
        "context_index": 10,
        "question": "How much RAM does the Samsung Galaxy M13 have?",
        "answer": "4GB"
    },
    {
        "context_index": 10,
        "question": "What is the storage capacity of the Samsung Galaxy M13?",
        "answer": "64GB"
    },
    {
        "context_index": 10,
        "question": "What is the battery capacity of the Samsung Galaxy M13?",
        "answer": "6000mAh"
    },
    {
        "context_index": 10,
        "question": "How much RAM can the Samsung Galaxy M13 have with RAM Plus?",
        "answer": "Upto 8GB RAM with RAM Plus"
    },
    {
        "context_index": 11,
        "question": "What is the model name of the Tecno smartphone?",
        "answer": "Tecno Camon 20"
    },
    {
        "context_index": 11,
        "question": "What is the color option available for the Tecno Camon 20?",
        "answer": "Serenity Blue"
    },
    {
        "context_index": 11,
        "question": "How much RAM does the Tecno Camon 20 have?",
        "answer": "8GB"
    },
    {
        "context_index": 11,
        "question": "What is the storage capacity of the Tecno Camon 20?",
        "answer": "256GB"
    },
    {
        "context_index": 11,
        "question": "How much expandable RAM can the Tecno Camon 20 have?",
        "answer": "16GB"
    },
    {
        "context_index": 11,
        "question": "What is the resolution of the rear camera on the Tecno Camon 20?",
        "answer": "64MP RGBW"
    },
    {
        "context_index": 11,
        "question": "What is the size of the display on the Tecno Camon 20?",
        "answer": "6.67 FHD+"
    },
    {
        "context_index": 11,
        "question": "What type of display does the Tecno Camon 20 have?",
        "answer": "Big AMOLED"
    },
    {
        "context_index": 11,
        "question": "What feature is integrated into the display of the Tecno Camon 20?",
        "answer": "In-Display Fingerprint Sensor"
    },
    {
        "context_index": 12,
        "question": "What is the model name of the Tecno tablet?",
        "answer": "Tecno POP 7 Pro"
    },
    {
        "context_index": 12,
        "question": "What is the color option available for the Tecno POP 7 Pro?",
        "answer": "Uyuni Blue"
    },
    {
        "context_index": 12,
        "question": "How much RAM does the Tecno POP 7 Pro have?",
        "answer": "3GB"
    },
    {
        "context_index": 12,
        "question": "What is the storage capacity of the Tecno POP 7 Pro?",
        "answer": "64GB"
    },
    {
        "context_index": 12,
        "question": "What type of port does the Tecno POP 7 Pro have?",
        "answer": "Type C Port"
    },
    {
        "context_index": 12,
        "question": "What is the resolution of the dual camera on the Tecno POP 7 Pro?",
        "answer": "12MP"
    },
    {
        "context_index": 12,
        "question": "How much RAM can the Tecno POP 7 Pro have with Memory Fusion?",
        "answer": "Up to 6GB RAM with Memory Fusion"
    }
]


train_data = []
contexts_data = []

for i, context in enumerate(contexts):
    qas = []
    for qa in questions_answers:
        if qa["context_index"] == i:
            answer_start = context.find(qa["answer"])
            if answer_start != -1:
                qas.append({
                    "id": str(len(qas) + 1).zfill(5),
                    "is_impossible": False,
                    "question": qa["question"],
                    "answers": [
                        {
                            "text": qa["answer"],
                            "answer_start": answer_start,
                        }
                    ],
                })
    contexts_data.append({
        "context": context,
        "qas": qas,
    })

train_data.extend(contexts_data)

# Print the train_data to verify the format
import json
# print(json.dumps(train_data, indent=4))

with open('amazon_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)


    
test_data = []
contexts_data = []

for i, context in enumerate(contexts):
    qas = []
    for qa in questions_answers:
        if qa["context_index"] == i:
            answer_start = context.find(qa["answer"])
            if answer_start != -1:
                qas.append({
                    "id": str(len(qas) + 1).zfill(5),
                    "is_impossible": False,
                    "question": qa["question"],
                    "answers": [
                        {
                            "text": qa["answer"],
                            "answer_start": answer_start,
                        }
                    ],
                })
    contexts_data.append({
        "context": context,
        "qas": qas,
    })

test_data.extend(contexts_data)

# Print the train_data to verify the format
import json
# print(json.dumps(train_data, indent=4))

with open('amazon_data_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)


# Import required libraries to fine tune transformer models lik BERT
import simpletransformers
import json
import logging
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs




# Read dataset to fine tune BERT model

# Training data
with open(r"amazon_data_train.json", "r") as read_file:
    train = json.load(read_file)

# Validation dataset
with open(r"amazon_data_test.json", "r") as read_file:
    test = json.load(read_file)



model_type="bert"
model_name= "bert-base-cased"

# Create folder to save fine tuned bert model inside working directory

import os
output_dir = 'bert_outputs'
os.mkdir(output_dir)

# Define transformer model arguments before training the BERT model

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"{output_dir}/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 30,
    "evaluate_during_training_steps": 1000,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size":8,
    "train_batch_size": 8,
    "eval_batch_size": 8
}

model = QuestionAnsweringModel(model_type,model_name, args=train_args, use_cuda=False)

# Train the model
model.train_model(train, eval_data=test)



# Streamlit app layout
st.title('PDF Text Extractor')

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Display uploaded file content (optional)
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.write("**Uploaded PDF Content:**")
    st.write(bytes_data.decode('utf-8'))

question = st.text_input("Enter your question:")
# Button to trigger processing (optional)
if st.button('Process PDF'):

    # Process logic using the uploaded file (replace with your implementation)
     pdf_reader = PdfReader(uploaded_file)
     text = ""
     for page in pdf_reader.pages:
         text += page.extract_text()
     answer = answer_question(question, text)
     st.write("**Answer:**", answer)