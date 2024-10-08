# preprocess_pdfs.py

import os
import json
import re
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

# Use parameters from config.py
input_folder = config.INPUT_PDF_FOLDER
output_folder = config.OUTPUT_JSON_FOLDER
chunk_size = config.CHUNK_SIZE
chunk_overlap = config.CHUNK_OVERLAP

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Data cleaning function
def clean_text(text):
    # 1. Merge words separated by hyphens and line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 2. Replace line breaks with spaces
    text = text.replace('\n', ' ')
    # 3. Remove citations (e.g., [1], (1), etc.)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Remove content after "References"
def remove_references(text):
    # Define keywords and look for them at the start of a line
    reference_keywords = ['references', 'bibliography']
    pattern = re.compile(r'^\s*(references|bibliography)\b', re.IGNORECASE | re.MULTILINE)
    
    match = pattern.search(text)
    if match:
        text = text[:match.start()]
    
    return text

# Extract main and section titles
def extract_titles(raw_text):
    main_title = raw_text.split("\n", 1)[0].strip()
    section_pattern = re.compile(r"^(?:\d+\.?|\d+\.\d+|\b[A-Z]+\b)\s+[A-Za-z]", re.MULTILINE)
    section_titles = section_pattern.findall(raw_text)
    return main_title, section_titles

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

# Get all PDF files in input folder
pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

# Process PDFs with progress bar
for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    file_path = os.path.join(input_folder, pdf_file)
    loader_py = PyPDFLoader(file_path)
    pages_py = loader_py.load()
    pages_content_py = [page.page_content for page in pages_py]
    raw_text = "\n".join(pages_content_py)
    main_title, section_titles = extract_titles(raw_text)
    text_no_references = remove_references(raw_text)
    chunks = text_splitter.split_text(text_no_references)
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    chunked_data = [
        {
            "main_title": main_title,
            "chunk_index": idx + 1,
            "chunk_length": len(chunk),
            "chunk_content": chunk
        }
        for idx, chunk in enumerate(cleaned_chunks)
    ]
    output_json_path_py = os.path.join(output_folder, pdf_file.replace(".pdf", "_chunked.json"))
    with open(output_json_path_py, "w", encoding="utf-8") as json_file:
        json.dump(chunked_data, json_file, ensure_ascii=False, indent=4)
