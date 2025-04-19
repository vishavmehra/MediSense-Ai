# MediSense AI

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Demo](#demo)
- [Author](#author)
- [License](#license)

## Project Overview
Llama-MD is a modular framework for fine-tuning, augmenting, and deploying a large language model with Retrieval-Augmented Generation (RAG) capabilities, tailored to the medical domain (Pregnancy and Gynecology). It includes utilities for PDF ingestion, topic modelling, Wikipedia scraping, embedding generation, vector database management, and a Streamlit-based chat interface.

## Directory Structure
```bash
├── Code
│   ├── data
│   │   ├── pdfreader
│   │   │   └── pdfreader.py         # Convert PDFs to raw text
│   │   ├── topic_modelling
│   │   │   └── topic_modelling.py   # Extract common topics from HF datasets
│   │   ├── wikipedia
│   │   │   └── wiki_scrape.py       # Scrape Wikipedia for RAG context
│   │   └── Raw_Text                 # Output folder for extracted text
│   ├── finetune
│   │   ├── finetune.py              # Fine-tune Llama model
│   │   └── inference.py             # Run inference with the fine-tuned model
│   ├── models                       # Pre-trained model checkpoints
│   ├── rag
│   │   ├── embeddings.py            # Generate document embeddings
│   │   ├── vector_db.py             # Setup and query Pinecone vector DB
│   │   ├── pdf_data_extractor.py    # Embed PDF-derived text into Pinecone
│   │   ├── wiki_data_extractor.py   # Embed Wikipedia text into Pinecone
│   │   ├── test.py                  # RAG end‑to‑end test script
│   │   └── utils.py                 # Helpers for Bedrock or HF calls
│   ├── utils
│   │   └── preprocess.py            # Preprocessing for RAG pipeline
│   ├── .env                         # Environment variables for vector DB, API keys
│   ├── app.py                       # Streamlit chatbot interface
│   ├── ReadMe.md                    # This file
│   └── requirements.txt             # Python dependencies
```

## Setup & Installation
1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/Llama-MD.git
   cd Llama-MD/Code
   ```
2. **Create & activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows PowerShell
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Add environment variables**:
   Create a `.env` file in `Code/`:
   ```ini
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   ```

## Usage
**Fine-tuning**:
```bash
python finetune/finetune.py --data_dir data/Raw_Text --output_dir models/finetuned
```

**Inference (CLI)**:
```bash
python finetune/inference.py --model_dir models/finetuned
```

**RAG Embeddings & Test**:
```bash
python rag/embeddings.py
python rag/vector_db.py
python rag/test.py
```

**Launch Streamlit Chatbot**:
```bash
cd Code
streamlit run app.py --server.port 8888
```  
Then open `http://localhost:8888` in your browser.

## Demo
A short video demonstrating the end‑to‑end workflow (PDF ingestion → RAG QA → chat UI):

[![Watch the demo]](demo.mp4)
https://drive.google.com/file/d/1NurOHUMOyR_VOeHnnPRyD62GcemBqSBh/view?usp=sharing

Alternatively, if your platform supports HTML video:
```html
<video src="assets/demo.mp4" controls width="800" />
```

## Author
- **Name:** Vishav Mehra
- **Email:** mehravishav26@gmail.com
- **GitHub:** [github.com/vishamehra](https://github.com/mehra)
- **LinkedIn:** [linkedin.com/in/vishav-mehra-7551072088](www.linkedin.com/in/vishav-mehra-755107208)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

