# DoziScholar

DoziScholar is an open-source, research-grade Question-Answering tool for extracting precise answers from academic articles (PDFs) in both English and Persian (Farsi).  
Bring your own LLM API (OpenAI, OpenRouter, DeepSeek, etc.), and get referenced answers to your questions — with full multilingual support.

---

## Features

- **PDF Question Answering** (English & Persian)
- **Semantic Search** with Sentence Transformers + FAISS
- **Pluggable LLM API** (OpenRouter, OpenAI, ... — just add your API key)
- **Citations**: Answers always reference source page & section
- **Multilingual Pipeline**: Persian queries & answers handled automatically via integrated translation module

---

## How it works

1. **Load PDF** ➔ Split into overlapping chunks  
2. **Build Embedding Index**  
3. **Retrieve Most Relevant Chunks for Question**  
4. **Prompt LLM with Question & Chunks**  
5. **Return Answer + References**  
6. (If Persian) Auto-translate Q&A for user

---

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/DoziScholar.git
cd DoziScholar
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

On first run, you will be prompted for:

- **Your LLM API key** (e.g. from OpenRouter)
- **Model name** (e.g. `openrouter/cypher-alpha:free`)
- **API URL**

Or edit them in `config.py` for easier use.

**Run:**

```bash
python3 Sin.py
```
---
## Why DoziScholar?

Fully open-source, easy to extend
Works with any LLM API, not vendor-locked
Accurate, referenced, academic answers
Best-in-class Farsi support

---
## Roadmap

 English/Persian PDF QA
 Telegram & Android bot
 Web UI
 Support for DOCX/HTML
 Answer evaluation metrics
 
---
## Contributing

 Pull requests are welcome!
 Want to add a feature or connect a new API? Open an issue or PR.
 
---
Developed by [Amirmahdi Ebrahimi]

Contact: [amirsbus1@gmail.com]


