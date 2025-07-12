import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Load and split PDF into chunks
pdf_path = "test.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks, metadatas = [], []
for page in pages:
    page_num = page.metadata.get("page")
    for idx, text in enumerate(text_splitter.split_text(page.page_content)):
        chunks.append(text)
        metadatas.append({"page": page_num, "chunk_index": idx})

# 2. Create embeddings and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunks, show_progress_bar=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 3. Retrieval function
def retrieve_chunks(question: str, top_k: int = 3) -> list:
    q_emb = embedding_model.encode([question])
    distances, indices = index.search(np.array(q_emb), k=top_k)
    return [
        {
            "text": chunks[i],
            "page": metadatas[i]["page"],
            "chunk_index": metadatas[i]["chunk_index"],
            "score": float(dist)
        }
        for dist, i in zip(distances[0], indices[0])
    ]

# 4. Prompt template 
custom_template = """
Answer the following question ONLY using the provided context below.
If the answer is not found in the context, say 'Not found in context.'
Cite all supporting text as [Page X, Chunk Y].

Context:
{summaries}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=custom_template,
    input_variables=["summaries", "question"]
)

def build_context(retrieved: list) -> str:
    return "\n\n".join([
        f"[Page {c['page']}, Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved
    ])

# 5. OpenRouter LLM integration با فیلد Pydantic به‌جای __init__
class OpenRouterLLM(LLM):
    api_key: str
    model_name: str = "openrouter/cypher-alpha:free"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "openrouter"

# 6. End-to-end execution
def answer_question(question: str, top_k: int = 3) -> str:
    retrieved = retrieve_chunks(question, top_k)
    context = build_context(retrieved)

    llm = OpenRouterLLM(
        api_key="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"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(summaries=context, question=question)


class TranslateLLM(LLM):
    api_key: str
    model_name: str
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "openrouter"


def translate_fa2en(text):
    llm = TranslateLLM(api_key="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", model_name="deepseek/deepseek-r1-0528:free")

    prompt = f"""Translate the following Persian text to English. Only output the English translation, with no extra explanation.

    Persian: {text}
    English:"""

    response = llm(prompt)
    # Strip whitespace, just in case
    return response.strip()

def translate_en2fa(text):
    llm = TranslateLLM(api_key="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", model_name="deepseek/deepseek-r1-0528:free")

    prompt = f"""Translate the following English text to Persian. Only output the Persian translation, with no extra explanation.

       English: {text}
       Persian:"""

    response = llm(prompt)
    # Strip whitespace, just in case
    return response.strip()



if __name__ == "__main__":
    while True:
        q = input("سؤال را وارد کنید (یا '1' برای خروج): ").strip()
        if q == "1":
            print("خروج از برنامه.")
            break
        q_new = translate_fa2en(q)
        print(f"Processing: {q_new}")
        pred = answer_question(q, top_k=3)
        pred_new = translate_en2fa(pred)
        print("Answer:", pred_new)