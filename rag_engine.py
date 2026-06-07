import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    INDEX_PATH,
    CHUNKS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    RERANK_MODEL_NAME,
    RETRIEVE_TOP_K,
    FINAL_TOP_K,
    MIN_RELEVANCE_SCORE

)
from loaders import load_documents


class RAGEngine:    
    #def __init__(self):
        # self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # self.generator = pipeline(
        #     "text2text-generation",
        #     model=GENERATION_MODEL_NAME
        # )
        # self.index = None
        # self.chunks = []
        
    def __init__(self):
        self.embedding_model = None
        # self.generator = None
        self.reranker = None
        self.gen_model = None
        self.tokenizer = None
        self.index = None
        self.chunks = []

    def load_models(self):
        if self.embedding_model is None:
         self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        if self.reranker is None:
         self.reranker = CrossEncoder(RERANK_MODEL_NAME)

        if self.tokenizer is None:
         self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)

        if self.gen_model is None:
         self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        text = text.replace("\r", "\n")
        chunks = []

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += max(1, chunk_size - overlap)

        return chunks

    def build_knowledge_base(self, data_dir: str):
        documents = load_documents(data_dir)
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": chunk
                })

        if not all_chunks:
            raise ValueError("Nenhum conteúdo válido foi encontrado nos arquivos.")

        embeddings = self.embedding_model.encode(
            [c["text"] for c in all_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        self.index = index
        self.chunks = all_chunks

        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_knowledge_base(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError("Índice vetorial não encontrado. Gere a base primeiro.")

        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
    )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if self.index is None:
            raise ValueError("Índice não carregado.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                item = dict(self.chunks[idx])
                item["score"] = float(dist)
                results.append(item)

        return results

    def retrieve_candidates(self, query: str, top_k: int = RETRIEVE_TOP_K) -> List[Dict]:
        if self.index is None:
         raise ValueError("Índice não carregado.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                item = dict(self.chunks[idx])
                item["vector_score"] = float(dist)
                results.append(item)

        return results

    def rerank(self, query: str, candidates: List[Dict], final_top_k: int = FINAL_TOP_K) -> List[Dict]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        filtered = [r for r in reranked if r["rerank_score"] >= MIN_RELEVANCE_SCORE]
        return filtered[:final_top_k]
        
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(
            [f"Fonte: {c['source']}\nTrecho: {c['text']}" for c in contexts]
        )

        prompt = f"""
Você é um assistente que responde perguntas com base apenas no contexto fornecido.

Regras:
- Responda em português.
- Seja objetivo e fiel ao conteúdo.
- Se a resposta não estiver no contexto, diga claramente que a informação não foi encontrada nos arquivos.
- Sempre que possível, cite a fonte do trecho usado.
- responda apenas com evidências explícitas do contexto, sem inferências ou suposições.

Contexto:
{context_text}

Pergunta:
{query}

Resposta:
"""
        return prompt.strip()

    def answer(self, query: str) -> Tuple[str, List[Dict]]:
        candidates = self.retrieve_candidates(query=query, top_k=RETRIEVE_TOP_K)
        contexts = self.rerank(query=query, candidates=candidates, final_top_k=FINAL_TOP_K)

        if not contexts:
            return "Não encontrei trechos suficientemente relevantes nos arquivos para responder com segurança.", []

        prompt = self.build_prompt(query, contexts)
        response = self.generate_text(prompt)

        return response, contexts