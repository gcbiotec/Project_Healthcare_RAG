#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

#GENERATION_MODEL_NAME = "google/flan-t5-base"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

#RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DATA_DIR = "data/arquivos_base"
INDEX_PATH = "storage/faiss_index.bin"
CHUNKS_PATH = "storage/chunks.pkl"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1

RETRIEVE_TOP_K = 10
FINAL_TOP_K = 4
MIN_RELEVANCE_SCORE = 0.0