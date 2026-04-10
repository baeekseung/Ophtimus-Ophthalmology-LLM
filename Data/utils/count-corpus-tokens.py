from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

def count_corpus_tokens(corpus: str, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> int:
    """코퍼스 문자열의 전체 토큰 수를 반환합니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(corpus, add_special_tokens=False, truncation=False)
    return len(tokens["input_ids"])


if __name__ == "__main__":
    corpus = "Hello, world!"
    print(count_corpus_tokens(corpus))