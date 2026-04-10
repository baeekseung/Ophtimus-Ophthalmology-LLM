"""
PubMed 논문 요약 및 정형화 파이프라인

parsed-papers.xlsx를 읽어 각 논문을 요약합니다.
- context length ≤ 70K 토큰: Stuff 방식 (전체 텍스트 한 번에 요약)
- context length > 70K 토큰: Map-Reduce 방식 (페이지별 요약 후 통합)

결과는 summarized-papers.xlsx로 저장됩니다.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 현재 파일 위치를 작업 디렉토리로 설정
current_file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_file_path))

# 프로젝트 루트의 utils 경로 추가
# pubmed-paper/ → corpus/ → Data/ → OPHTIMUS/ (3단계 상위)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "../../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data", "utils"))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import importlib.util

# 하이픈이 포함된 파일명은 importlib으로 동적 임포트
_token_counter_path = os.path.join(PROJECT_ROOT, "Data", "utils", "count-corpus-tokens.py")
_spec = importlib.util.spec_from_file_location("count_corpus_tokens", _token_counter_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
count_corpus_tokens = _module.count_corpus_tokens

from summarize_utils import stuff_summarize, map_reduce_summarize  # noqa: E402

# ────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────

INPUT_EXCEL_PATH = "./parsed-papers.xlsx"
OUTPUT_EXCEL_PATH = "./summarized-papers.xlsx"

TOKEN_THRESHOLD = 70_000  # Stuff vs Map-Reduce 분기 기준

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0


# ────────────────────────────────────────────────────────────────────
# 메인 로직
# ────────────────────────────────────────────────────────────────────

def summarize_paper(
    file_name: str,
    pages_df: pd.DataFrame,
    llm: ChatOpenAI,
) -> dict:
    """
    단일 논문의 모든 페이지를 요약하고 결과를 반환합니다.

    Args:
        file_name: 논문 파일명
        pages_df: 해당 논문의 페이지별 DataFrame (page, contents 컬럼 포함)
        llm: ChatOpenAI 인스턴스

    Returns:
        요약 결과 딕셔너리 {file, token_count, method, summary}
    """
    # 페이지 순서 정렬 후 텍스트 추출
    pages_df = pages_df.sort_values("page")
    pages = pages_df["contents"].tolist()
    full_text = "\n\n\n".join(pages)

    # 토큰 수 측정
    token_count = count_corpus_tokens(full_text)
    print(f"  토큰 수: {token_count:,}")

    # 요약 방식 결정 및 실행
    if token_count <= TOKEN_THRESHOLD:
        method = "stuff"
        print(f"  방식: Stuff (≤ {TOKEN_THRESHOLD:,} 토큰)")
        summary = stuff_summarize(full_text, llm)
    else:
        method = "map_reduce"
        print(f"  방식: Map-Reduce (> {TOKEN_THRESHOLD:,} 토큰, {len(pages)}페이지)")
        summary = map_reduce_summarize(pages, llm)

    return {
        "file": file_name,
        "token_count": token_count,
        "method": method,
        "summary": summary,
    }


def main():
    # LLM 초기화
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # 파싱된 논문 로드
    df = pd.read_excel(INPUT_EXCEL_PATH)
    files = df["file"].unique()
    print(f"총 {len(files)}개 논문 요약 시작\n{'=' * 50}")

    results = []
    for idx, file_name in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {file_name} 처리 중...")
        pages_df = df[df["file"] == file_name]

        try:
            result = summarize_paper(file_name, pages_df, llm)
            results.append(result)
            print(f"  완료")
        except Exception as e:
            print(f"  오류 발생: {e}")
            results.append({
                "file": file_name,
                "token_count": -1,
                "method": "error",
                "summary": f"오류: {e}",
            })

    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\n{'=' * 50}")
    print(f"완료: {OUTPUT_EXCEL_PATH} ({len(result_df)}건 저장)")

    # 요약 통계 출력
    method_counts = result_df["method"].value_counts()
    print("\n[요약 방식 통계]")
    for method, count in method_counts.items():
        print(f"  {method}: {count}건")


if __name__ == "__main__":
    main()
