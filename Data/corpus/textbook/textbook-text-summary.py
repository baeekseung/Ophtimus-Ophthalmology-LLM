"""
안과 교과서 텍스트 정형화 파이프라인

loaded-textbooks.xlsx를 읽어 각 페이지를 LLM으로 정형화합니다.
- 빈 페이지(contents가 공백)는 건너뜁니다.
- 각 페이지를 Stuff 방식으로 개별 처리합니다.

결과는 formatted-textbooks.xlsx로 저장됩니다.
"""

import importlib.util
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# 현재 파일 위치를 작업 디렉토리로 설정
current_file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_file_path))

# 프로젝트 루트 및 utils 경로 추가
# textbook/ → corpus/ → Data/ → 프로젝트 루트 (3단계 상위)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "../../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data", "utils"))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 하이픈이 포함된 파일명은 importlib으로 동적 임포트
_token_counter_path = os.path.join(PROJECT_ROOT, "Data", "utils", "count-corpus-tokens.py")
_spec = importlib.util.spec_from_file_location("count_corpus_tokens", _token_counter_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
count_corpus_tokens = _module.count_corpus_tokens

from summarize_utils import textbook_page_format

# ────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────

INPUT_EXCEL_PATH = "./loaded-textbooks2.xlsx"
OUTPUT_EXCEL_PATH = "./formatted-textbooks.xlsx"

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0


# ────────────────────────────────────────────────────────────────────
# 메인 로직
# ────────────────────────────────────────────────────────────────────

def format_page(page_text: str, llm: ChatOpenAI) -> str:
    """
    단일 페이지 텍스트를 LLM으로 정형화합니다.
    빈 페이지는 호출 전에 필터링되어야 합니다.

    Args:
        page_text: 페이지 원문 텍스트
        llm: ChatOpenAI 인스턴스

    Returns:
        정제된 자연어 산문 텍스트
    """
    return textbook_page_format(page_text, llm)


def format_textbook(file_name: str, pages_df: pd.DataFrame, llm: ChatOpenAI) -> list[dict]:
    """
    단일 교과서의 모든 페이지를 순서대로 정형화합니다.

    Args:
        file_name: 교과서 파일명
        pages_df: 해당 교과서의 페이지별 DataFrame (page, contents 컬럼 포함)
        llm: ChatOpenAI 인스턴스

    Returns:
        정형화 결과 딕셔너리 리스트 [{file, page, contents, formatted}, ...]
    """
    pages_df = pages_df.sort_values("page").reset_index(drop=True)

    # 빈 페이지 필터링
    non_empty = pages_df[pages_df["contents"].str.strip() != ""]
    skipped = len(pages_df) - len(non_empty)
    print(f"  전체 {len(pages_df)}페이지 중 빈 페이지 {skipped}개 제외 → {len(non_empty)}페이지 처리")

    results = []
    for _, row in tqdm(non_empty.iterrows(), total=len(non_empty), desc=f"  {file_name}", unit="page"):
        try:
            formatted = format_page(row["contents"], llm)
        except Exception as e:
            print(f"\n  [오류] {file_name} p.{row['page']}: {e}")
            formatted = f"오류: {e}"

        results.append({
            "file": file_name,
            "page": row["page"],
            "contents": row["contents"],
            "formatted": formatted,
        })

    return results


def main():
    # LLM 초기화
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # 로드된 교과서 텍스트 불러오기
    df = pd.read_excel(INPUT_EXCEL_PATH)
    files = df["file"].unique()
    print(f"총 {len(files)}개 교과서 정형화 시작\n{'=' * 50}")

    all_results = []
    for idx, file_name in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {file_name} 처리 중...")
        pages_df = df[df["file"] == file_name]

        results = format_textbook(file_name, pages_df, llm)
        all_results.extend(results)

        # 교과서별 중간 결과 통계
        no_content = sum(1 for r in results if r["formatted"].strip() == "콘텐츠 없음")
        print(f"  완료: {len(results)}페이지 처리 (콘텐츠 없음: {no_content}페이지)")

    # 결과 저장
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print(f"\n{'=' * 50}")
    print(f"완료: {OUTPUT_EXCEL_PATH}")
    print(f"  처리 파일 수  : {result_df['file'].nunique()}개")
    print(f"  총 처리 페이지: {len(result_df)}페이지")
    no_content_total = (result_df["formatted"].str.strip() == "콘텐츠 없음").sum()
    print(f"  콘텐츠 없음   : {no_content_total}페이지")
    error_total = result_df["formatted"].str.startswith("오류:").sum()
    if error_total > 0:
        print(f"  오류 발생     : {error_total}페이지")


if __name__ == "__main__":
    main()
