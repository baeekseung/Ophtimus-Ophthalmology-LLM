import re
import pandas as pd
import os

# 현재 파일의 경로를 가져옴
current_file_path = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리 경로를 가져옴
current_dir = os.path.dirname(current_file_path)
# 해당 디렉토리로 작업 디렉토리 변경
os.chdir(current_dir)


def parse_mcqa(text):
    # Regular expression to find [Ques], [a], [b], [c], [d], [e], [Ans], [Exp] blocks
    pattern = (
        r"\[Ques\]: (.*?)"
        r"(?:\[a\]: (.*?))?"
        r"(?:\[b\]: (.*?))?"
        r"(?:\[c\]: (.*?))?"
        r"(?:\[d\]: (.*?))?"
        r"(?:\[e\]: (.*?))?"
        r"\[Ans\]: (.*?)"
        r"\[Exp\]: (.*?)(?=\[Ques\]:|\Z)"
    )
    matches = re.findall(pattern, text, re.DOTALL)

    # Strip extra spaces and format as a list of dictionaries
    mcqa_list = [
        {
            "question": q.strip(),
            "a": a.strip() if a else "",
            "b": b.strip() if b else "",
            "c": c.strip() if c else "",
            "d": d.strip() if d else "",
            "e": e.strip() if e else "",
            "answer": ans.strip(),
            "explanation": exp.strip(),
        }
        for q, a, b, c, d, e, ans, exp in matches
    ]
    return mcqa_list


def parse_mcqa_and_save_to_excel(input_file, column_name, output_file):
    # Step 1: 엑셀 파일에서 데이터 불러오기
    df = pd.read_excel(input_file)

    # Step 2: 특정 열(column_name)에서 데이터를 읽어오기
    text_data = df[column_name].dropna().tolist()  # 결측값 제거하고 리스트로 변환

    # Step 3: 모든 텍스트에 대해 parse_mcqa 적용
    all_mcqa_pairs = []
    for text in text_data:
        mcqa_pairs = parse_mcqa(text)
        all_mcqa_pairs.extend(mcqa_pairs)  # 여러 줄의 질문-답변 쌍을 합침

    # Step 4: 결과를 새로운 데이터프레임으로 변환
    if all_mcqa_pairs:  # 결과가 있을 때만 저장
        mcqa_df = pd.DataFrame(all_mcqa_pairs)

        # Step 5: 결과를 새로운 엑셀 파일로 저장
        mcqa_df.to_excel(output_file, index=False)
    else:
        print("No valid MCQA pairs to save.")


# 사용 예시
input_excel_file = (
    "./Generated Ophthalmology QA/Generated Ophthalmology MCQA_0_3430.xlsx"  # 입력 파일 경로
)
output_excel_file = (
    "./Parsed Ophthalmology QA/Parsed Ophthalmology MCQA_0_3430.xlsx"  # 출력 파일 경로
)
column_to_parse = "answer"  # 열 이름

parse_mcqa_and_save_to_excel(input_excel_file, column_to_parse, output_excel_file)
