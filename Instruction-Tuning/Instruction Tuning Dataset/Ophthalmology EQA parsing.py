import re
import pandas as pd
import os

# 현재 파일의 경로를 가져옴
current_file_path = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리 경로를 가져옴
current_dir = os.path.dirname(current_file_path)
# 해당 디렉토리로 작업 디렉토리 변경
os.chdir(current_dir)

def parse_qa_pairs(text):
    # Regular expression to find [Q] (question) and [A] (answer) blocks
    qa_pairs = re.findall(r'\[Q\]: (.*?)\[A\]: (.*?)(?=\[Q\]:|\Z)', text, re.DOTALL)
    # Strip extra spaces and format as a list of dictionaries
    qa_list = [{'question': q.strip(), 'answer': a.strip()} for q, a in qa_pairs]
    return qa_list


def parse_qa_pairs_and_save_to_excel(input_file, column_name, output_file):
    # Step 1: 엑셀 파일에서 데이터 불러오기
    df = pd.read_excel(input_file)

    # Step 2: 특정 열(column_name)에서 데이터를 읽어오기
    text_data = df[column_name].dropna().tolist()  # 결측값 제거하고 리스트로 변환

    # Step 3: 모든 텍스트에 대해 parse_qa_pairs 적용
    all_qa_pairs = []
    for text in text_data:
        qa_pairs = parse_qa_pairs(text)
        # "It is not related to ophthalmology."가 아닌 question/answer이고 "et al"이 포함되지 않은 것만 저장
        filtered_qa_pairs = [
            qa for qa in qa_pairs
            if qa['question'] != "It is not related to ophthalmology."
               and "et al" not in qa['question']
               and "et al" not in qa['answer']
        ]
        all_qa_pairs.extend(filtered_qa_pairs)  # 여러 줄의 질문-답변 쌍을 합침

    # Step 4: 결과를 새로운 데이터프레임으로 변환
    if all_qa_pairs:  # 결과가 있을 때만 저장
        qa_df = pd.DataFrame(all_qa_pairs)

        # Step 5: 결과를 새로운 엑셀 파일로 저장
        qa_df.to_excel(output_file, index=False)
    else:
        print("No valid Q&A pairs to save.")


# 사용 예시
input_excel_file = './Generated Ophthalmology QA/Generated Ophthalmology QA.xlsx'  # 입력 파일 경로
output_excel_file = './Parsed Ophthalmology QA/Parsed Ophthalmology QA.xlsx'  # 출력 파일 경로
column_to_parse = 'answer'  # 열 이름

parse_qa_pairs_and_save_to_excel(input_excel_file, column_to_parse, output_excel_file)


