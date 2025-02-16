import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_excel_files_from_folder(folder_path):
    excel_files = [
        f for f in os.listdir(folder_path) if f.endswith(".xlsx") or f.endswith(".xls")
    ]
    data_frames = []

    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        # "no content"가 포함된 행을 제외
        df_filtered = df[
            ~df.apply(
                lambda row: row.astype(str).str.contains("No Content").any(), axis=1
            )
        ]
        data_frames.append(df_filtered)

    # 모든 데이터프레임을 하나로 결합
    combined_df = pd.concat(data_frames, ignore_index=True)
    # 결합된 데이터프레임을 엑셀 파일로 저장
    combined_df.to_excel("Ophthalmology_PubMed_refined.xlsx", index=False)

    return combined_df

print(load_excel_files_from_folder("./Refined Ophthalmology PubMed"))
