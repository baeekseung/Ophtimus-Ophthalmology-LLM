"""
docs 리스트를 로컬에 저장하고 로드하는 유틸리티 함수들
"""
import os
import pickle
import json
from datetime import datetime
from typing import List, Optional, Any

def save_docs_to_local(docs: List[Any], save_dir: str = "./Ophtimus-RAG-System/saved_docs", filename: Optional[str] = None) -> Optional[str]:

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"docs_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(docs, f)
        print(f"문서가 성공적으로 저장되었습니다: {filepath}")
        print(f"저장된 문서 수: {len(docs)}")
        return filepath
    except Exception as e:
        print(f"문서 저장 중 오류 발생: {e}")
        return None
    

def load_docs_from_local(filepath: str) -> Optional[List[Any]]:
    try:
        with open(filepath, 'rb') as f:
            docs = pickle.load(f)
        print(f"문서가 성공적으로 로드되었습니다: {filepath}")
        print(f"로드된 문서 수: {len(docs)}")
        return docs
    except Exception as e:
        print(f"문서 로드 중 오류 발생: {e}")
        return None
