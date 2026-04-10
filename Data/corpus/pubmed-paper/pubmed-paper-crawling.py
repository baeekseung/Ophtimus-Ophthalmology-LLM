import os
import io
import time
import tarfile
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_API    = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
ICITE_API = "https://icite.od.nih.gov/api/pubs"
SLEEP     = 0.35   # NCBI 속도 제한: 초당 3회 이하


# PMC 검색 (오픈소스 논문만 검색하기 위해서 PMC에서 1차 검색)
def search_pmc(keyword: str, max_results: int = 500) -> list:
    """
    PMC에서 키워드로 오픈액세스 논문을 검색합니다.
    relevance(관련도) 정렬로 PMID가 있는 논문만 반환됩니다.

    반환값: PMC ID 목록 (예: ['11529683', '10948212', ...])
    """
    params = {
        "db": "pmc",
        "term": keyword,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        response = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
        response.raise_for_status()
        return response.json()["esearchresult"]["idlist"]
    except Exception as e:
        print(f"[검색 오류] {e}")
        return []


# PMC ID → PubMed ID 변환
def get_pmc_to_pmid_map(pmc_ids: list) -> dict:
    """
    PMC ID를 PubMed ID로 변환합니다. (인용수 조회에 PubMed ID가 필요)
    200개씩 배치로 처리합니다.

    반환값: {pmc_id: pmid} 딕셔너리
    """
    result = {}
    batch_size = 200

    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i : i + batch_size]
        params = {
            "db":      "pmc",
            "id":      ",".join(batch),
            "retmode": "json",
        }
        try:
            response = requests.get(f"{NCBI_BASE}/esummary.fcgi", params=params, timeout=30)
            response.raise_for_status()
            data = response.json().get("result", {})

            for pmc_id in batch:
                article_ids = data.get(pmc_id, {}).get("articleids", [])
                # articleids 목록에서 pmid 타입을 찾습니다
                for id_info in article_ids:
                    if id_info.get("idtype") == "pmid" and id_info.get("value"):
                        result[pmc_id] = id_info["value"]
                        break
        except Exception as e:
            print(f"[ID 변환 오류] {e}")

        time.sleep(SLEEP)

    return result


# 인용수 조회
def get_citation_counts(pmids: list) -> dict:
    """
    iCite API로 PubMed 논문들의 인용수를 조회합니다.
    200개씩 배치로 처리합니다.

    반환값: {pmid: 인용수} 딕셔너리
    """
    result = {}
    batch_size = 200

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {"pmids": ",".join(batch)}
        try:
            response = requests.get(ICITE_API, params=params, timeout=30)
            response.raise_for_status()
            for item in response.json().get("data", []):
                pmid = str(item.get("pmid", ""))
                # citation_count가 None인 경우 0으로 처리
                count = item.get("citation_count") or 0
                result[pmid] = count
        except Exception as e:
            print(f"[인용수 조회 오류] {e}")

        time.sleep(SLEEP)

    return result


# PDF URL 조회
def get_pdf_url(pmc_id: str):
    """
    PMC OA API로 논문의 PDF 다운로드 URL을 조회합니다.
    ftp:// URL을 https://로 변환하여 반환합니다.

    반환값: (url, format) 또는 (None, None)
      - format은 'pdf' 또는 'tgz'
    """
    try:
        response = requests.get(OA_API, params={"id": f"PMC{pmc_id}"}, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)

        # 오픈액세스가 아닌 경우 오류 태그가 있음
        if root.find(".//error") is not None:
            return None, None

        pdf_url = None
        tgz_url = None

        # <link> 태그에서 PDF 또는 TGZ URL 추출
        for link in root.findall(".//link"):
            fmt = link.get("format", "")
            href = link.get("href", "").replace(
                "ftp://ftp.ncbi.nlm.nih.gov",
                "https://ftp.ncbi.nlm.nih.gov"
            )
            if fmt == "pdf" and not pdf_url:
                pdf_url = href
            elif fmt == "tgz" and not tgz_url:
                tgz_url = href

        # PDF 직접 링크 우선, 없으면 TGZ 아카이브 사용
        if pdf_url:
            return pdf_url, "pdf"
        if tgz_url:
            return tgz_url, "tgz"

    except Exception as e:
        print(f"  [URL 조회 오류] PMC{pmc_id}: {e}")

    return None, None


# PDF 다운로드
def download_pdf(url: str, fmt: str, save_path: str) -> bool:
    """
    PDF 파일을 다운로드하여 저장합니다.
    - fmt=='pdf': 직접 다운로드
    - fmt=='tgz': 아카이브에서 PDF 파일을 추출

    반환값: 성공 시 True, 실패 시 False
    """
    try:
        timeout = 60 if fmt == "tgz" else 30
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        if fmt == "pdf":
            # PDF 파일인지 확인 (헤더 체크)
            if not response.content.startswith(b"%PDF"):
                return False
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True

        elif fmt == "tgz":
            # TGZ 아카이브에서 PDF 파일 추출
            archive = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
            pdf_member = next(
                (m for m in archive.getmembers() if m.name.endswith(".pdf")),
                None
            )
            if pdf_member is None:
                return False
            pdf_data = archive.extractfile(pdf_member).read()
            with open(save_path, "wb") as f:
                f.write(pdf_data)
            return True

    except Exception as e:
        print(f"  [다운로드 오류] {e}")
        # 실패한 경우 불완전한 파일 삭제
        if os.path.exists(save_path):
            os.remove(save_path)

    return False



def main(keyword: str, k: int, output_dir: str):
    """
    전체 파이프라인:
    1. PMC 검색 → 2. ID 변환 → 3. 인용수 조회 → 4. 정렬 → 5. PDF 다운로드
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. PMC에서 논문 검색
    print(f"\n[1/4] '{keyword}' 키워드로 PMC 검색 중...")
    pmc_ids = search_pmc(keyword, max_results=500)
    print(f"  → {len(pmc_ids)}개 논문 발견")

    if not pmc_ids:
        print("검색 결과가 없습니다.")
        return

    # 2. PMC ID → PubMed ID 변환
    print("[2/4] PubMed ID 변환 중...")
    pmc_to_pmid = get_pmc_to_pmid_map(pmc_ids)
    print(f"  → {len(pmc_to_pmid)}개 PMID 확인")

    # 3. 인용수 조회
    print("[3/4] 인용수 조회 중...")
    pmids = list(pmc_to_pmid.values())
    citation_counts = get_citation_counts(pmids)

    # 4. 인용수 기준 내림차순 정렬
    print("[4/4] 인용수 기준으로 정렬 후 PDF 다운로드 시작...")
    scored = []
    for pmc_id in pmc_ids:
        pmid = pmc_to_pmid.get(pmc_id)
        count = citation_counts.get(pmid, 0) if pmid else 0
        scored.append((pmc_id, count))
    scored.sort(key=lambda x: x[1], reverse=True)

    # 5. 상위 논문부터 k개 다운로드
    downloaded = 0
    tried = 0

    for pmc_id, citations in tqdm(scored, desc="다운로드 진행"):
        if downloaded >= k:
            break

        tried += 1
        save_path = os.path.join(output_dir, f"{pmc_id}.pdf")

        # 이미 다운로드된 파일은 스킵
        if os.path.exists(save_path):
            print(f"  PMC{pmc_id}: 이미 존재 (스킵)")
            downloaded += 1
            continue

        # PDF URL 조회
        url, fmt = get_pdf_url(pmc_id)
        time.sleep(SLEEP)

        if url is None:
            continue  # 오픈액세스 아님 → 다음 논문

        # 다운로드 시도
        success = download_pdf(url, fmt, save_path)
        if success:
            downloaded += 1
            print(f"  PMC{pmc_id}: 저장 완료 ({citations}회 인용, {fmt})")
        else:
            print(f"  PMC{pmc_id}: 다운로드 실패 → 다음 논문으로")

    print(f"\n완료: {downloaded}/{k}개 다운로드 (총 {tried}개 시도)")
    print(f"저장 경로: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    KEYWORD = "diabetic retinopathy"   # 검색 키워드
    K = 5   # 다운로드할 논문 수
    OUTPUT_DIR = "Data/corpus/Data/Documents"  # 저장 폴더

    main(KEYWORD, K, OUTPUT_DIR)
