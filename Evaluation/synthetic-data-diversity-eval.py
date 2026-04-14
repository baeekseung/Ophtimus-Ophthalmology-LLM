import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환하는 인코더"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_from_csv(data_path: str, instruction_col: str = "instruction", answer_col: str = "answer", max_samples: Optional[int] = None) -> Tuple[list, list]:
    df = pd.read_csv(data_path)

    if instruction_col not in df.columns:
        raise ValueError(f"컬럼 '{instruction_col}'이 CSV에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
    if answer_col not in df.columns:
        raise ValueError(f"컬럼 '{answer_col}'이 CSV에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")

    # 결측값 제거
    df = df.dropna(subset=[instruction_col, answer_col])

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    instructions = df[instruction_col].astype(str).tolist()
    answers = df[answer_col].astype(str).tolist()

    print(f"CSV 로드 완료: {len(instructions)}개 샘플 ({data_path})")
    return instructions, answers


def load_from_hf_dataset(dataset_name: str, instruction_col: str = "instruction", answer_col: str = "output", split: str = "train", max_samples: Optional[int] = None) -> Tuple[list, list]:
    from datasets import load_dataset

    print(f"HuggingFace 데이터셋 로드 중: {dataset_name} (split={split})")
    dataset = load_dataset(dataset_name, split=split)

    if instruction_col not in dataset.column_names:
        raise ValueError(
            f"컬럼 '{instruction_col}'이 데이터셋에 없습니다. 사용 가능한 컬럼: {dataset.column_names}"
        )
    if answer_col not in dataset.column_names:
        raise ValueError(
            f"컬럼 '{answer_col}'이 데이터셋에 없습니다. 사용 가능한 컬럼: {dataset.column_names}"
        )

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    instructions = [str(x) for x in dataset[instruction_col]]
    answers = [str(x) for x in dataset[answer_col]]

    # 빈 문자열 필터링
    valid_pairs = [(i, a) for i, a in zip(instructions, answers) if i.strip() and a.strip()]
    instructions, answers = zip(*valid_pairs) if valid_pairs else ([], [])
    instructions, answers = list(instructions), list(answers)

    print(f"HuggingFace 데이터셋 로드 완료: {len(instructions)}개 샘플")
    return instructions, answers


def load_embedding_model(model_name: str):
    """SentenceTransformer 임베딩 모델을 로드한다."""
    from sentence_transformers import SentenceTransformer

    print(f"임베딩 모델 로드 중: {model_name}")
    model = SentenceTransformer(model_name)
    print("임베딩 모델 로드 완료")
    return model


def compute_embeddings(texts: list, model, batch_size: int = 64, desc: str = "임베딩 계산") -> np.ndarray:
    """텍스트 리스트의 임베딩을 배치 단위로 계산한다."""
    print(f"{desc}: {len(texts)}개 텍스트, batch_size={batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 코사인 유사도 계산 최적화
    )
    print(f"{desc} 완료: shape={embeddings.shape}")
    return embeddings


def compute_avg_pairwise_cosine_similarity(
    embeddings: np.ndarray,
    sample_size: int = 2000,
) -> Dict[str, Any]:
    """
    평균 쌍별 코사인 유사도를 계산한다.
    값이 낮을수록 데이터가 다양함을 의미한다.
    N > sample_size 시 랜덤 샘플링으로 메모리를 절약한다.
    """
    n = len(embeddings)
    sampled = False

    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        emb = embeddings[idx]
        sampled = True
        print(f"  쌍별 유사도: {n}개 중 {sample_size}개 샘플링")
    else:
        emb = embeddings

    # 이미 normalize된 임베딩이면 내적 = 코사인 유사도
    sim_matrix = np.dot(emb, emb.T)

    # 대각선(자기 자신과의 유사도 = 1.0) 제외
    n_emb = len(emb)
    mask = np.ones((n_emb, n_emb), dtype=bool)
    np.fill_diagonal(mask, False)
    similarities = sim_matrix[mask]

    return {
        "avg_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "sampled": sampled,
        "sample_size": int(len(emb)),
        "total_size": int(n),
    }


def compute_intrinsic_dimensionality(
    embeddings: np.ndarray,
    variance_thresholds: Tuple[float, float] = (0.95, 0.99),
) -> Dict[str, Any]:
    """
    PCA를 사용해 내재적 차원수를 계산한다.
    분산 임계값(95%, 99%)을 설명하는 데 필요한 주성분 수를 반환한다.
    N > 10000 시 IncrementalPCA를 사용해 메모리를 절약한다.
    """
    n, d = embeddings.shape
    use_incremental = n > 10000

    if use_incremental:
        from sklearn.decomposition import IncrementalPCA

        print(f"  내재적 차원수: IncrementalPCA 사용 (N={n})")
        n_components = min(d, 512)
        pca = IncrementalPCA(n_components=n_components, batch_size=1000)
        pca.fit(embeddings)
    else:
        from sklearn.decomposition import PCA

        n_components = min(n - 1, d)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    result = {}

    for threshold in variance_thresholds:
        dims = int(np.searchsorted(cumvar, threshold) + 1)
        key = f"intrinsic_dim_{int(threshold * 100)}"
        result[key] = dims

    result["embedding_dim"] = int(d)
    result["total_variance_explained"] = float(cumvar[-1])
    result["top10_variance"] = float(cumvar[min(9, len(cumvar) - 1)])

    return result


def compute_vendi_score(
    embeddings: np.ndarray,
    sample_size: int = 2000,
) -> Dict[str, Any]:
    """
    Vendi Score를 계산한다.
    유사도 행렬의 고유값 분포 엔트로피의 지수(exp)로,
    값이 클수록 데이터가 다양함을 의미한다.
    N > sample_size 시 랜덤 샘플링으로 메모리를 절약한다.
    """
    from scipy.linalg import eigvalsh

    n = len(embeddings)
    sampled = False

    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        emb = embeddings[idx]
        sampled = True
        print(f"  Vendi Score: {n}개 중 {sample_size}개 샘플링")
    else:
        emb = embeddings

    # 대칭 유사도 행렬 (normalize된 임베딩: 내적 = 코사인 유사도)
    K = np.dot(emb, emb.T).astype(np.float64)

    # 대칭 행렬에 최적화된 고유값 분해
    eigenvalues = eigvalsh(K)

    # 수치 오차로 인한 음수 고유값 필터링
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # 정규화
    eigenvalues = eigenvalues / eigenvalues.sum()

    # 엔트로피 계산: H = -sum(λ * log(λ))
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-15))

    # Vendi Score = exp(H)
    vendi_score = float(np.exp(entropy))

    return {
        "vendi_score": vendi_score,
        "entropy": float(entropy),
        "n_eigenvalues": int(len(eigenvalues)),
        "sampled": sampled,
        "sample_size": int(len(emb)),
        "total_size": int(n),
    }


def compute_cluster_diversity(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    sample_size_silhouette: int = 2000,
) -> Dict[str, Any]:
    """
    K-Means 클러스터링 후 클러스터 분포의 엔트로피를 계산한다.
    균등 분포에 가까울수록 다양성이 높음을 의미한다.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)

    # 클러스터 수 안전장치: 최소 데이터 크기 보장
    n_clusters = min(n_clusters, n // 10, n - 1)
    n_clusters = max(n_clusters, 2)

    print(f"  클러스터 다양성: K-Means (k={n_clusters}, N={n})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 클러스터 크기 분포
    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    # 엔트로피 계산
    probs = cluster_sizes / cluster_sizes.sum()
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_entropy = float(np.log(n_clusters))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Silhouette Score (대용량 시 샘플링)
    if n > sample_size_silhouette:
        idx = np.random.choice(n, size=sample_size_silhouette, replace=False)
        sil_score = float(silhouette_score(embeddings[idx], labels[idx]))
    else:
        sil_score = float(silhouette_score(embeddings, labels))

    return {
        "n_clusters": int(n_clusters),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "silhouette_score": sil_score,
        "cluster_sizes": cluster_sizes.tolist(),
        "cluster_size_std": float(np.std(cluster_sizes)),
    }


def compute_coverage_score(
    embeddings: np.ndarray,
    grid_size: int = 20,
) -> Dict[str, Any]:
    """
    PCA 2D 투영 후 격자 점유율로 커버리지 점수를 계산한다.
    임베딩 공간을 얼마나 골고루 채우는지를 측정한다.
    전체 데이터를 사용해 빠르게 계산한다.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)

    # 격자 범위 설정 (outlier 영향 완화를 위해 5~95 퍼센타일 사용)
    x_min, x_max = np.percentile(proj[:, 0], [5, 95])
    y_min, y_max = np.percentile(proj[:, 1], [5, 95])

    occupied = set()
    for x, y in proj:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            xi = min(int((x - x_min) / (x_max - x_min + 1e-10) * grid_size), grid_size - 1)
            yi = min(int((y - y_min) / (y_max - y_min + 1e-10) * grid_size), grid_size - 1)
            occupied.add((xi, yi))

    total_cells = grid_size * grid_size
    coverage_score = float(len(occupied) / total_cells)

    # 임베딩 공간 분산 (전체 퍼짐 정도)
    embedding_spread = float(np.mean(np.var(embeddings, axis=0)))

    return {
        "coverage_score": coverage_score,
        "occupied_cells": int(len(occupied)),
        "total_cells": int(total_cells),
        "grid_size": int(grid_size),
        "embedding_spread": embedding_spread,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }



# 결과 집계
def compute_diversity_score(metrics: Dict[str, Any], n: int) -> float:
    """
    개별 메트릭에서 종합 다양성 점수(0~1)를 계산한다.
    - 1 - avg_similarity: 낮은 유사도 = 높은 다양성
    - log(vendi_score) / log(N): 정규화된 Vendi Score
    - normalized_entropy: 클러스터 엔트로피
    - coverage_score: 공간 커버리지
    4개 가중 평균으로 최종 점수를 산출한다.
    """
    scores = []
    weights = []

    # 평균 쌍별 유사도 반전 (낮을수록 좋으므로)
    if "avg_pairwise_cosine_similarity" in metrics:
        avg_sim = metrics["avg_pairwise_cosine_similarity"].get("avg_similarity", 0.5)
        # 클리핑: 유사도는 [-1, 1] 범위지만 실제로는 [0, 1] 근처
        sim_score = float(np.clip(1.0 - avg_sim, 0.0, 1.0))
        scores.append(sim_score)
        weights.append(0.3)

    # 정규화된 Vendi Score
    if "vendi_score" in metrics:
        vendi = metrics["vendi_score"].get("vendi_score", 1.0)
        if n > 1 and vendi > 1:
            vendi_normalized = float(np.clip(np.log(vendi) / np.log(n), 0.0, 1.0))
        else:
            vendi_normalized = 0.0
        scores.append(vendi_normalized)
        weights.append(0.3)

    # 클러스터 엔트로피
    if "cluster_diversity" in metrics:
        norm_entropy = metrics["cluster_diversity"].get("normalized_entropy", 0.0)
        scores.append(float(np.clip(norm_entropy, 0.0, 1.0)))
        weights.append(0.2)

    # 커버리지 점수
    if "coverage_score" in metrics:
        coverage = metrics["coverage_score"].get("coverage_score", 0.0)
        scores.append(float(np.clip(coverage, 0.0, 1.0)))
        weights.append(0.2)

    if not scores:
        return 0.0

    # 가중 평균
    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return float(weighted_sum / total_weight)


def evaluate_diversity(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    sample_size: int = 2000,
) -> Dict[str, Any]:
    """임베딩 배열에 대해 5가지 다양성 메트릭을 모두 계산한다."""
    results = {}

    print("  [1/5] 평균 쌍별 코사인 유사도 계산 중...")
    results["avg_pairwise_cosine_similarity"] = compute_avg_pairwise_cosine_similarity(
        embeddings, sample_size=sample_size
    )

    print("  [2/5] 내재적 차원수 계산 중...")
    results["intrinsic_dimensionality"] = compute_intrinsic_dimensionality(embeddings)

    print("  [3/5] Vendi Score 계산 중...")
    results["vendi_score"] = compute_vendi_score(embeddings, sample_size=sample_size)

    print("  [4/5] 클러스터 다양성 계산 중...")
    results["cluster_diversity"] = compute_cluster_diversity(
        embeddings, n_clusters=n_clusters
    )

    print("  [5/5] 커버리지 점수 계산 중...")
    results["coverage_score"] = compute_coverage_score(embeddings)

    return results


def save_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """결과를 JSON 파일로 저장한다."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"\n결과 저장 완료: {output_path}")


# 시각화
def visualize_embeddings(
    instruction_embeddings: np.ndarray,
    answer_embeddings: np.ndarray,
    output_path: str,
    method: str = "tsne",
    sample_size: int = 1000,
) -> None:
    """
    instruction/answer/combined 임베딩을 2D로 투영해 시각화한다.
    t-SNE 또는 UMAP을 사용하며, UMAP 미설치 시 t-SNE로 자동 폴백한다.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # UMAP 선택적 임포트 (없으면 t-SNE 폴백)
    use_umap = False
    if method == "umap":
        try:
            import umap

            use_umap = True
        except ImportError:
            print("UMAP 미설치. t-SNE로 자동 폴백합니다.")
            method = "tsne"

    def reduce_dim(emb: np.ndarray, label: str) -> np.ndarray:
        """임베딩을 2D로 차원 축소한다."""
        n = len(emb)
        if n > sample_size:
            idx = np.random.choice(n, size=sample_size, replace=False)
            emb = emb[idx]

        print(f"  {label} 차원 축소 중 ({method.upper()}, N={len(emb)})...")

        if use_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(emb)
        else:
            from sklearn.manifold import TSNE

            perplexity = min(30, len(emb) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            return reducer.fit_transform(emb)

    instr_2d = reduce_dim(instruction_embeddings, "Instruction")
    ans_2d = reduce_dim(answer_embeddings, "Answer")

    # Combined: instruction + answer concatenate
    combined_emb = np.concatenate([instruction_embeddings, answer_embeddings], axis=0)
    combined_2d = reduce_dim(combined_emb, "Combined")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"임베딩 다양성 시각화 ({method.upper()})", fontsize=16, fontweight="bold")

    datasets_2d = [
        (instr_2d, "Instruction", axes[0]),
        (ans_2d, "Answer", axes[1]),
        (combined_2d, "Combined", axes[2]),
    ]

    for data_2d, title, ax in datasets_2d:
        # 산점도
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.4, s=10, color="steelblue")

        # 밀도 등고선 overlay
        try:
            sns.kdeplot(
                x=data_2d[:, 0],
                y=data_2d[:, 1],
                ax=ax,
                levels=5,
                color="red",
                linewidths=0.8,
                alpha=0.6,
            )
        except Exception:
            pass  # KDE 실패 시 무시

        ax.set_title(f"{title} Embeddings", fontsize=12)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    plt.tight_layout()

    # PNG 저장
    viz_path = output_path.replace(".json", f"_viz_{method}.png")
    if not viz_path.endswith(".png"):
        viz_path += f"_viz_{method}.png"

    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"시각화 저장 완료: {viz_path}")


# ============================================================
# [섹션 7] CLI argparse + main()
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM 파인튜닝 데이터 다양성 평가 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 데이터 소스 (상호 배제)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="CSV 파일 경로",
    )
    data_group.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace 데이터셋 이름 (예: tatsu-lab/alpaca)",
    )

    # 컬럼 설정
    parser.add_argument(
        "--instruction-col",
        type=str,
        default="instruction",
        help="instruction 컬럼명",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="output",
        help="answer 컬럼명",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HuggingFace 데이터셋 split",
    )

    # 임베딩 모델 설정
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer 모델명",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="임베딩 계산 배치 크기",
    )

    # 평가 설정
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="K-Means 클러스터 수",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="대용량 데이터 샘플링 크기 (쌍별 유사도, Vendi Score)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="로드할 최대 샘플 수",
    )

    # 출력 설정
    parser.add_argument(
        "--output",
        type=str,
        default="./diversity_eval_results.json",
        help="결과 JSON 저장 경로",
    )

    # 시각화 설정
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="임베딩 시각화 활성화",
    )
    parser.add_argument(
        "--viz-method",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="차원 축소 방법 (tsne | umap)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 데이터 소스 검증
    if args.data_path is None and args.hf_dataset is None:
        print("오류: --data-path 또는 --hf-dataset 중 하나를 지정해야 합니다.")
        sys.exit(1)

    print("=" * 60)
    print("LLM 파인튜닝 데이터 다양성 평가 시작")
    print("=" * 60)

    # 1. 데이터 로딩
    print("\n[1단계] 데이터 로딩...")
    if args.data_path:
        instructions, answers = load_from_csv(
            args.data_path,
            instruction_col=args.instruction_col,
            answer_col=args.answer_col,
            max_samples=args.max_samples,
        )
        data_source = args.data_path
    else:
        instructions, answers = load_from_hf_dataset(
            args.hf_dataset,
            instruction_col=args.instruction_col,
            answer_col=args.answer_col,
            split=args.hf_split,
            max_samples=args.max_samples,
        )
        data_source = args.hf_dataset

    total_samples = len(instructions)
    print(f"총 {total_samples}개 샘플 로드 완료")

    # 2. 임베딩 계산
    print("\n[2단계] 임베딩 계산...")
    model = load_embedding_model(args.model)

    instruction_embeddings = compute_embeddings(
        instructions, model, batch_size=args.batch_size, desc="Instruction 임베딩"
    )
    answer_embeddings = compute_embeddings(
        answers, model, batch_size=args.batch_size, desc="Answer 임베딩"
    )
    combined_embeddings = np.concatenate(
        [instruction_embeddings, answer_embeddings], axis=0
    )

    # 3. 다양성 평가
    print("\n[3단계] Instruction 다양성 평가...")
    instruction_metrics = evaluate_diversity(
        instruction_embeddings,
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
    )

    print("\n[4단계] Answer 다양성 평가...")
    answer_metrics = evaluate_diversity(
        answer_embeddings,
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
    )

    print("\n[5단계] Combined 다양성 평가...")
    combined_metrics = evaluate_diversity(
        combined_embeddings,
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
    )

    # 4. 종합 점수 계산
    instruction_score = compute_diversity_score(instruction_metrics, total_samples)
    answer_score = compute_diversity_score(answer_metrics, total_samples)
    combined_score = compute_diversity_score(combined_metrics, total_samples * 2)

    # 5. 결과 구조화
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "data_source": data_source,
            "total_samples": total_samples,
            "instruction_col": args.instruction_col,
            "answer_col": args.answer_col,
            "n_clusters": args.n_clusters,
            "sample_size": args.sample_size,
        },
        "instruction_diversity": instruction_metrics,
        "answer_diversity": answer_metrics,
        "combined_diversity": combined_metrics,
        "summary": {
            "instruction_diversity_score": round(instruction_score, 4),
            "answer_diversity_score": round(answer_score, 4),
            "combined_diversity_score": round(combined_score, 4),
        },
    }

    # 6. 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)
    print(f"  Instruction 다양성 점수: {instruction_score:.4f}")
    print(f"  Answer 다양성 점수:      {answer_score:.4f}")
    print(f"  Combined 다양성 점수:    {combined_score:.4f}")
    print()
    print("  [Instruction 상세]")
    print(f"    평균 쌍별 유사도:    {instruction_metrics['avg_pairwise_cosine_similarity']['avg_similarity']:.4f} (낮을수록 다양)")
    print(f"    내재적 차원수(95%):  {instruction_metrics['intrinsic_dimensionality']['intrinsic_dim_95']}")
    print(f"    Vendi Score:         {instruction_metrics['vendi_score']['vendi_score']:.2f}")
    print(f"    클러스터 엔트로피:   {instruction_metrics['cluster_diversity']['normalized_entropy']:.4f}")
    print(f"    커버리지 점수:       {instruction_metrics['coverage_score']['coverage_score']:.4f}")

    # 7. JSON 저장
    print("\n[6단계] 결과 저장...")
    save_results_to_json(results, args.output)

    # 8. 시각화 (선택적)
    if args.visualize:
        print("\n[7단계] 시각화 생성...")
        visualize_embeddings(
            instruction_embeddings,
            answer_embeddings,
            output_path=args.output,
            method=args.viz_method,
        )

    print("\n완료!")


if __name__ == "__main__":
    main()
