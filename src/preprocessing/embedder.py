"""
문서 임베딩 생성 모듈
한국어 문서를 벡터로 변환합니다.
"""

import numpy as np
import pickle
import yaml
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm

# 설정 파일 로드
config_path = Path(__file__).parent.parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}


class DocumentEmbedder:
    """문서 임베딩 생성 클래스"""

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Args:
            model_name: 사용할 임베딩 모델 이름
            cache_dir: 임베딩 캐시 디렉토리
        """
        self.model_name = model_name or config.get("unsupervised", {}).get(
            "embedding_model", "jhgan/ko-sroberta-multitask"
        )
        self.cache_dir = Path(cache_dir or config.get("data", {}).get(
            "processed_path", "data/processed"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """임베딩 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer

            print(f"임베딩 모델 로딩 중: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ 임베딩 모델 로딩 완료")

            # 모델 정보 출력
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                dim = self.model.get_sentence_embedding_dimension()
                print(f"  임베딩 차원: {dim}")

        except ImportError:
            print("⚠️ sentence-transformers가 설치되지 않았습니다.")
            print("  pip install sentence-transformers 명령으로 설치해주세요.")
            raise
        except Exception as e:
            print(f"⚠️ 모델 로딩 실패: {e}")
            raise

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        문서들을 벡터로 변환

        Args:
            texts: 문서 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행 상황 표시 여부
            normalize: 벡터 정규화 여부
            save_cache: 캐시 저장 여부
            cache_name: 캐시 파일명

        Returns:
            문서 임베딩 배열 (n_documents, embedding_dim)
        """
        if not self.model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")

        if not texts:
            return np.array([])

        # 캐시 확인
        if cache_name:
            cache_path = self.cache_dir / f"{cache_name}.npy"
            if cache_path.exists():
                print(f"캐시에서 임베딩 로드: {cache_path}")
                embeddings = np.load(cache_path)
                if len(embeddings) == len(texts):
                    return embeddings
                else:
                    print("캐시된 임베딩 수가 일치하지 않습니다. 재생성합니다.")

        print(f"{len(texts)}개 문서 임베딩 생성 중...")

        # 임베딩 생성
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패: {e}")
            # 폴백: 단순 임베딩 시도
            embeddings = []
            for text in tqdm(texts, desc="임베딩 생성", disable=not show_progress):
                try:
                    emb = self.model.encode(text, convert_to_numpy=True)
                    embeddings.append(emb)
                except:
                    # 실패시 영벡터
                    dim = self.model.get_sentence_embedding_dimension()
                    embeddings.append(np.zeros(dim))
            embeddings = np.array(embeddings)

        print(f"✓ 임베딩 생성 완료: shape={embeddings.shape}")

        # 캐시 저장
        if save_cache and cache_name:
            cache_path = self.cache_dir / f"{cache_name}.npy"
            np.save(cache_path, embeddings)
            print(f"✓ 임베딩 캐시 저장: {cache_path}")

        return embeddings

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        단일 문서 임베딩

        Args:
            text: 문서 텍스트
            normalize: 벡터 정규화 여부

        Returns:
            문서 임베딩 벡터
        """
        if not self.model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embedding

    def compute_similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        두 임베딩 세트 간의 코사인 유사도 계산

        Args:
            embeddings1: 첫 번째 임베딩 세트
            embeddings2: 두 번째 임베딩 세트

        Returns:
            유사도 행렬
        """
        # 정규화
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # 코사인 유사도
        similarity = np.dot(embeddings1, embeddings2.T)

        return similarity

    def find_similar_documents(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[tuple]:
        """
        유사한 문서 찾기

        Args:
            query_embedding: 쿼리 문서 임베딩
            document_embeddings: 문서 임베딩 세트
            top_k: 상위 k개 반환
            threshold: 유사도 임계값

        Returns:
            (인덱스, 유사도) 튜플 리스트
        """
        # 유사도 계산
        similarities = self.compute_similarity(
            query_embedding.reshape(1, -1), document_embeddings
        )[0]

        # 정렬
        sorted_indices = np.argsort(similarities)[::-1]

        # 상위 k개 선택
        results = []
        for idx in sorted_indices[:top_k]:
            similarity = similarities[idx]
            if threshold is None or similarity >= threshold:
                results.append((int(idx), float(similarity)))

        return results

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        임베딩 저장

        Args:
            embeddings: 임베딩 배열
            filepath: 저장 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix == ".npy":
            np.save(filepath, embeddings)
        elif filepath.suffix == ".pkl":
            with open(filepath, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            # 기본값: numpy 형식
            np.save(filepath.with_suffix(".npy"), embeddings)

        print(f"✓ 임베딩 저장 완료: {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        임베딩 로드

        Args:
            filepath: 파일 경로

        Returns:
            임베딩 배열
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        if filepath.suffix == ".npy":
            embeddings = np.load(filepath)
        elif filepath.suffix == ".pkl":
            with open(filepath, "rb") as f:
                embeddings = pickle.load(f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")

        print(f"✓ 임베딩 로드 완료: {filepath}, shape={embeddings.shape}")
        return embeddings


class FastTextEmbedder:
    """FastText 기반 임베딩 (경량 대안)"""

    def __init__(self):
        """FastText 임베딩 초기화"""
        self.model = None
        print("⚠️ FastText 임베딩은 아직 구현되지 않았습니다.")
        print("  DocumentEmbedder를 사용하세요.")


def test_embedder():
    """임베딩 모듈 테스트"""
    print("\n=== 임베딩 모듈 테스트 ===\n")

    # 테스트 문서
    test_docs = [
        "최저임금이 9,860원으로 인상되었습니다.",
        "노동자들의 생계가 개선될 것으로 기대됩니다.",
        "소상공인들은 인건비 부담을 우려하고 있습니다.",
        "경제 전반에 미치는 영향을 분석해야 합니다.",
    ]

    try:
        # 임베딩 생성기 초기화
        embedder = DocumentEmbedder()

        # 문서 임베딩
        embeddings = embedder.embed_documents(
            test_docs, show_progress=False, save_cache=False
        )

        print(f"임베딩 shape: {embeddings.shape}")
        print(f"임베딩 타입: {embeddings.dtype}")

        # 유사도 계산
        print("\n문서 간 유사도:")
        similarity_matrix = embedder.compute_similarity(embeddings, embeddings)

        for i in range(len(test_docs)):
            for j in range(i + 1, len(test_docs)):
                print(
                    f"  문서 {i+1} <-> 문서 {j+1}: {similarity_matrix[i, j]:.3f}"
                )

        # 유사 문서 검색
        print("\n문서 1과 유사한 문서:")
        similar_docs = embedder.find_similar_documents(
            embeddings[0], embeddings[1:], top_k=3
        )

        for idx, similarity in similar_docs:
            print(f"  문서 {idx+2}: {similarity:.3f} - {test_docs[idx+1]}")

    except Exception as e:
        print(f"테스트 실패: {e}")
        print("sentence-transformers를 설치했는지 확인하세요.")


if __name__ == "__main__":
    test_embedder()