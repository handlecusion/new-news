"""
프레임 추출 모듈
BERTopic을 사용하여 뉴스 기사에서 프레임을 자동으로 발견합니다.
"""

import json
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 설정 파일 로드
config_path = Path(__file__).parent.parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}


class FrameExtractor:
    """BERTopic 기반 프레임 추출 클래스"""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        min_topic_size: Optional[int] = None,
        nr_topics: Optional[str | int] = None,
        language: str = "korean",
        verbose: bool = True,
    ):
        """
        Args:
            embedding_model: 임베딩 모델 이름
            min_topic_size: 최소 토픽 크기
            nr_topics: 토픽 수 ("auto" 또는 정수)
            language: 언어 설정
            verbose: 로그 출력 여부
        """
        self.embedding_model = embedding_model or config.get("unsupervised", {}).get(
            "embedding_model", "jhgan/ko-sroberta-multitask"
        )
        self.min_topic_size = min_topic_size or config.get("unsupervised", {}).get(
            "min_topic_size", 5
        )
        self.nr_topics = nr_topics or config.get("unsupervised", {}).get(
            "nr_topics", "auto"
        )
        self.language = language
        self.verbose = verbose

        self.topic_model = None
        self.vectorizer = None
        self._initialize_models()

    def _initialize_models(self):
        """BERTopic 모델 초기화"""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer

            if self.verbose:
                print(f"BERTopic 모델 초기화 중...")
                print(f"  임베딩 모델: {self.embedding_model}")
                print(f"  최소 토픽 크기: {self.min_topic_size}")
                print(f"  토픽 수: {self.nr_topics}")

            # 한국어 토크나이저 설정
            if self.language == "korean":
                try:
                    from konlpy.tag import Mecab

                    mecab = Mecab()

                    def korean_tokenizer(text):
                        """한국어 명사 추출 토크나이저"""
                        tokens = []
                        try:
                            pos_tagged = mecab.pos(text)
                            tokens = [
                                word for word, pos in pos_tagged if pos.startswith("N")
                            ]
                        except:
                            tokens = text.split()
                        return tokens

                    self.vectorizer = CountVectorizer(
                        tokenizer=korean_tokenizer,
                        min_df=config.get("unsupervised", {}).get("min_df", 2),
                        max_df=config.get("unsupervised", {}).get("max_df", 0.8),
                    )
                except ImportError:
                    print("⚠️ KoNLPy가 설치되지 않았습니다. 기본 토크나이저를 사용합니다.")
                    self.vectorizer = CountVectorizer(
                        min_df=2, max_df=0.8, ngram_range=(1, 2)
                    )
            else:
                self.vectorizer = CountVectorizer(
                    min_df=2, max_df=0.8, ngram_range=(1, 2)
                )

            # BERTopic 모델 생성
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=self.vectorizer,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics if self.nr_topics != "auto" else None,
                calculate_probabilities=config.get("unsupervised", {}).get(
                    "calculate_probabilities", True
                ),
                verbose=self.verbose,
                language="english",  # BERTopic 내부 설정
            )

            if self.verbose:
                print("✓ BERTopic 모델 초기화 완료")

        except ImportError as e:
            print(f"⚠️ 필요한 패키지가 설치되지 않았습니다: {e}")
            print("  pip install bertopic 명령으로 설치해주세요.")
            raise

    def extract_frames(
        self, documents: List[Dict] | List[str], return_probs: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        문서에서 프레임(토픽) 추출

        Args:
            documents: 문서 리스트 (딕셔너리 또는 문자열)
            return_probs: 확률값 반환 여부

        Returns:
            topics: 각 문서의 토픽 ID
            probs: 각 문서의 토픽 확률 (optional)
        """
        if not self.topic_model:
            raise ValueError("BERTopic 모델이 초기화되지 않았습니다.")

        # 텍스트 추출
        if isinstance(documents[0], dict):
            texts = []
            for doc in documents:
                # 제목과 본문 결합
                title = doc.get("title", "")
                content = doc.get("content", "")
                text = f"{title} {content}".strip()
                texts.append(text)
        else:
            texts = documents

        if self.verbose:
            print(f"\n{len(texts)}개 문서에서 프레임 추출 중...")

        # BERTopic 학습 및 변환
        topics, probs = self.topic_model.fit_transform(texts)

        # NumPy 배열로 변환
        topics = np.array(topics)

        if self.verbose:
            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            n_outliers = np.sum(topics == -1)
            print(f"✓ 프레임 추출 완료")
            print(f"  발견된 프레임 수: {n_topics}")
            print(f"  outlier 문서 수: {n_outliers}")

        if return_probs and probs is not None:
            return topics, np.array(probs)
        else:
            return topics, None

    def get_frame_info(self, n_words: int = 10) -> List[Dict]:
        """
        각 프레임의 정보 추출

        Args:
            n_words: 프레임당 추출할 키워드 수

        Returns:
            프레임 정보 리스트
        """
        if not self.topic_model:
            raise ValueError("모델이 학습되지 않았습니다.")

        topic_info = self.topic_model.get_topic_info()
        frames = []

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:  # outliers 제외
                continue

            # 토픽 키워드 추출
            keywords = self.topic_model.get_topic(topic_id)[:n_words]

            frame = {
                "frame_id": int(topic_id),
                "keywords": [word for word, score in keywords],
                "keyword_scores": [float(score) for word, score in keywords],
                "size": int(row["Count"]),
                "name": row.get("Name", f"Frame_{topic_id}"),
            }
            frames.append(frame)

        return frames

    def assign_frame_names(
        self, frames: List[Dict], method: str = "keyword"
    ) -> List[Dict]:
        """
        프레임에 의미있는 이름 부여

        Args:
            frames: 프레임 정보 리스트
            method: 명명 방법 ("keyword", "manual", "llm")

        Returns:
            이름이 부여된 프레임 리스트
        """
        if method == "keyword":
            # 상위 키워드 기반 명명
            for frame in frames:
                top_keywords = frame["keywords"][:3]
                frame["suggested_name"] = "_".join(top_keywords)

        elif method == "manual":
            # 수동 명명 (키워드 기반 추론)
            frame_name_map = {
                # 키워드 패턴별 프레임 이름
                "노동자": "노동자_생계_프레임",
                "생계": "노동자_생계_프레임",
                "임금": "임금_수준_프레임",
                "소상공인": "소상공인_부담_프레임",
                "자영업": "소상공인_부담_프레임",
                "부담": "경제_부담_프레임",
                "고용": "고용_영향_프레임",
                "일자리": "고용_영향_프레임",
                "경제": "경제_효과_프레임",
                "성장": "경제_성장_프레임",
                "정책": "정책_방향_프레임",
                "정부": "정부_역할_프레임",
            }

            for frame in frames:
                # 키워드에서 프레임 이름 추론
                suggested_name = None
                for keyword in frame["keywords"]:
                    for pattern, name in frame_name_map.items():
                        if pattern in keyword:
                            suggested_name = name
                            break
                    if suggested_name:
                        break

                if not suggested_name:
                    suggested_name = f"프레임_{frame['frame_id']}"

                frame["suggested_name"] = suggested_name

        elif method == "llm":
            # LLM을 사용한 자동 명명 (구현 예정)
            print("⚠️ LLM 기반 명명은 아직 구현되지 않았습니다.")
            for frame in frames:
                frame["suggested_name"] = f"프레임_{frame['frame_id']}"

        return frames

    def get_representative_docs(
        self, topic_id: int, n_docs: int = 3
    ) -> List[Tuple[str, float]]:
        """
        특정 프레임의 대표 문서 추출

        Args:
            topic_id: 토픽 ID
            n_docs: 추출할 문서 수

        Returns:
            (문서, 확률) 튜플 리스트
        """
        if not self.topic_model:
            raise ValueError("모델이 학습되지 않았습니다.")

        try:
            docs = self.topic_model.get_representative_docs(topic_id)
            if docs:
                return docs[:n_docs]
        except:
            pass

        return []

    def update_topics(self, docs: List[str], topics: List[int]) -> None:
        """
        토픽 업데이트 (수동 조정)

        Args:
            docs: 문서 리스트
            topics: 새로운 토픽 할당
        """
        if not self.topic_model:
            raise ValueError("모델이 학습되지 않았습니다.")

        self.topic_model.update_topics(docs, topics)

    def reduce_topics(self, nr_topics: int) -> None:
        """
        토픽 수 감소

        Args:
            nr_topics: 목표 토픽 수
        """
        if not self.topic_model:
            raise ValueError("모델이 학습되지 않았습니다.")

        self.topic_model.reduce_topics(docs=None, nr_topics=nr_topics)

    def save_model(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        if not self.topic_model:
            raise ValueError("저장할 모델이 없습니다.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.topic_model.save(str(path))
        print(f"✓ 모델 저장 완료: {path}")

    def load_model(self, path: str):
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        from bertopic import BERTopic

        self.topic_model = BERTopic.load(path)
        print(f"✓ 모델 로드 완료: {path}")


def extract_frames_from_json(json_path: str, output_dir: str = "results"):
    """
    JSON 파일에서 프레임 추출 (테스트/실행용)

    Args:
        json_path: 입력 JSON 파일 경로
        output_dir: 결과 저장 디렉토리
    """
    # 데이터 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data["articles"]
    print(f"로드된 기사 수: {len(articles)}")

    # 프레임 추출기 초기화
    extractor = FrameExtractor()

    # 프레임 추출
    topics, probs = extractor.extract_frames(articles)

    # 프레임 정보 추출
    frames = extractor.get_frame_info(n_words=15)
    frames = extractor.assign_frame_names(frames, method="manual")

    # 결과 저장
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 정보 저장
    frames_path = output_dir / "frames.json"
    with open(frames_path, "w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=2)
    print(f"✓ 프레임 정보 저장: {frames_path}")

    # 기사별 프레임 할당 저장
    article_frames = []
    for i, article in enumerate(articles):
        result = {
            "article_id": article.get("article_id", f"article_{i}"),
            "media_outlet": article["media_outlet"],
            "bias_score": article["bias_score"],
            "title": article["title"][:100],
            "assigned_frame": int(topics[i]),
        }
        if probs is not None:
            result["frame_probabilities"] = probs[i].tolist()
        article_frames.append(result)

    article_frames_path = output_dir / "article_frames.json"
    with open(article_frames_path, "w", encoding="utf-8") as f:
        json.dump(article_frames, f, ensure_ascii=False, indent=2)
    print(f"✓ 기사별 프레임 저장: {article_frames_path}")

    # 요약 출력
    print("\n=== 프레임 추출 결과 ===")
    for frame in frames:
        print(f"\n프레임 {frame['frame_id']}: {frame['suggested_name']}")
        print(f"  문서 수: {frame['size']}")
        print(f"  주요 키워드: {', '.join(frame['keywords'][:5])}")

    return extractor, topics, probs, frames


if __name__ == "__main__":
    # 테스트 실행
    json_path = "data/input/articles.json"
    if Path(json_path).exists():
        extract_frames_from_json(json_path)
    else:
        print(f"⚠️ 파일을 찾을 수 없습니다: {json_path}")
        print("  먼저 generate_sample_data.py를 실행하세요.")