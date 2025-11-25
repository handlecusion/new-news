"""
프레임 해석 모듈
각 프레임의 특성을 분석하고 구분 이유를 설명합니다.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# 설정 모듈 로드
from src import config


class FrameInterpreter:
    """프레임 해석 및 대표 문장 추출 클래스"""

    def __init__(
        self,
        articles: List[Dict],
        frame_assignments: np.ndarray,
        frame_probs: Optional[np.ndarray] = None,
        frame_info: Optional[List[Dict]] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Args:
            articles: 기사 리스트
            frame_assignments: 프레임 할당
            frame_probs: 프레임 확률 분포
            frame_info: 프레임 정보 (키워드 등)
            embeddings: 문서 임베딩 (대표 문장 선택에 사용)
        """
        self.articles = articles
        self.frame_assignments = frame_assignments
        self.frame_probs = frame_probs
        self.frame_info = frame_info or []
        self.embeddings = embeddings

        # 프레임별 기사 인덱스 매핑
        self.frame_to_articles = self._create_frame_mapping()

    def _create_frame_mapping(self) -> Dict[int, List[int]]:
        """프레임별 기사 인덱스 매핑 생성"""
        mapping = {}
        for idx, frame_id in enumerate(self.frame_assignments):
            if frame_id == -1:  # outlier 제외
                continue
            if frame_id not in mapping:
                mapping[frame_id] = []
            mapping[frame_id].append(idx)
        return mapping

    def get_representative_articles(
        self, frame_id: int, n_articles: int = 5, method: str = "probability"
    ) -> List[Dict]:
        """
        특정 프레임의 대표 기사 추출

        Args:
            frame_id: 프레임 ID
            n_articles: 추출할 기사 수
            method: 선택 방법 ("probability", "central", "diverse")

        Returns:
            대표 기사 리스트
        """
        if frame_id not in self.frame_to_articles:
            return []

        article_indices = self.frame_to_articles[frame_id]

        if method == "probability":
            # 프레임 확률이 높은 기사 선택
            if self.frame_probs is not None:
                frame_probs = self.frame_probs[article_indices, frame_id]
                top_indices = np.argsort(frame_probs)[::-1][:n_articles]
                selected_indices = [article_indices[i] for i in top_indices]
                probabilities = [frame_probs[i] for i in top_indices]
            else:
                # 확률이 없으면 랜덤 선택
                import random
                selected_indices = random.sample(
                    article_indices, min(n_articles, len(article_indices))
                )
                probabilities = [1.0] * len(selected_indices)

        elif method == "central":
            # 중심에 가까운 기사 선택 (임베딩 기반)
            if self.embeddings is not None:
                frame_embeddings = self.embeddings[article_indices]
                centroid = frame_embeddings.mean(axis=0)

                # 중심과의 거리 계산
                distances = np.linalg.norm(frame_embeddings - centroid, axis=1)
                closest_indices = np.argsort(distances)[:n_articles]
                selected_indices = [article_indices[i] for i in closest_indices]
                probabilities = [1.0 - distances[i] for i in closest_indices]
            else:
                # 임베딩이 없으면 probability 방법 사용
                return self.get_representative_articles(
                    frame_id, n_articles, method="probability"
                )

        elif method == "diverse":
            # 다양한 기사 선택 (임베딩 기반 다양성 샘플링)
            if self.embeddings is not None:
                frame_embeddings = self.embeddings[article_indices]
                selected_indices = []
                remaining_indices = list(range(len(article_indices)))

                # 첫 번째는 랜덤 선택
                import random
                first_idx = random.choice(remaining_indices)
                selected_indices.append(article_indices[first_idx])
                remaining_indices.remove(first_idx)

                # 나머지는 최대 거리 선택
                while len(selected_indices) < n_articles and remaining_indices:
                    selected_embeddings = self.embeddings[selected_indices]
                    max_min_distance = -1
                    best_idx = None

                    for idx in remaining_indices:
                        candidate_embedding = frame_embeddings[idx]
                        min_distance = min(
                            np.linalg.norm(candidate_embedding - emb)
                            for emb in selected_embeddings
                        )
                        if min_distance > max_min_distance:
                            max_min_distance = min_distance
                            best_idx = idx

                    if best_idx is not None:
                        selected_indices.append(article_indices[best_idx])
                        remaining_indices.remove(best_idx)

                probabilities = [1.0] * len(selected_indices)
            else:
                return self.get_representative_articles(
                    frame_id, n_articles, method="probability"
                )

        # 기사 정보 구성
        representative_articles = []
        for idx, prob in zip(selected_indices, probabilities):
            article = self.articles[idx].copy()
            article["selection_score"] = float(prob)
            article["article_index"] = int(idx)
            representative_articles.append(article)

        return representative_articles

    def extract_key_sentences(
        self, text: str, n_sentences: int = 3, keyword_boost: Optional[List[str]] = None
    ) -> List[str]:
        """
        텍스트에서 핵심 문장 추출

        Args:
            text: 입력 텍스트
            n_sentences: 추출할 문장 수
            keyword_boost: 키워드 부스팅 (프레임 키워드)

        Returns:
            핵심 문장 리스트
        """
        # 문장 분리
        sentences = []
        for s in text.split("."):
            s = s.strip()
            if len(s) > 10:  # 너무 짧은 문장 제외
                sentences.append(s)

        if len(sentences) <= n_sentences:
            return sentences

        # 문장별 점수 계산
        scores = []
        for sent in sentences:
            score = len(sent)  # 기본: 문장 길이

            # 키워드 포함 시 점수 부스팅
            if keyword_boost:
                for keyword in keyword_boost:
                    if keyword in sent:
                        score += 100

            scores.append(score)

        # 상위 문장 선택 (원본 순서 유지)
        top_indices = np.argsort(scores)[::-1][:n_sentences]
        top_indices = sorted(top_indices)  # 원본 순서대로

        return [sentences[i] for i in top_indices]

    def analyze_frame_characteristics(self, frame_id: int) -> Dict:
        """
        특정 프레임의 특성 분석

        Args:
            frame_id: 프레임 ID

        Returns:
            프레임 특성 딕셔너리
        """
        if frame_id not in self.frame_to_articles:
            return {}

        article_indices = self.frame_to_articles[frame_id]
        frame_articles = [self.articles[i] for i in article_indices]

        # 프레임 메타 정보
        frame_meta = next(
            (f for f in self.frame_info if f["frame_id"] == frame_id), None
        )

        # 편향도 분석
        bias_scores = [a["bias_score"] for a in frame_articles]
        mean_bias = np.mean(bias_scores)
        std_bias = np.std(bias_scores)

        # 편향 레이블 분포
        bias_labels = []
        for score in bias_scores:
            if score < -0.3:
                bias_labels.append("진보")
            elif score > 0.3:
                bias_labels.append("보수")
            else:
                bias_labels.append("중도")
        bias_dist = Counter(bias_labels)

        # 언론사 분포
        media_dist = Counter([a["media_outlet"] for a in frame_articles])

        # 시간적 분포 (날짜가 있다면)
        dates = []
        for a in frame_articles:
            if "published_date" in a:
                dates.append(a["published_date"])

        # 프레임 성향 판단
        if mean_bias < -0.3:
            frame_tendency = "진보 성향"
        elif mean_bias > 0.3:
            frame_tendency = "보수 성향"
        else:
            frame_tendency = "중도 성향"

        # 프레임 일관성 (편향도 표준편차가 낮으면 일관적)
        # bias_score 범위: -1.0 ~ +1.0 (총 범위 2.0)
        # 표준편차 기준:
        #   - 높음: std < 0.3 (편향도가 한 방향으로 집중)
        #   - 중간: 0.3 ≤ std < 0.5 (일정한 경향성 존재)
        #   - 낮음: std ≥ 0.5 (편향도가 넓게 분산)
        if std_bias < 0.3:
            consistency = "높음"
        elif std_bias < 0.5:
            consistency = "중간"
        else:
            consistency = "낮음"

        characteristics = {
            "frame_id": frame_id,
            "n_articles": len(frame_articles),
            "keywords": frame_meta.get("keywords", [])[:10] if frame_meta else [],
            "suggested_name": frame_meta.get("suggested_name", "") if frame_meta else "",
            "bias_stats": {
                "mean": float(mean_bias),
                "std": float(std_bias),
                "min": float(min(bias_scores)),
                "max": float(max(bias_scores)),
            },
            "bias_distribution": dict(bias_dist),
            "media_distribution": dict(media_dist.most_common(10)),
            "frame_tendency": frame_tendency,
            "consistency": consistency,
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None,
            },
        }

        return characteristics

    def explain_frame_distinction(self, frame_id: int, n_examples: int = 3) -> Dict:
        """
        프레임 구분 이유 설명

        Args:
            frame_id: 프레임 ID
            n_examples: 예시 문장 수

        Returns:
            설명 딕셔너리
        """
        characteristics = self.analyze_frame_characteristics(frame_id)
        if not characteristics:
            return {}

        # 대표 기사 추출
        representative_articles = self.get_representative_articles(
            frame_id, n_articles=n_examples, method="probability"
        )

        # 각 기사에서 핵심 문장 추출
        examples = []
        keywords = characteristics.get("keywords", [])[:5]

        for article in representative_articles:
            title = article["title"]
            content = article.get("content", "")
            full_text = f"{title}. {content}"

            # 핵심 문장 추출
            key_sentences = self.extract_key_sentences(
                full_text, n_sentences=2, keyword_boost=keywords
            )

            examples.append({
                "article_id": article.get("article_id", ""),
                "media_outlet": article["media_outlet"],
                "bias_score": article["bias_score"],
                "title": title,
                "key_sentences": key_sentences,
                "selection_score": article.get("selection_score", 0),
            })

        # 구분 이유 생성
        reasons = []

        # 1. 키워드 기반 이유
        if keywords:
            reasons.append(f"주요 키워드: {', '.join(keywords)}")

        # 2. 편향 성향
        reasons.append(f"편향 성향: {characteristics['frame_tendency']}")
        reasons.append(
            f"평균 편향도: {characteristics['bias_stats']['mean']:.2f} "
            f"(±{characteristics['bias_stats']['std']:.2f})"
        )

        # 3. 일관성
        reasons.append(f"프레임 내 일관성: {characteristics['consistency']}")

        # 4. 주요 언론사
        top_media = list(characteristics["media_distribution"].items())[:3]
        if top_media:
            media_str = ", ".join([f"{m}({c})" for m, c in top_media])
            reasons.append(f"주요 언론사: {media_str}")

        explanation = {
            "frame_id": frame_id,
            "frame_name": characteristics["suggested_name"],
            "n_articles": characteristics["n_articles"],
            "characteristics": characteristics,
            "distinction_reasons": reasons,
            "representative_examples": examples,
        }

        return explanation

    def create_frame_interpretation_report(
        self, save_path: Optional[str] = None, n_examples: int = 5
    ) -> Dict:
        """
        전체 프레임 해석 리포트 생성

        Args:
            save_path: 저장 경로
            n_examples: 프레임당 예시 수

        Returns:
            해석 리포트 딕셔너리
        """
        print("\n" + "=" * 60)
        print("프레임 해석 리포트 생성")
        print("=" * 60)

        report = {
            "summary": {
                "total_frames": len(self.frame_to_articles),
                "total_articles": len(self.articles),
                "outliers": int(np.sum(self.frame_assignments == -1)),
            },
            "frame_interpretations": [],
        }

        # 각 프레임 해석
        for frame_id in sorted(self.frame_to_articles.keys()):
            print(f"\n프레임 {frame_id} 분석 중...")

            explanation = self.explain_frame_distinction(frame_id, n_examples)
            report["frame_interpretations"].append(explanation)

            # 콘솔 출력
            print(f"\n=== 프레임 {frame_id}: {explanation['frame_name']} ===")
            print(f"기사 수: {explanation['n_articles']}")
            print("\n[구분 이유]")
            for reason in explanation["distinction_reasons"]:
                print(f"  • {reason}")

            print("\n[대표 예시]")
            for i, example in enumerate(explanation["representative_examples"][:3], 1):
                print(f"\n  {i}. {example['title']}")
                print(f"     언론사: {example['media_outlet']} | 편향도: {example['bias_score']:.2f}")
                print(f"     핵심 문장:")
                for sent in example["key_sentences"]:
                    print(f"       - {sent}")

        # 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                # numpy/pandas 객체를 JSON 직렬화 가능한 형태로 변환
                json_report = self._prepare_for_json(report)
                json.dump(json_report, f, ensure_ascii=False, indent=2)

            print(f"\n✓ 프레임 해석 리포트 저장: {save_path}")

        return report

    def _prepare_for_json(self, obj):
        """JSON 직렬화를 위한 데이터 변환"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj

    def compare_frames(self, frame_id1: int, frame_id2: int) -> Dict:
        """
        두 프레임 비교 분석

        Args:
            frame_id1: 첫 번째 프레임 ID
            frame_id2: 두 번째 프레임 ID

        Returns:
            비교 결과 딕셔너리
        """
        char1 = self.analyze_frame_characteristics(frame_id1)
        char2 = self.analyze_frame_characteristics(frame_id2)

        if not char1 or not char2:
            return {}

        # 키워드 비교
        keywords1 = set(char1.get("keywords", []))
        keywords2 = set(char2.get("keywords", []))
        common_keywords = keywords1 & keywords2
        unique_keywords1 = keywords1 - keywords2
        unique_keywords2 = keywords2 - keywords1

        # 편향도 차이
        bias_diff = abs(char1["bias_stats"]["mean"] - char2["bias_stats"]["mean"])

        # 언론사 중복
        media1 = set(char1["media_distribution"].keys())
        media2 = set(char2["media_distribution"].keys())
        common_media = media1 & media2

        comparison = {
            "frame_1": {
                "id": frame_id1,
                "name": char1["suggested_name"],
                "n_articles": char1["n_articles"],
                "bias_mean": char1["bias_stats"]["mean"],
                "unique_keywords": list(unique_keywords1)[:5],
            },
            "frame_2": {
                "id": frame_id2,
                "name": char2["suggested_name"],
                "n_articles": char2["n_articles"],
                "bias_mean": char2["bias_stats"]["mean"],
                "unique_keywords": list(unique_keywords2)[:5],
            },
            "comparison": {
                "common_keywords": list(common_keywords),
                "bias_difference": float(bias_diff),
                "common_media_count": len(common_media),
                "similarity_score": len(common_keywords) / max(len(keywords1), len(keywords2))
                if keywords1 or keywords2 else 0,
            },
        }

        return comparison


def interpret_frames(
    articles_path: str = None,
    frames_path: str = "results/frames.json",
    article_frames_path: str = "results/article_frames.json",
    embeddings_path: Optional[str] = None,
    output_path: str = "results/analysis/frame_interpretation.json",
):
    """
    프레임 해석 실행 (테스트/실행용)

    Args:
        articles_path: 기사 데이터 경로 (기본값: config.yaml의 data.input_path)
        frames_path: 프레임 정보 경로
        article_frames_path: 기사별 프레임 경로
        embeddings_path: 임베딩 경로 (기본값: config.yaml의 data.processed_path)
        output_path: 출력 경로
    """
    articles_path = articles_path or config.get_input_path()
    embeddings_path = embeddings_path or f"{config.get_processed_path()}embeddings.npy"

    # 데이터 로드
    with open(articles_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"]

    with open(frames_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    with open(article_frames_path, "r", encoding="utf-8") as f:
        article_frames = json.load(f)

    # 프레임 할당 배열 생성
    frame_assignments = np.array([af["assigned_frame"] for af in article_frames])

    # 프레임 확률
    frame_probs = None
    if article_frames[0].get("frame_probabilities"):
        frame_probs = np.array([af["frame_probabilities"] for af in article_frames])

    # 임베딩 로드 (있다면)
    embeddings = None
    if embeddings_path and Path(embeddings_path).exists():
        embeddings = np.load(embeddings_path)
        print(f"✓ 임베딩 로드: {embeddings.shape}")

    # 해석기 초기화
    interpreter = FrameInterpreter(
        articles, frame_assignments, frame_probs, frames, embeddings
    )

    # 해석 리포트 생성
    report = interpreter.create_frame_interpretation_report(
        save_path=output_path, n_examples=5
    )

    print("\n✓ 프레임 해석 완료")
    return interpreter, report


if __name__ == "__main__":
    # 테스트 실행
    input_path = config.get_input_path()
    if (
        Path(input_path).exists()
        and Path("results/frames.json").exists()
        and Path("results/article_frames.json").exists()
    ):
        interpret_frames()
    else:
        print("⚠️ 필요한 파일을 찾을 수 없습니다.")
        print("  먼저 프레임 추출을 실행하세요.")
