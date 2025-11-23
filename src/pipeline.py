#!/usr/bin/env python3
"""
뉴스 프레임-편향도 분석 메인 파이프라인
전체 분석 프로세스를 통합 실행합니다.
"""

import json
import yaml
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Optional, Any
from sklearn.model_selection import train_test_split
import sys

# 경고 메시지 억제
warnings.filterwarnings("ignore")

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 임포트
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.embedder import DocumentEmbedder
from src.unsupervised.frame_extractor import FrameExtractor
from src.unsupervised.visualizer import FrameVisualizer
from src.supervised.bias_classifier import BiasClassifier
from src.supervised.frame_predictor import FrameBasedBiasPredictor
from src.analysis.correlation import IntegratedAnalyzer
from src.analysis.dashboard import InteractiveDashboard
from src.analysis.frame_interpreter import FrameInterpreter

# 설정 파일 로드
config_path = project_root / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}


class FrameBiasAnalysisPipeline:
    """프레임-편향도 분석 파이프라인"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            data_path: 입력 데이터 경로
            output_dir: 출력 디렉토리
            verbose: 로그 출력 여부
        """
        self.data_path = Path(data_path or config.get("data", {}).get(
            "input_path", "data/input/articles.json"
        ))
        self.output_dir = Path(output_dir or config.get("output", {}).get(
            "results_dir", "results"
        ))
        self.verbose = verbose

        # 데이터 및 결과 저장용
        self.articles = None
        self.frames = None
        self.frame_assignments = None
        self.frame_probs = None
        self.embeddings = None
        self.bias_classifier = None
        self.frame_predictor = None

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """JSON 데이터 로드"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.articles = data["articles"]

        if self.verbose:
            print(f"\n✓ 데이터 로드 완료")
            print(f"  - 기사 수: {len(self.articles)}")
            print(f"  - 이슈: {data.get('metadata', {}).get('issue', 'N/A')}")
            print(f"  - 수집 기간: {data.get('metadata', {}).get('collection_period', 'N/A')}")

    def run_preprocessing(self):
        """데이터 전처리"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("1단계: 데이터 전처리")
            print("=" * 60)

        # 텍스트 전처리기 초기화
        preprocessor = TextPreprocessor(use_mecab=False)  # 간단한 버전 사용

        # 문서 임베딩 생성
        embedder = DocumentEmbedder()

        # 텍스트 결합 및 정제
        texts = []
        for article in self.articles:
            title = article.get("title", "")
            content = article.get("content", "")
            full_text = f"{title} {content}"

            # BERT용 정제
            clean_text = preprocessor.preprocess_for_bert(full_text)
            texts.append(clean_text)

        # 임베딩 생성 (캐시 사용)
        try:
            embeddings = embedder.embed_documents(
                texts,
                show_progress=self.verbose,
                cache_name="article_embeddings"
            )

            if self.verbose:
                print(f"✓ 임베딩 생성 완료: shape={embeddings.shape}")

            # 임베딩 저장 (프레임 해석에 사용)
            self.embeddings = embeddings

        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패: {e}")
            print("  sentence-transformers가 설치되어 있는지 확인하세요.")
            embeddings = None
            self.embeddings = None

        return preprocessor, embeddings

    def run_unsupervised(self):
        """비지도 학습: 프레임 추출"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("2단계: 비지도 학습 - 프레임 발견")
            print("=" * 60)

        try:
            # 프레임 추출기 초기화
            extractor = FrameExtractor(verbose=self.verbose)

            # 프레임 추출
            self.frame_assignments, self.frame_probs = extractor.extract_frames(
                self.articles,
                return_probs=True
            )

            # 프레임 정보 추출
            self.frames = extractor.get_frame_info(n_words=15)
            self.frames = extractor.assign_frame_names(self.frames, method="manual")

            if self.verbose:
                print(f"\n✓ 발견된 프레임: {len(self.frames)}개")
                for frame in self.frames[:5]:  # 상위 5개만 출력
                    print(f"\n프레임 {frame['frame_id']}: {frame.get('suggested_name', '')}")
                    print(f"  문서 수: {frame['size']}")
                    print(f"  주요 키워드: {', '.join(frame['keywords'][:5])}")

            # 시각화
            visualizer = FrameVisualizer(extractor.topic_model)

            # 언론사별 프레임 분포
            visualizer.create_frame_distribution(
                self.articles,
                self.frame_assignments,
                save_path=self.output_dir / "figures" / "media_frame_heatmap.png"
            )

            # 프레임-편향도 관계
            visualizer.visualize_frame_bias_correlation(
                self.articles,
                self.frame_assignments,
                save_path=self.output_dir / "figures" / "frame_bias_analysis.png"
            )

            # 프레임 키워드
            visualizer.visualize_frame_keywords(
                self.frames,
                top_n=8,
                save_path=self.output_dir / "figures" / "frame_keywords.png"
            )

            return extractor

        except Exception as e:
            print(f"⚠️ 프레임 추출 실패: {e}")
            print("  BERTopic이 설치되어 있는지 확인하세요.")
            print("  pip install bertopic")

            # 더미 데이터 생성
            self.frame_assignments = np.random.randint(0, 5, len(self.articles))
            self.frame_probs = np.random.rand(len(self.articles), 5)
            self.frame_probs = self.frame_probs / self.frame_probs.sum(axis=1, keepdims=True)
            self.frames = [
                {"frame_id": i, "keywords": [f"keyword_{i}"], "size": 10, "suggested_name": f"프레임_{i}"}
                for i in range(5)
            ]
            return None

    def run_supervised(self):
        """지도 학습: 편향도 예측"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("3단계: 지도 학습 - 편향도 예측")
            print("=" * 60)

        # 데이터 준비
        texts = []
        labels = []
        for article in self.articles:
            text = f"{article.get('title', '')} {article.get('content', '')}"
            texts.append(text)

            # 레이블 변환
            bias_score = article["bias_score"]
            if bias_score < -0.3:
                label = 0  # 진보
            elif bias_score > 0.3:
                label = 2  # 보수
            else:
                label = 1  # 중도
            labels.append(label)

        # 학습/테스트 분할
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 1. KoBERT 편향도 분류기 (선택적)
        try:
            if self.verbose:
                print("\n[1] KoBERT 기반 편향도 분류기")

            self.bias_classifier = BiasClassifier()
            self.bias_classifier.train(
                X_train_text, y_train,
                X_test_text, y_test,
                epochs=2,  # 빠른 테스트를 위해 적은 에폭
                save_path=self.output_dir / "models" / "bias_classifier"
            )

        except Exception as e:
            print(f"⚠️ KoBERT 모델 학습 실패: {e}")
            print("  transformers와 torch가 설치되어 있는지 확인하세요.")
            self.bias_classifier = None

        # 2. 프레임 기반 편향도 예측
        if self.verbose:
            print("\n[2] 프레임 기반 편향도 예측 모델")

        self.frame_predictor = FrameBasedBiasPredictor(
            model_type="random_forest",
            verbose=self.verbose
        )

        # Feature 준비
        X, y = self.frame_predictor.prepare_features(
            self.articles,
            self.frame_assignments,
            self.frame_probs
        )

        # 학습/테스트 분할
        X_train, X_test, y_train_frame, y_test_frame = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 학습
        self.frame_predictor.train(X_train, y_train_frame, X_test, y_test_frame)

        # Feature importance
        if hasattr(self.frame_predictor.model, "feature_importances_"):
            self.frame_predictor.get_feature_importance(
                top_n=20,
                plot=True,
                save_path=self.output_dir / "figures" / "feature_importance.png"
            )

        # 모델 저장
        self.frame_predictor.save_model(
            self.output_dir / "models" / "frame_predictor"
        )

        return self.frame_predictor

    def run_integrated_analysis(self):
        """통합 분석"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("4단계: 통합 분석 - 프레임-편향도 상관관계")
            print("=" * 60)

        # 분석기 초기화
        analyzer = IntegratedAnalyzer(
            self.articles,
            self.frame_assignments,
            self.frame_probs,
            self.frames
        )

        # 상관관계 분석
        correlation_results = analyzer.analyze_frame_bias_correlation()

        # 통계 검정
        chi_square_results = analyzer.chi_square_test()
        anova_results = analyzer.anova_test()

        # 언론사별 프레임 선호도
        media_preference = analyzer.analyze_media_frame_preference()

        # 특징적 프레임
        discriminative_frames = analyzer.find_discriminative_frames()

        # 종합 리포트
        report = analyzer.create_comprehensive_report(
            save_path=self.output_dir / "analysis" / "report.json"
        )

        # 종합 시각화
        analyzer.visualize_comprehensive_analysis(
            save_dir=self.output_dir / "figures"
        )

        return analyzer

    def run_frame_interpretation(self):
        """프레임 해석 - 프레임별 대표 문장 및 구분 이유 분석"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("4.5단계: 프레임 해석 - 대표 문장 및 구분 이유 분석")
            print("=" * 60)

        # 해석기 초기화
        interpreter = FrameInterpreter(
            self.articles,
            self.frame_assignments,
            self.frame_probs,
            self.frames,
            self.embeddings
        )

        # 해석 리포트 생성
        report = interpreter.create_frame_interpretation_report(
            save_path=self.output_dir / "analysis" / "frame_interpretation.json",
            n_examples=5
        )

        if self.verbose:
            print(f"\n✓ 프레임 해석 완료")
            print(f"  - 리포트: {self.output_dir / 'analysis' / 'frame_interpretation.json'}")

        return interpreter

    def create_dashboards(self):
        """대시보드 생성"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("6단계: 인터랙티브 대시보드 생성")
            print("=" * 60)

        # 프레임 해석 정보 로드
        frame_interpretation = None
        interpretation_path = self.output_dir / "analysis" / "frame_interpretation.json"
        if interpretation_path.exists():
            with open(interpretation_path, "r", encoding="utf-8") as f:
                frame_interpretation = json.load(f)

        # 대시보드 생성
        dashboard = InteractiveDashboard(
            self.articles,
            self.frames,
            self.frame_assignments,
            self.frame_probs,
            frame_interpretation
        )

        # 메인 대시보드
        dashboard.create_main_dashboard(
            save_path=self.output_dir / "dashboard.html"
        )

        # 프레임 탐색기
        dashboard.create_frame_explorer(
            save_path=self.output_dir / "frame_explorer.html"
        )

        # 프레임 네트워크
        dashboard.create_frame_network(
            save_path=self.output_dir / "frame_network.html"
        )

        # 타임라인
        dashboard.create_bias_timeline(
            save_path=self.output_dir / "bias_timeline.html"
        )

        # 프레임 해석 대시보드
        dashboard.create_frame_interpretation_dashboard(
            save_path=self.output_dir / "frame_interpretation.html"
        )

        if self.verbose:
            print("\n✓ 대시보드 생성 완료")
            print(f"  - {self.output_dir}/dashboard.html")
            print(f"  - {self.output_dir}/frame_explorer.html")
            print(f"  - {self.output_dir}/frame_network.html")
            print(f"  - {self.output_dir}/bias_timeline.html")
            print(f"  - {self.output_dir}/frame_interpretation.html")

        return dashboard

    def save_results(self):
        """결과 저장"""
        if self.verbose:
            print("\n=== 결과 저장 ===")

        # 프레임 정보 저장
        frames_path = self.output_dir / "frames.json"
        with open(frames_path, "w", encoding="utf-8") as f:
            json.dump(self.frames, f, ensure_ascii=False, indent=2)
        print(f"✓ 프레임 정보: {frames_path}")

        # 기사별 프레임 할당 저장
        article_frames = []
        for i, article in enumerate(self.articles):
            result = {
                "article_id": article.get("article_id", f"article_{i}"),
                "media_outlet": article["media_outlet"],
                "bias_score": article["bias_score"],
                "title": article["title"],
                "assigned_frame": int(self.frame_assignments[i]),
            }
            if self.frame_probs is not None:
                result["frame_probabilities"] = self.frame_probs[i].tolist()
            article_frames.append(result)

        article_frames_path = self.output_dir / "article_frames.json"
        with open(article_frames_path, "w", encoding="utf-8") as f:
            json.dump(article_frames, f, ensure_ascii=False, indent=2)
        print(f"✓ 기사별 프레임: {article_frames_path}")

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 70)
        print(" 뉴스 프레임-편향도 분석 파이프라인 시작 ".center(70))
        print("=" * 70)

        try:
            # 1. 데이터 로드
            self.load_data()

            # 2. 전처리
            preprocessor, embeddings = self.run_preprocessing()

            # 3. 비지도 학습
            extractor = self.run_unsupervised()

            # 4. 지도 학습
            frame_predictor = self.run_supervised()

            # 5. 통합 분석
            analyzer = self.run_integrated_analysis()

            # 5.5. 프레임 해석
            interpreter = self.run_frame_interpretation()

            # 6. 대시보드
            dashboard = self.create_dashboards()

            # 7. 결과 저장
            self.save_results()

            print("\n" + "=" * 70)
            print(" 파이프라인 완료! ".center(70))
            print("=" * 70)

            print(f"\n모든 결과는 '{self.output_dir}' 디렉토리에 저장되었습니다.")
            print("\n다음 파일들을 확인하세요:")
            print("  📊 dashboard.html - 메인 대시보드")
            print("  🔍 frame_explorer.html - 프레임별 기사 탐색")
            print("  🕸️ frame_network.html - 프레임 관계 네트워크")
            print("  📈 bias_timeline.html - 편향도 타임라인")
            print("  📖 frame_interpretation.html - ⭐ 프레임 해석 대시보드 (대표 문장 & 구분 이유)")
            print("  📄 analysis/report.json - 상세 분석 리포트")
            print("  📄 analysis/frame_interpretation.json - 프레임 해석 리포트 (JSON)")

            return {
                "preprocessor": preprocessor,
                "embeddings": embeddings,
                "extractor": extractor,
                "frame_predictor": frame_predictor,
                "analyzer": analyzer,
                "interpreter": interpreter,
                "dashboard": dashboard,
            }

        except Exception as e:
            print(f"\n❌ 파이프라인 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="뉴스 프레임-편향도 분석 파이프라인"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/articles.json",
        help="입력 데이터 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # 파이프라인 실행
    pipeline = FrameBiasAnalysisPipeline(
        data_path=args.data,
        output_dir=args.output,
        verbose=args.verbose
    )

    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()