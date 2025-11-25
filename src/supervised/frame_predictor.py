"""
프레임 기반 편향도 예측 모듈
프레임 분포를 feature로 사용하여 편향도를 예측합니다.
"""

import json
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 설정 모듈 로드
from src import config


class FrameBasedBiasPredictor:
    """프레임 분포를 feature로 사용하는 편향도 예측기"""

    def __init__(
        self,
        model_type: str = "random_forest",
        normalize_features: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            model_type: 모델 종류 (random_forest, logistic, gradient_boosting)
            normalize_features: feature 정규화 여부
            verbose: 로그 출력 여부
        """
        self.model_type = model_type
        self.normalize_features = normalize_features
        self.verbose = verbose

        self.model = None
        self.scaler = StandardScaler() if normalize_features else None
        self.feature_names = []
        self.bias_thresholds = config.get("supervised", {}).get(
            "bias_thresholds",
            {"progressive": -0.3, "conservative": 0.3}
        )

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class="ovr",
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")

        if self.verbose:
            print(f"✓ {self.model_type} 모델 초기화 완료")

    def prepare_features(
        self,
        articles: List[Dict],
        frame_assignments: np.ndarray,
        frame_probs: Optional[np.ndarray] = None,
        additional_features: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        프레임 정보를 feature로 변환

        Args:
            articles: 기사 리스트
            frame_assignments: 프레임 할당 (n_articles,)
            frame_probs: 프레임 확률 분포 (n_articles, n_frames)
            additional_features: 추가 feature

        Returns:
            X: feature 행렬
            y: 레이블 배열
        """
        features = []
        labels = []

        # 프레임 수 확인
        n_frames = frame_probs.shape[1] if frame_probs is not None else 0

        # Feature 이름 설정
        self.feature_names = []

        if frame_probs is not None:
            # 프레임 확률 분포를 feature로 사용
            for i in range(n_frames):
                self.feature_names.append(f"frame_{i}_prob")

        # 주요 프레임 one-hot encoding
        unique_frames = np.unique(frame_assignments[frame_assignments >= 0])
        for frame_id in unique_frames:
            self.feature_names.append(f"is_frame_{frame_id}")

        # 추가 feature 이름
        if additional_features:
            self.feature_names.extend(additional_features.keys())

        # 각 기사에 대해 feature 생성
        for i, article in enumerate(articles):
            feature = []

            # 프레임 확률 분포
            if frame_probs is not None:
                feature.extend(frame_probs[i])

            # 주요 프레임 one-hot
            one_hot = np.zeros(len(unique_frames))
            if frame_assignments[i] >= 0:
                frame_idx = np.where(unique_frames == frame_assignments[i])[0]
                if len(frame_idx) > 0:
                    one_hot[frame_idx[0]] = 1
            feature.extend(one_hot)

            # 추가 feature
            if additional_features:
                for key in additional_features:
                    if i < len(additional_features[key]):
                        feature.append(additional_features[key][i])
                    else:
                        feature.append(0)

            features.append(feature)

            # 레이블 (편향도 -> 카테고리)
            bias_score = article["bias_score"]
            if bias_score < self.bias_thresholds["progressive"]:
                label = 0  # 진보
            elif bias_score > self.bias_thresholds["conservative"]:
                label = 2  # 보수
            else:
                label = 1  # 중도
            labels.append(label)

        X = np.array(features)
        y = np.array(labels)

        if self.verbose:
            print(f"Feature 생성 완료: shape={X.shape}")
            print(f"레이블 분포: 진보={np.sum(y==0)}, 중도={np.sum(y==1)}, 보수={np.sum(y==2)}")

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        모델 학습

        Args:
            X_train: 학습 feature
            y_train: 학습 레이블
            X_val: 검증 feature
            y_val: 검증 레이블
        """
        if self.verbose:
            print(f"\n=== {self.model_type} 모델 학습 ===")
            print(f"학습 데이터: {X_train.shape}")

        # Feature 정규화
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        # 모델 학습
        self.model.fit(X_train, y_train)

        # 학습 성능
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="macro")

        print(f"학습 정확도: {train_acc:.4f}")
        print(f"학습 F1: {train_f1:.4f}")

        # 검증 성능
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")

            print(f"검증 정확도: {val_acc:.4f}")
            print(f"검증 F1: {val_f1:.4f}")

            # 상세 결과
            if self.verbose:
                print("\n=== Classification Report ===")
                print(classification_report(
                    y_val, y_val_pred,
                    target_names=["진보", "중도", "보수"]
                ))

        # Cross validation (학습 데이터만 사용)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"\n교차검증 F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_confusion_matrix: bool = True,
    ) -> Dict:
        """
        모델 평가

        Args:
            X_test: 테스트 feature
            y_test: 테스트 레이블
            return_confusion_matrix: 혼동 행렬 반환 여부

        Returns:
            평가 결과 딕셔너리
        """
        # Feature 정규화
        if self.scaler:
            X_test = self.scaler.transform(X_test)

        # 예측
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results = {
            "accuracy": accuracy,
            "f1_macro": f1,
            "predictions": y_pred,
            "probabilities": y_proba,
        }

        if self.verbose:
            print("\n=== 평가 결과 ===")
            print(f"정확도: {accuracy:.4f}")
            print(f"F1 (macro): {f1:.4f}")

            print("\n=== Classification Report ===")
            print(classification_report(
                y_test, y_pred,
                target_names=["진보", "중도", "보수"]
            ))

        if return_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            results["confusion_matrix"] = cm

            if self.verbose:
                print("\n=== Confusion Matrix ===")
                print("        예측")
                print("        진보  중도  보수")
                print("실제")
                labels = ["진보", "중도", "보수"]
                for i, label in enumerate(labels):
                    print(f"{label:4}   {cm[i, 0]:4} {cm[i, 1]:4} {cm[i, 2]:4}")

        return results

    def get_feature_importance(
        self,
        top_n: int = 20,
        plot: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Feature 중요도 분석

        Args:
            top_n: 상위 feature 수
            plot: 시각화 여부
            save_path: 저장 경로

        Returns:
            feature 중요도 데이터프레임
        """
        if not hasattr(self.model, "feature_importances_"):
            print("⚠️ 이 모델은 feature importance를 지원하지 않습니다.")
            return None

        # Feature 중요도 추출
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        if self.verbose:
            print("\n=== Feature Importance (Top 20) ===")
            print(importance_df.head(top_n).to_string(index=False))

        if plot:
            # 시각화
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)

            plt.barh(range(len(top_features)), top_features["importance"])
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Feature importance 저장: {save_path}")
                plt.close()
            else:
                plt.show()

        return importance_df

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측

        Args:
            X: feature 행렬

        Returns:
            예측 레이블
        """
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        예측 확률

        Args:
            X: feature 행렬

        Returns:
            예측 확률
        """
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        model_path = path / f"{self.model_type}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # 스케일러 저장
        if self.scaler:
            scaler_path = path / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        # 설정 저장
        config_path = path / "frame_predictor_config.json"
        config_data = {
            "model_type": self.model_type,
            "normalize_features": self.normalize_features,
            "feature_names": self.feature_names,
            "bias_thresholds": self.bias_thresholds,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ 모델 저장 완료: {path}")

    def load_model(self, path: str):
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        path = Path(path)

        # 설정 로드
        config_path = path / "frame_predictor_config.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)
            self.model_type = config_data["model_type"]
            self.feature_names = config_data["feature_names"]
            self.bias_thresholds = config_data.get("bias_thresholds", {})

        # 모델 로드
        model_path = path / f"{self.model_type}_model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # 스케일러 로드
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        print(f"✓ 모델 로드 완료: {path}")


def compare_models(
    articles: List[Dict],
    frame_assignments: np.ndarray,
    frame_probs: np.ndarray,
    test_size: float = 0.2,
) -> Dict:
    """
    여러 모델 비교

    Args:
        articles: 기사 리스트
        frame_assignments: 프레임 할당
        frame_probs: 프레임 확률
        test_size: 테스트 크기

    Returns:
        모델별 성능 딕셔너리
    """
    results = {}
    model_types = ["random_forest", "logistic", "gradient_boosting"]

    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"{model_type.upper()} 모델")
        print(f"{'=' * 50}")

        # 모델 초기화
        predictor = FrameBasedBiasPredictor(model_type=model_type)

        # Feature 준비
        X, y = predictor.prepare_features(articles, frame_assignments, frame_probs)

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 학습
        predictor.train(X_train, y_train, X_test, y_test)

        # 평가
        eval_results = predictor.evaluate(X_test, y_test)

        results[model_type] = {
            "accuracy": eval_results["accuracy"],
            "f1_macro": eval_results["f1_macro"],
            "predictor": predictor,
        }

    # 결과 요약
    print("\n" + "=" * 50)
    print("모델 비교 결과")
    print("=" * 50)

    comparison_df = pd.DataFrame(results).T[["accuracy", "f1_macro"]]
    print(comparison_df.to_string())

    return results


if __name__ == "__main__":
    # 테스트: 프레임 추출 결과가 있는 경우
    input_path = config.get_input_path()
    if (
        Path(input_path).exists()
        and Path("results/article_frames.json").exists()
    ):
        # 데이터 로드
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data["articles"]

        with open("results/article_frames.json", "r", encoding="utf-8") as f:
            article_frames = json.load(f)

        # 프레임 정보 복원
        frame_assignments = np.array([af["assigned_frame"] for af in article_frames])

        # 더미 프레임 확률 생성 (실제로는 BERTopic 결과 사용)
        n_frames = len(np.unique(frame_assignments[frame_assignments >= 0]))
        frame_probs = np.random.rand(len(articles), n_frames)
        frame_probs = frame_probs / frame_probs.sum(axis=1, keepdims=True)

        # 모델 비교
        compare_models(articles, frame_assignments, frame_probs)

    else:
        print("⚠️ 필요한 파일을 찾을 수 없습니다.")
        print("  1. generate_sample_data.py 실행")
        print("  2. frame_extractor.py 실행")
        print("  이후 이 스크립트를 실행하세요.")