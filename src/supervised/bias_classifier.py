"""
편향도 분류 모듈
KoBERT를 사용하여 뉴스 기사의 정치적 편향도를 예측합니다.
"""

import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# 설정 파일 로드
config_path = Path(__file__).parent.parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}


class BiasDataset(Dataset):
    """편향도 분류용 데이터셋"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            texts: 텍스트 리스트
            labels: 레이블 리스트 (0: 진보, 1: 중도, 2: 보수)
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BiasClassifier:
    """KoBERT 기반 편향도 분류기"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 3,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: 사용할 모델 이름
            num_labels: 분류 클래스 수 (3: 진보/중도/보수)
            device: 디바이스 (cuda/cpu)
        """
        self.model_name = model_name or config.get("supervised", {}).get(
            "model_name", "klue/bert-base"  # KoBERT 대신 KLUE-BERT 사용 (더 안정적)
        )
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.bias_thresholds = config.get("supervised", {}).get(
            "bias_thresholds",
            {"progressive": -0.3, "conservative": 0.3}
        )

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            print(f"모델 로딩 중: {self.model_name}")
            print(f"디바이스: {self.device}")

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 모델 로드
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,  # 분류 헤드 크기 불일치 무시
            )
            self.model.to(self.device)

            print(f"✓ 모델 초기화 완료")

        except ImportError:
            print("⚠️ transformers가 설치되지 않았습니다.")
            print("  pip install transformers 명령으로 설치해주세요.")
            raise
        except Exception as e:
            print(f"⚠️ 모델 로딩 실패: {e}")
            raise

    def bias_to_label(self, bias_score: float) -> int:
        """
        편향도 점수를 레이블로 변환

        Args:
            bias_score: 편향도 점수 (-1 ~ 1)

        Returns:
            레이블 (0: 진보, 1: 중도, 2: 보수)
        """
        if bias_score < self.bias_thresholds["progressive"]:
            return 0  # 진보
        elif bias_score > self.bias_thresholds["conservative"]:
            return 2  # 보수
        else:
            return 1  # 중도

    def prepare_data(self, articles: List[Dict]) -> Tuple[List[str], List[int]]:
        """
        기사 데이터 준비

        Args:
            articles: 기사 리스트

        Returns:
            texts: 텍스트 리스트
            labels: 레이블 리스트
        """
        texts = []
        labels = []

        for article in articles:
            # 제목과 본문 결합
            title = article.get("title", "")
            content = article.get("content", "")
            text = f"{title} {content}".strip()

            # 레이블 변환
            bias_score = article.get("bias_score", 0)
            label = self.bias_to_label(bias_score)

            texts.append(text)
            labels.append(label)

        return texts, labels

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """
        모델 학습

        Args:
            train_texts: 학습 텍스트
            train_labels: 학습 레이블
            val_texts: 검증 텍스트
            val_labels: 검증 레이블
            epochs: 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
            save_path: 모델 저장 경로
        """
        if not self.model or not self.tokenizer:
            raise ValueError("모델이 초기화되지 않았습니다.")

        # 하이퍼파라미터 설정
        epochs = epochs or config.get("supervised", {}).get("num_epochs", 5)
        batch_size = batch_size or config.get("supervised", {}).get("batch_size", 16)
        learning_rate = learning_rate or config.get("supervised", {}).get(
            "learning_rate", 5e-5
        )

        print(f"\n=== 학습 설정 ===")
        print(f"에폭 수: {epochs}")
        print(f"배치 크기: {batch_size}")
        print(f"학습률: {learning_rate}")
        print(f"학습 데이터: {len(train_texts)}개")
        if val_texts:
            print(f"검증 데이터: {len(val_texts)}개")

        # 데이터셋 생성
        train_dataset = BiasDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_texts and val_labels:
            val_dataset = BiasDataset(val_texts, val_labels, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 학습
        self.model.train()
        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\n에폭 {epoch + 1}/{epochs}")

            # 학습
            train_loss = 0
            train_preds = []
            train_labels_list = []

            for batch in tqdm(train_loader, desc="학습"):
                # 데이터 준비
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 순전파
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # 역전파
                loss.backward()
                optimizer.step()

                # 기록
                train_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

            # 학습 지표
            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels_list, train_preds)
            train_f1 = f1_score(train_labels_list, train_preds, average="macro")

            print(f"학습 손실: {avg_train_loss:.4f}")
            print(f"학습 정확도: {train_acc:.4f}")
            print(f"학습 F1: {train_f1:.4f}")

            # 검증
            if val_loader:
                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                print(f"검증 손실: {val_loss:.4f}")
                print(f"검증 정확도: {val_acc:.4f}")
                print(f"검증 F1: {val_f1:.4f}")

                # 최고 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        self.save_model(save_path)
                        print(f"✓ 최고 모델 저장: {save_path}")

        print("\n✓ 학습 완료")

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        모델 평가

        Args:
            data_loader: 데이터 로더

        Returns:
            loss: 평균 손실
            accuracy: 정확도
            f1: F1 스코어
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        self.model.train()
        return avg_loss, accuracy, f1

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        편향도 예측

        Args:
            texts: 예측할 텍스트 리스트

        Returns:
            예측 결과 리스트
        """
        if not self.model or not self.tokenizer:
            raise ValueError("모델이 초기화되지 않았습니다.")

        self.model.eval()
        predictions = []

        for text in tqdm(texts, desc="예측"):
            # 토크나이징
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # 예측
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred = torch.argmax(logits, dim=-1).item()

            # 결과 저장
            label_names = ["진보", "중도", "보수"]
            predictions.append({
                "label": pred,
                "label_name": label_names[pred],
                "probabilities": probs.cpu().numpy().tolist(),
                "prob_progressive": float(probs[0]),
                "prob_neutral": float(probs[1]),
                "prob_conservative": float(probs[2]),
            })

        return predictions

    def save_model(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 모델과 토크나이저 저장
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # 설정 저장
        config_path = path / "bias_classifier_config.json"
        config_data = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        path = Path(path)

        # 모델과 토크나이저 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

        # 설정 로드
        config_path = path / "bias_classifier_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
                self.num_labels = config_data.get("num_labels", 3)
                self.bias_thresholds = config_data.get("bias_thresholds", {})

        print(f"✓ 모델 로드 완료: {path}")


def train_bias_classifier(
    json_path: str = "data/input/articles.json",
    model_save_path: str = "models/bias_classifier",
    test_size: float = 0.2,
):
    """
    편향도 분류기 학습 (테스트/실행용)

    Args:
        json_path: 데이터 경로
        model_save_path: 모델 저장 경로
        test_size: 테스트 데이터 비율
    """
    # 데이터 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"]
    print(f"로드된 기사 수: {len(articles)}")

    # 분류기 초기화
    classifier = BiasClassifier()

    # 데이터 준비
    texts, labels = classifier.prepare_data(articles)

    # 학습/검증 분할
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # 학습
    classifier.train(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        epochs=3,  # 테스트를 위해 적은 수로 설정
        save_path=model_save_path,
    )

    # 테스트 예측
    print("\n=== 테스트 예측 ===")
    test_texts = val_texts[:5]
    predictions = classifier.predict(test_texts)

    for i, pred in enumerate(predictions):
        print(f"\n텍스트 {i+1}: {test_texts[i][:100]}...")
        print(f"  예측: {pred['label_name']}")
        print(f"  확률: 진보={pred['prob_progressive']:.3f}, "
              f"중도={pred['prob_neutral']:.3f}, "
              f"보수={pred['prob_conservative']:.3f}")

    return classifier


if __name__ == "__main__":
    # 테스트 실행
    if Path("data/input/articles.json").exists():
        train_bias_classifier()
    else:
        print("⚠️ 데이터 파일을 찾을 수 없습니다.")
        print("  먼저 generate_sample_data.py를 실행하세요.")