# 뉴스 프레임 분석: 비지도/지도 학습 하이브리드 접근

## 프로젝트 개요

### 목적
단일 이슈에 대한 뉴스 기사들을 분석하여:
1. **비지도 학습:** 데이터로부터 자동으로 프레임 패턴을 발견
2. **지도 학습:** 언론사 편향도를 기반으로 편향 예측 모델 학습
3. **통합 분석:** 발견된 프레임과 편향도의 관계 규명

### 핵심 연구 질문
1. 동일 이슈에 대해 어떤 프레임들이 자동으로 추출되는가?
2. 추출된 프레임과 언론사 편향도 간에 어떤 관계가 있는가?
3. 프레임 정보를 활용하여 편향도를 예측할 수 있는가?

### 데이터 형식
```json
{
  "articles": [
    {
      "article_id": "20240115_hankyoreh_001",
      "media_outlet": "한겨레",
      "bias_score": -0.7,
      "title": "최저임금 9,860원 확정... 노동자 생계 안정 기대",
      "content": "2024년도 최저임금이 시간당 9,860원으로 최종 확정되었다...",
      "published_date": "2024-01-15",
      "url": "https://..."
    },
    ...
  ]
}
```

---

## 전체 파이프라인
```
┌─────────────────────────────────────────────────────────┐
│                    JSON 데이터 입력                        │
│              (단일 이슈, 100-200개 기사)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  데이터 전처리                             │
│  - 텍스트 정제, 형태소 분석, 임베딩 생성                     │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  비지도 학습 경로  │    │   지도 학습 경로   │
│  (프레임 발견)    │    │  (편향도 예측)     │
└────────┬─────────┘    └─────────┬────────┘
         │                        │
         │                        │
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│ 프레임 클러스터링  │    │ 편향도 분류 모델   │
│ - Topic Modeling │    │ - KoBERT 기반     │
│ - Clustering     │    │ - 언론사 라벨     │
└────────┬─────────┘    └─────────┬────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
         ┌─────────────────────────┐
         │    통합 분석 및 시각화     │
         │ - 프레임-편향도 상관관계   │
         │ - 인터랙티브 대시보드      │
         └─────────────────────────┘
```

---

## Part 1: 데이터 전처리

### 입력 데이터 구조
```python
# data/input/articles.json
{
  "metadata": {
    "issue": "최저임금 인상",
    "collection_period": "2024-01-01 ~ 2024-01-31",
    "total_articles": 150
  },
  "articles": [...]
}
```

### 전처리 파이프라인

#### Step 1: 텍스트 정제
```python
import re
from konlpy.tag import Mecab

class TextPreprocessor:
    def __init__(self):
        self.mecab = Mecab()
        
    def clean_text(self, text):
        """텍스트 정제"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 제거 (일부 보존)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """형태소 분석 및 토큰화"""
        # 명사, 동사, 형용사만 추출
        tokens = self.mecab.pos(text)
        filtered = [
            word for word, pos in tokens 
            if pos.startswith('N') or pos.startswith('V') or pos.startswith('VA')
        ]
        return filtered
    
    def remove_stopwords(self, tokens):
        """불용어 제거"""
        stopwords = {'있다', '하다', '되다', '이다', '것'}  # 확장 필요
        return [t for t in tokens if t not in stopwords]
```

#### Step 2: 문서 임베딩 생성
```python
from sentence_transformers import SentenceTransformer

class DocumentEmbedder:
    def __init__(self, model_name='jhgan/ko-sroberta-multitask'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """문서들을 벡터로 변환"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings  # shape: (n_docs, 768)
```

---

## Part 2: 비지도 학습 - 프레임 발견

### 접근 방법 비교

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **LDA** | 토픽 모델링 | 해석 용이, 토픽별 키워드 제공 | 문맥 파악 약함 |
| **BERTopic** | BERT + 클러스터링 | 의미적 유사도 활용, 시각화 우수 | 계산 비용 높음 |
| **K-Means** | 임베딩 기반 클러스터링 | 빠름, 간단함 | 클러스터 수 사전 지정 필요 |
| **HDBSCAN** | 밀도 기반 클러스터링 | 클러스터 수 자동, outlier 처리 | 파라미터 튜닝 필요 |

### 권장 접근: BERTopic

#### 왜 BERTopic인가?
1. **의미 기반:** BERT 임베딩으로 문맥적 유사도 포착
2. **자동 토픽 수:** HDBSCAN으로 자동 결정
3. **해석 가능:** c-TF-IDF로 토픽당 대표 키워드 추출
4. **시각화:** 인터랙티브 시각화 내장

#### 구현
```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Mecab
import pandas as pd

class FrameExtractor:
    def __init__(self, language='korean'):
        # 한국어 토크나이저
        mecab = Mecab()
        
        # Custom vectorizer (명사만 추출)
        self.vectorizer = CountVectorizer(
            tokenizer=lambda x: [
                word for word, pos in mecab.pos(x) 
                if pos.startswith('N')
            ],
            min_df=2,
            max_df=0.8
        )
        
        # BERTopic 모델
        self.topic_model = BERTopic(
            embedding_model='jhgan/ko-sroberta-multitask',
            vectorizer_model=self.vectorizer,
            min_topic_size=5,  # 최소 5개 문서
            nr_topics='auto',  # 자동으로 토픽 수 결정
            calculate_probabilities=True
        )
    
    def extract_frames(self, documents):
        """프레임(토픽) 추출"""
        # 제목 + 본문 결합
        texts = [
            f"{doc['title']} {doc['content']}" 
            for doc in documents
        ]
        
        # 토픽 모델링
        topics, probs = self.topic_model.fit_transform(texts)
        
        return topics, probs
    
    def get_frame_info(self, n_words=10):
        """각 프레임의 대표 키워드 추출"""
        topic_info = self.topic_model.get_topic_info()
        
        frames = []
        for topic_id in topic_info['Topic']:
            if topic_id == -1:  # outliers
                continue
            
            words = self.topic_model.get_topic(topic_id)[:n_words]
            frame = {
                'frame_id': topic_id,
                'keywords': [word for word, score in words],
                'keyword_scores': [score for word, score in words],
                'size': topic_info[topic_info['Topic'] == topic_id]['Count'].values[0]
            }
            frames.append(frame)
        
        return frames
    
    def assign_frame_names(self, frames):
        """프레임에 의미있는 이름 부여 (수동 또는 LLM 활용)"""
        # Option 1: 수동 라벨링
        # 연구자가 키워드를 보고 직접 명명
        
        # Option 2: LLM 활용 (GPT-4, Claude 등)
        for frame in frames:
            keywords = ', '.join(frame['keywords'][:5])
            prompt = f"다음 키워드들로 구성된 뉴스 프레임의 이름을 한국어로 짧게 지어주세요: {keywords}"
            # frame['name'] = call_llm(prompt)
            pass
        
        return frames
```

#### 시각화
```python
class FrameVisualizer:
    def __init__(self, topic_model):
        self.topic_model = topic_model
    
    def visualize_topics(self):
        """토픽 간 관계 시각화"""
        fig = self.topic_model.visualize_topics()
        fig.write_html("results/figures/topic_map.html")
    
    def visualize_barchart(self, top_n=8):
        """토픽별 키워드 바차트"""
        fig = self.topic_model.visualize_barchart(
            top_n_topics=top_n,
            n_words=10
        )
        fig.write_html("results/figures/topic_keywords.html")
    
    def visualize_hierarchy(self):
        """토픽 계층 구조"""
        hierarchical_topics = self.topic_model.hierarchical_topics(documents)
        fig = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        fig.write_html("results/figures/topic_hierarchy.html")
    
    def create_frame_distribution(self, articles, topics):
        """언론사별 프레임 분포"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = pd.DataFrame({
            'media': [a['media_outlet'] for a in articles],
            'bias': [a['bias_score'] for a in articles],
            'frame': topics
        })
        
        # 히트맵
        pivot = df.pivot_table(
            index='media',
            columns='frame',
            aggfunc='size',
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('언론사별 프레임 분포')
        plt.xlabel('프레임 ID')
        plt.ylabel('언론사')
        plt.tight_layout()
        plt.savefig('results/figures/media_frame_heatmap.png', dpi=300)
```

### 프레임 해석 가이드

발견된 프레임을 해석하기 위한 체크리스트:
```python
def interpret_frame(frame_keywords, sample_articles):
    """
    프레임 해석 가이드
    
    질문:
    1. 이 키워드들이 공통적으로 강조하는 것은 무엇인가?
    2. 어떤 관점/시각을 반영하는가?
    3. 누구의 이익/손실을 다루는가?
    4. 어떤 감정적 톤을 가지는가? (긍정/부정/중립)
    5. 원인/결과 중 무엇을 강조하는가?
    """
    
    interpretation = {
        'dominant_perspective': '',  # 주된 관점
        'stakeholder': '',           # 주요 이해관계자
        'emotional_tone': '',        # 감정 톤
        'focus': '',                 # 초점 (원인/결과/해결책)
        'suggested_name': ''         # 제안 프레임 이름
    }
    
    return interpretation
```

---

## Part 3: 지도 학습 - 편향도 예측

### 데이터 준비
```python
def prepare_supervised_data(articles):
    """지도 학습용 데이터 준비"""
    
    # 편향도를 범주형으로 변환
    def bias_to_label(score):
        if score < -0.3:
            return 0  # 진보
        elif score > 0.3:
            return 2  # 보수
        else:
            return 1  # 중도
    
    data = []
    for article in articles:
        data.append({
            'text': f"{article['title']} {article['content']}",
            'bias_score': article['bias_score'],
            'bias_label': bias_to_label(article['bias_score']),
            'media_outlet': article['media_outlet']
        })
    
    return pd.DataFrame(data)
```

### 모델 1: KoBERT 기반 편향도 분류기
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)

class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BiasClassifier:
    def __init__(self, model_name='skt/kobert-base-v1', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """모델 학습"""
        
        # 데이터셋 생성
        train_dataset = BiasDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = BiasDataset(val_texts, val_labels, self.tokenizer)
        
        # 학습 설정
        training_args = TrainingArguments(
            output_dir='./models/bias_classifier',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss'
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # 학습
        trainer.train()
        
        return trainer
    
    def predict(self, texts):
        """편향도 예측"""
        self.model.eval()
        
        predictions = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            predictions.append({
                'label': pred,
                'probabilities': probs.tolist()
            })
        
        return predictions
    
    @staticmethod
    def compute_metrics(eval_pred):
        """평가 지표 계산"""
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        
        return {
            'accuracy': acc,
            'f1': f1
        }
```

### 모델 2: 프레임 정보를 활용한 편향도 예측
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class FrameBasedBiasPredictor:
    """프레임 분포를 feature로 사용하는 편향도 예측기"""
    
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
    
    def prepare_features(self, articles, frame_assignments, frame_probs):
        """프레임 정보를 feature로 변환"""
        
        n_frames = frame_probs.shape[1]
        
        features = []
        labels = []
        
        for i, article in enumerate(articles):
            # Feature: 각 프레임에 대한 확률 분포
            frame_vector = frame_probs[i]
            
            # 추가 feature: 가장 강한 프레임
            dominant_frame = frame_assignments[i]
            one_hot = np.zeros(n_frames)
            if dominant_frame >= 0:
                one_hot[dominant_frame] = 1
            
            # Feature 결합
            feature = np.concatenate([frame_vector, one_hot])
            features.append(feature)
            
            # Label
            bias_score = article['bias_score']
            if bias_score < -0.3:
                label = 0  # 진보
            elif bias_score > 0.3:
                label = 2  # 보수
            else:
                label = 1  # 중도
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def train(self, X_train, y_train):
        """모델 학습"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_pred = self.model.predict(X_test)
        
        print("=== Classification Report ===")
        print(classification_report(
            y_test, y_pred,
            target_names=['진보', '중도', '보수']
        ))
        
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature Importance (Random Forest만)
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def predict_proba(self, X):
        """확률값 예측"""
        return self.model.predict_proba(X)
```

---

## Part 4: 통합 분석

### 프레임-편향도 상관관계 분석
```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

class IntegratedAnalyzer:
    def __init__(self, articles, frame_assignments, frame_probs):
        self.df = pd.DataFrame({
            'media': [a['media_outlet'] for a in articles],
            'bias_score': [a['bias_score'] for a in articles],
            'bias_label': [self._bias_to_label(a['bias_score']) for a in articles],
            'frame': frame_assignments
        })
        
        # 프레임 확률 추가
        for i in range(frame_probs.shape[1]):
            self.df[f'frame_{i}_prob'] = frame_probs[:, i]
    
    @staticmethod
    def _bias_to_label(score):
        if score < -0.3:
            return '진보'
        elif score > 0.3:
            return '보수'
        else:
            return '중도'
    
    def analyze_frame_bias_correlation(self):
        """프레임-편향도 상관관계 분석"""
        
        results = []
        
        # 각 프레임별로 편향도와의 상관관계 계산
        frame_cols = [col for col in self.df.columns if col.startswith('frame_') and col.endswith('_prob')]
        
        for frame_col in frame_cols:
            frame_id = frame_col.split('_')[1]
            corr, p_value = pearsonr(self.df['bias_score'], self.df[frame_col])
            
            results.append({
                'frame_id': int(frame_id),
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('correlation', key=abs, ascending=False)
        
        return result_df
    
    def create_contingency_table(self):
        """교차표 생성 및 카이제곱 검정"""
        
        # 교차표
        contingency = pd.crosstab(
            self.df['bias_label'],
            self.df['frame'],
            normalize='index'  # 행 기준 정규화
        )
        
        # 카이제곱 검정
        chi2, p_value, dof, expected = chi2_contingency(
            pd.crosstab(self.df['bias_label'], self.df['frame'])
        )
        
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        return contingency
    
    def visualize_frame_bias_relationship(self, frame_names=None):
        """프레임-편향도 관계 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 히트맵: 편향 그룹별 프레임 분포
        contingency = pd.crosstab(
            self.df['bias_label'],
            self.df['frame'],
            normalize='index'
        )
        
        if frame_names:
            contingency.columns = [frame_names.get(c, f'Frame {c}') for c in contingency.columns]
        
        sns.heatmap(
            contingency,
            annot=True,
            fmt='.2%',
            cmap='RdYlBu_r',
            ax=axes[0, 0],
            cbar_kws={'label': '비율'}
        )
        axes[0, 0].set_title('편향 그룹별 프레임 분포')
        axes[0, 0].set_xlabel('프레임')
        axes[0, 0].set_ylabel('편향 그룹')
        
        # 2. 박스플롯: 프레임별 편향도 분포
        self.df[self.df['frame'] >= 0].boxplot(
            column='bias_score',
            by='frame',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('프레임별 편향도 분포')
        axes[0, 1].set_xlabel('프레임 ID')
        axes[0, 1].set_ylabel('편향도')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)
        
        # 3. 산점도: 주요 프레임들의 확률 vs 편향도
        frame_cols = [col for col in self.df.columns if col.startswith('frame_') and col.endswith('_prob')][:3]
        
        for i, frame_col in enumerate(frame_cols):
            frame_id = frame_col.split('_')[1]
            axes[1, 0].scatter(
                self.df['bias_score'],
                self.df[frame_col],
                alpha=0.6,
                label=f'Frame {frame_id}'
            )
        
        axes[1, 0].set_xlabel('편향도')
        axes[1, 0].set_ylabel('프레임 확률')
        axes[1, 0].set_title('편향도 vs 프레임 확률 (상위 3개 프레임)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 바이올린 플롯: 각 프레임의 편향도 분포
        valid_df = self.df[self.df['frame'] >= 0]
        sns.violinplot(
            data=valid_df,
            x='frame',
            y='bias_score',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('프레임별 편향도 분포 (바이올린 플롯)')
        axes[1, 1].set_xlabel('프레임 ID')
        axes[1, 1].set_ylabel('편향도')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('results/figures/integrated_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
```

### 인터랙티브 대시보드
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class InteractiveDashboard:
    def __init__(self, articles, frames, frame_assignments, frame_probs):
        self.articles = articles
        self.frames = frames
        self.frame_assignments = frame_assignments
        self.frame_probs = frame_probs
    
    def create_dashboard(self):
        """인터랙티브 대시보드 생성"""
        
        # 데이터 준비
        df = pd.DataFrame({
            'media': [a['media_outlet'] for a in self.articles],
            'bias_score': [a['bias_score'] for a in self.articles],
            'frame': self.frame_assignments,
            'title': [a['title'] for a in self.articles]
        })
        
        # Figure 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '프레임 분포',
                '편향도 분포',
                '프레임-편향도 관계',
                '언론사별 프레임 사용'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # 1. 프레임 분포
        frame_counts = df['frame'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=frame_counts.index,
                y=frame_counts.values,
                name='프레임 빈도',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. 편향도 분포
        fig.add_trace(
            go.Histogram(
                x=df['bias_score'],
                nbinsx=20,
                name='편향도',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. 프레임-편향도 관계 (산점도)
        for frame_id in df['frame'].unique():
            if frame_id < 0:  # outliers 제외
                continue
            
            frame_df = df[df['frame'] == frame_id]
            fig.add_trace(
                go.Scatter(
                    x=frame_df['bias_score'],
                    y=[frame_id] * len(frame_df),
                    mode='markers',
                    name=f'Frame {frame_id}',
                    text=frame_df['title'],
                    hovertemplate='<b>%{text}</b><br>편향도: %{x:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. 언론사별 프레임 사용
        media_frame = df.groupby(['media', 'frame']).size().unstack(fill_value=0)
        for frame_id in media_frame.columns:
            if frame_id < 0:
                continue
            
            fig.add_trace(
                go.Bar(
                    x=media_frame.index,
                    y=media_frame[frame_id],
                    name=f'Frame {frame_id}'
                ),
                row=2, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="뉴스 프레임 분석 대시보드"
        )
        
        fig.update_xaxes(title_text="프레임 ID", row=1, col=1)
        fig.update_xaxes(title_text="편향도", row=1, col=2)
        fig.update_xaxes(title_text="편향도", row=2, col=1)
        fig.update_xaxes(title_text="언론사", row=2, col=2)
        
        fig.update_yaxes(title_text="빈도", row=1, col=1)
        fig.update_yaxes(title_text="빈도", row=1, col=2)
        fig.update_yaxes(title_text="프레임 ID", row=2, col=1)
        fig.update_yaxes(title_text="기사 수", row=2, col=2)
        
        # 저장
        fig.write_html('results/dashboard.html')
        
        return fig
    
    def create_frame_explorer(self):
        """프레임별 기사 탐색기"""
        
        df = pd.DataFrame({
            'media': [a['media_outlet'] for a in self.articles],
            'bias_score': [a['bias_score'] for a in self.articles],
            'frame': self.frame_assignments,
            'title': [a['title'] for a in self.articles],
            'content': [a['content'][:200] + '...' for a in self.articles]
        })
        
        # 프레임별 키워드 정보 추가
        frame_info = {f['frame_id']: ', '.join(f['keywords'][:5]) for f in self.frames}
        df['frame_keywords'] = df['frame'].map(frame_info)
        
        # 인터랙티브 테이블
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['프레임 ID', '키워드', '언론사', '편향도', '제목'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    df['frame'],
                    df['frame_keywords'],
                    df['media'],
                    df['bias_score'].round(2),
                    df['title']
                ],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title='프레임별 기사 탐색',
            height=600
        )
        
        fig.write_html('results/frame_explorer.html')
        
        return fig
```

---

## Part 5: 메인 실행 파이프라인
```python
import json
from sklearn.model_selection import train_test_split

class FrameBiasAnalysisPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.articles = None
        self.frames = None
        self.frame_assignments = None
        self.frame_probs = None
        self.bias_classifier = None
    
    def load_data(self):
        """JSON 데이터 로드"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.articles = data['articles']
        print(f"✓ Loaded {len(self.articles)} articles")
    
    def run_unsupervised(self):
        """비지도 학습: 프레임 추출"""
        print("\n=== 비지도 학습: 프레임 발견 ===")
        
        # 프레임 추출
        extractor = FrameExtractor()
        self.frame_assignments, self.frame_probs = extractor.extract_frames(self.articles)
        
        # 프레임 정보
        self.frames = extractor.get_frame_info()
        
        print(f"✓ Discovered {len(self.frames)} frames")
        
        # 프레임 출력
        for frame in self.frames:
            print(f"\nFrame {frame['frame_id']} (size: {frame['size']})")
            print(f"  Keywords: {', '.join(frame['keywords'][:10])}")
        
        # 시각화
        visualizer = FrameVisualizer(extractor.topic_model)
        visualizer.visualize_topics()
        visualizer.visualize_barchart()
        print("\n✓ Visualizations saved to results/figures/")
        
        return extractor
    
    def run_supervised(self):
        """지도 학습: 편향도 예측"""
        print("\n=== 지도 학습: 편향도 예측 ===")
        
        # 데이터 준비
        df = prepare_supervised_data(self.articles)
        
        # Train/Test 분할
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['bias_label'])
        
        # 모델 1: KoBERT 기반
        print("\n[1] KoBERT 기반 분류 모델 학습...")
        self.bias_classifier = BiasClassifier()
        trainer = self.bias_classifier.train(
            train_df['text'].tolist(),
            train_df['bias_label'].tolist(),
            test_df['text'].tolist(),
            test_df['bias_label'].tolist()
        )
        print("✓ KoBERT 모델 학습 완료")
        
        # 모델 2: 프레임 기반
        print("\n[2] 프레임 기반 예측 모델 학습...")
        frame_predictor = FrameBasedBiasPredictor()
        
        X, y = frame_predictor.prepare_features(
            self.articles,
            self.frame_assignments,
            self.frame_probs
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        frame_predictor.train(X_train, y_train)
        feature_importance = frame_predictor.evaluate(X_test, y_test)
        
        # Feature Importance 시각화
        if feature_importance is not None:
            self._plot_feature_importance(feature_importance)
        
        print("✓ 프레임 기반 모델 학습 완료")
        
        return frame_predictor
    
    def run_integrated_analysis(self):
        """통합 분석"""
        print("\n=== 통합 분석: 프레임-편향도 상관관계 ===")
        
        analyzer = IntegratedAnalyzer(
            self.articles,
            self.frame_assignments,
            self.frame_probs
        )
        
        # 상관관계 분석
        corr_results = analyzer.analyze_frame_bias_correlation()
        print("\n프레임-편향도 상관관계:")
        print(corr_results.to_string(index=False))
        
        # 교차표 및 카이제곱 검정
        print("\n교차표 분석:")
        contingency = analyzer.create_contingency_table()
        print(contingency)
        
        # 시각화
        frame_names = {f['frame_id']: ', '.join(f['keywords'][:3]) for f in self.frames}
        analyzer.visualize_frame_bias_relationship(frame_names)
        
        print("\n✓ 통합 분석 완료. 결과는 results/figures/에 저장됨")
        
        return analyzer
    
    def create_dashboard(self):
        """대시보드 생성"""
        print("\n=== 인터랙티브 대시보드 생성 ===")
        
        dashboard = InteractiveDashboard(
            self.articles,
            self.frames,
            self.frame_assignments,
            self.frame_probs
        )
        
        dashboard.create_dashboard()
        dashboard.create_frame_explorer()
        
        print("✓ 대시보드 생성 완료: results/dashboard.html")
        print("✓ 프레임 탐색기: results/frame_explorer.html")
    
    def save_results(self):
        """결과 저장"""
        print("\n=== 결과 저장 ===")
        
        # 프레임 정보 저장
        with open('results/frames.json', 'w', encoding='utf-8') as f:
            json.dump(self.frames, f, ensure_ascii=False, indent=2)
        
        # 기사별 프레임 할당 저장
        results = []
        for i, article in enumerate(self.articles):
            results.append({
                'article_id': article.get('article_id', f'article_{i}'),
                'media_outlet': article['media_outlet'],
                'bias_score': article['bias_score'],
                'assigned_frame': int(self.frame_assignments[i]),
                'frame_probabilities': self.frame_probs[i].tolist()
            })
        
        with open('results/article_frames.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("✓ 결과 저장 완료:")
        print("  - results/frames.json")
        print("  - results/article_frames.json")
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("뉴스 프레임-편향도 분석 파이프라인 시작")
        print("=" * 60)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 비지도 학습
        extractor = self.run_unsupervised()
        
        # 3. 지도 학습
        frame_predictor = self.run_supervised()
        
        # 4. 통합 분석
        analyzer = self.run_integrated_analysis()
        
        # 5. 대시보드
        self.create_dashboard()
        
        # 6. 결과 저장
        self.save_results()
        
        print("\n" + "=" * 60)
        print("파이프라인 완료!")
        print("=" * 60)
    
    def _plot_feature_importance(self, importance):
        """Feature Importance 시각화"""
        import matplotlib.pyplot as plt
        
        n_frames = len(self.frames)
        frame_importance = importance[:n_frames]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(frame_importance)), frame_importance)
        plt.xlabel('프레임 ID')
        plt.ylabel('Feature Importance')
        plt.title('프레임별 편향도 예측 기여도')
        plt.tight_layout()
        plt.savefig('results/figures/feature_importance.png', dpi=300)
        plt.close()

# 실행
if __name__ == '__main__':
    pipeline = FrameBiasAnalysisPipeline('data/input/articles.json')
    pipeline.run_full_pipeline()
```

---

## 프로젝트 구조
```
news-frame-bias-analysis/
│
├── data/
│   ├── input/
│   │   └── articles.json          # 입력 데이터
│   └── processed/
│       └── embeddings.npy         # 문서 임베딩 캐시
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_preprocessor.py   # 텍스트 전처리
│   │   └── embedder.py            # 문서 임베딩
│   │
│   ├── unsupervised/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py     # BERTopic 기반 프레임 추출
│   │   └── visualizer.py          # 프레임 시각화
│   │
│   ├── supervised/
│   │   ├── __init__.py
│   │   ├── bias_classifier.py     # KoBERT 편향도 분류
│   │   └── frame_predictor.py     # 프레임 기반 예측
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── correlation.py         # 상관관계 분석
│   │   └── dashboard.py           # 대시보드
│   │
│   └── pipeline.py                # 메인 파이프라인
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_unsupervised_frames.ipynb
│   ├── 03_supervised_bias.ipynb
│   └── 04_integrated_analysis.ipynb
│
├── models/
│   ├── bias_classifier/           # 학습된 KoBERT 모델
│   └── frame_predictor.pkl        # 프레임 기반 예측 모델
│
├── results/
│   ├── frames.json                # 발견된 프레임 정보
│   ├── article_frames.json        # 기사별 프레임 할당
│   ├── figures/                   # 시각화 결과
│   │   ├── topic_map.html
│   │   ├── topic_keywords.html
│   │   ├── integrated_analysis.png
│   │   └── feature_importance.png
│   ├── dashboard.html             # 인터랙티브 대시보드
│   └── frame_explorer.html        # 프레임 탐색기
│
├── requirements.txt
├── README.md
└── config.yaml
```

---

## 설치 및 실행

### requirements.txt
```
# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# NLP
konlpy>=0.6.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0

# Topic Modeling
bertopic>=0.15.0
hdbscan>=0.8.29
umap-learn>=0.5.3

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.14.0

# Utils
tqdm>=4.65.0
pyyaml>=6.0
```

### 실행 방법
```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비
# data/input/articles.json 파일 배치

# 3. 전체 파이프라인 실행
python src/pipeline.py

# 4. 결과 확인
# results/ 디렉토리의 시각화 및 대시보드 확인
```

---

## 예상 결과물

### 1. 발견된 프레임 (예시)
```json
[
  {
    "frame_id": 0,
    "keywords": ["노동자", "생계", "생활임금", "최저임금", "실질소득"],
    "size": 35,
    "suggested_name": "노동자 생계 보장 프레임"
  },
  {
    "frame_id": 1,
    "keywords": ["소상공인", "자영업자", "부담", "인건비", "경영악화"],
    "size": 42,
    "suggested_name": "소상공인 부담 프레임"
  },
  {
    "frame_id": 2,
    "keywords": ["고용", "일자리", "감소", "축소", "실업"],
    "size": 28,
    "suggested_name": "고용 감소 우려 프레임"
  }
]
```

### 2. 프레임-편향도 상관관계 (예시)
```
frame_id  correlation  p_value  significant
0         -0.68        0.0001   True         # 진보 성향
1         +0.72        0.0001   True         # 보수 성향
2         +0.54        0.0032   True         # 보수 성향
3         -0.12        0.1542   False        # 상관 없음
```

### 3. 편향도 예측 성능 (예시)
```
KoBERT 모델:
  Accuracy: 0.78
  F1-score (macro): 0.76

프레임 기반 모델:
  Accuracy: 0.71
  F1-score (macro): 0.68
```

---

## 후속 연구 방향

1. **멀티 이슈 확장:** 여러 이슈에 대해 동일 분석 수행
2. **시계열 분석:** 시간에 따른 프레임 변화 추적
3. **프레임 전이:** 한 프레임에서 다른 프레임으로의 변환 모델링
4. **실시간 분석:** 신규 기사에 대한 실시간 프레임/편향 분석

---

## 연락처

- **프로젝트 리더:** 섬놈
- **GitHub:** [repo_url]

---

**문서 버전:** 1.0  
**최종 수정일:** 2025-01-21  
**상태:** 개발 준비 완료