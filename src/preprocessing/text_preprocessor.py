"""
텍스트 전처리 모듈
한국어 뉴스 기사를 정제하고 토큰화합니다.
"""

import re
import yaml
from typing import List, Optional
from pathlib import Path

# 설정 파일 로드
config_path = Path(__file__).parent.parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}


class TextPreprocessor:
    """한국어 텍스트 전처리 클래스"""

    def __init__(self, use_mecab: bool = True):
        """
        Args:
            use_mecab: Mecab 사용 여부 (False일 경우 간단한 공백 분리 사용)
        """
        self.use_mecab = use_mecab
        self.mecab = None
        self.stopwords = set(
            config.get("preprocessing", {}).get(
                "stopwords",
                ["있다", "하다", "되다", "이다", "것", "수", "등", "및", "위해", "대한"]
            )
        )

        if use_mecab:
            try:
                from konlpy.tag import Mecab
                self.mecab = Mecab()
                print("✓ Mecab 토크나이저 초기화 완료")
            except Exception as e:
                print(f"⚠️ Mecab 초기화 실패: {e}")
                print("  간단한 토크나이저를 사용합니다.")
                self.use_mecab = False

    def clean_text(self, text: str) -> str:
        """
        텍스트 정제

        Args:
            text: 원본 텍스트

        Returns:
            정제된 텍스트
        """
        if not text:
            return ""

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # URL 제거
        text = re.sub(r"http\S+|www.\S+", "", text)

        # 이메일 제거
        text = re.sub(r"\S+@\S+", "", text)

        # 특수문자 제거 (한글, 영문, 숫자, 일부 구두점만 유지)
        text = re.sub(r"[^가-힣a-zA-Z0-9\s.,!?()'\"-]", " ", text)

        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text)

        # 양쪽 공백 제거
        text = text.strip()

        return text

    def tokenize(self, text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
        """
        형태소 분석 및 토큰화

        Args:
            text: 입력 텍스트
            pos_filter: 추출할 품사 태그 리스트 (기본값: 명사, 동사, 형용사)

        Returns:
            토큰 리스트
        """
        if not text:
            return []

        # 텍스트 정제
        text = self.clean_text(text)

        if self.use_mecab and self.mecab:
            # Mecab을 사용한 형태소 분석
            if pos_filter is None:
                # 명사(N), 동사(V), 형용사(VA)만 추출
                pos_filter = ["N", "V", "VA"]

            tokens = []
            try:
                pos_tagged = self.mecab.pos(text)
                for word, pos in pos_tagged:
                    # 품사 태그의 첫 글자 또는 첫 두 글자로 필터링
                    if any(pos.startswith(filter_pos) for filter_pos in pos_filter):
                        # 동사/형용사의 경우 원형 추출
                        if pos.startswith("V"):
                            # 동사 원형화 (간단한 규칙)
                            if word.endswith("다"):
                                tokens.append(word)
                            else:
                                tokens.append(word + "다")
                        else:
                            tokens.append(word)
            except Exception as e:
                print(f"⚠️ Mecab 토큰화 실패: {e}")
                # 폴백: 공백 기반 분리
                tokens = text.split()

            return tokens
        else:
            # Mecab을 사용하지 않는 경우 간단한 토큰화
            return text.split()

    def tokenize_nouns(self, text: str) -> List[str]:
        """
        명사만 추출

        Args:
            text: 입력 텍스트

        Returns:
            명사 토큰 리스트
        """
        return self.tokenize(text, pos_filter=["N"])

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        불용어 제거

        Args:
            tokens: 토큰 리스트

        Returns:
            불용어가 제거된 토큰 리스트
        """
        return [token for token in tokens if token not in self.stopwords]

    def preprocess(
        self,
        text: str,
        clean: bool = True,
        tokenize: bool = True,
        remove_stopwords: bool = True,
        pos_filter: Optional[List[str]] = None,
    ) -> str | List[str]:
        """
        전체 전처리 파이프라인

        Args:
            text: 입력 텍스트
            clean: 텍스트 정제 여부
            tokenize: 토큰화 여부
            remove_stopwords: 불용어 제거 여부
            pos_filter: 추출할 품사 태그

        Returns:
            전처리된 텍스트 또는 토큰 리스트
        """
        # 1. 텍스트 정제
        if clean:
            text = self.clean_text(text)

        # 2. 토큰화
        if tokenize:
            tokens = self.tokenize(text, pos_filter=pos_filter)

            # 3. 불용어 제거
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)

            return tokens
        else:
            return text

    def preprocess_for_bert(self, text: str, max_length: Optional[int] = None) -> str:
        """
        BERT 모델용 전처리 (토큰화하지 않고 정제만)

        Args:
            text: 입력 텍스트
            max_length: 최대 길이 (문자 수)

        Returns:
            정제된 텍스트
        """
        # 텍스트 정제
        text = self.clean_text(text)

        # 최대 길이 제한
        if max_length and len(text) > max_length:
            text = text[:max_length]

        return text


def test_preprocessor():
    """전처리기 테스트"""
    preprocessor = TextPreprocessor()

    # 테스트 텍스트
    test_texts = [
        "최저임금이 9,860원으로 인상되었습니다.",
        "노동자들의 생계가 나아질 것으로 기대됩니다.",
        "<p>HTML 태그가 포함된 텍스트입니다.</p>",
        "이메일 test@example.com과 URL http://example.com 제거 테스트",
    ]

    print("\n=== 전처리기 테스트 ===\n")

    for text in test_texts:
        print(f"원본: {text}")
        print(f"정제: {preprocessor.clean_text(text)}")

        tokens = preprocessor.tokenize(text)
        print(f"토큰화: {tokens}")

        tokens_no_stop = preprocessor.remove_stopwords(tokens)
        print(f"불용어 제거: {tokens_no_stop}")

        nouns = preprocessor.tokenize_nouns(text)
        print(f"명사만: {nouns}")

        print("-" * 50)


if __name__ == "__main__":
    test_preprocessor()