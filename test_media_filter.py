#!/usr/bin/env python3
"""
언론사명 필터링 테스트
언론사명이 키워드로 추출되지 않는지 검증합니다.
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.unsupervised.frame_extractor import FrameExtractor


def test_text_preprocessor():
    """텍스트 전처리기의 언론사명 필터링 테스트"""
    print("\n=== 텍스트 전처리기 언론사명 필터링 테스트 ===\n")

    preprocessor = TextPreprocessor(use_mecab=False)  # 간단한 테스트를 위해 Mecab 사용 안함

    # 언론사명이 포함된 테스트 텍스트
    test_texts = [
        "조선일보가 보도한 최저임금 인상 소식입니다.",
        "한겨레 기자가 취재한 노동자 생계 문제입니다.",
        "경향신문에서 분석한 경제 영향 보고서입니다.",
        "KBS 뉴스에서 전한 정부 정책 발표입니다.",
        "최저임금이 인상되면 소상공인 부담이 늘어납니다."  # 언론사명 없는 텍스트
    ]

    for text in test_texts:
        print(f"원본: {text}")

        # 토큰화
        tokens = preprocessor.tokenize(text)
        print(f"토큰화: {tokens}")

        # 불용어 제거 (언론사명 포함)
        tokens_filtered = preprocessor.remove_stopwords(tokens)
        print(f"필터링 후: {tokens_filtered}")

        # 언론사명이 제거되었는지 확인
        media_found = []
        for token in tokens:
            if token in preprocessor.stopwords:
                media_found.append(token)

        if media_found:
            print(f"✅ 제거된 언론사명/불용어: {media_found}")

        print("-" * 50)


def test_frame_extractor():
    """프레임 추출기의 언론사명 필터링 테스트"""
    print("\n=== 프레임 추출기 언론사명 필터링 테스트 ===\n")

    # 샘플 문서 생성 (언론사명 포함)
    sample_docs = [
        {"title": "조선일보 최저임금 인상 특집", "content": "최저임금이 인상되면 기업 부담이 증가합니다."},
        {"title": "한겨레 노동자 생계 보도", "content": "노동자들의 생계가 개선될 것으로 기대됩니다."},
        {"title": "경향신문 경제 분석", "content": "최저임금 인상의 경제적 효과를 분석합니다."},
        {"title": "중앙일보 정책 비평", "content": "정부의 최저임금 정책에 대한 비평입니다."},
        {"title": "소상공인 부담 증가", "content": "자영업자와 소상공인의 인건비 부담이 늘어납니다."},
        {"title": "고용 시장 변화", "content": "최저임금 인상이 고용 시장에 미치는 영향을 살펴봅니다."},
    ]

    try:
        # 프레임 추출기 초기화
        extractor = FrameExtractor(
            min_topic_size=2,  # 테스트를 위해 작은 값 사용
            nr_topics=3,
            verbose=True
        )

        # 프레임 추출
        print("\n프레임 추출 중...")
        topics, _ = extractor.extract_frames(sample_docs)

        # 프레임 정보 추출
        frames = extractor.get_frame_info(n_words=10)

        print("\n=== 추출된 프레임 정보 ===\n")

        # 언론사명 체크
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        media_outlets = set(config.get("preprocessing", {}).get("media_outlets", []))

        for frame in frames:
            print(f"프레임 {frame['frame_id']}:")
            print(f"  문서 수: {frame['size']}")
            print(f"  키워드: {', '.join(frame['keywords'][:5])}")

            # 언론사명이 키워드에 있는지 체크
            media_in_keywords = [kw for kw in frame['keywords'] if kw in media_outlets]

            if media_in_keywords:
                print(f"  ⚠️ 경고: 언론사명이 키워드에 포함됨: {media_in_keywords}")
            else:
                print(f"  ✅ 언론사명이 키워드에서 제외됨")

            print()

    except Exception as e:
        print(f"⚠️ 프레임 추출 테스트 실패: {e}")
        print("  (BERTopic 패키지가 필요합니다)")


def verify_config():
    """설정 파일 확인"""
    print("\n=== 설정 파일 확인 ===\n")

    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    media_outlets = config.get("preprocessing", {}).get("media_outlets", [])
    stopwords = config.get("preprocessing", {}).get("stopwords", [])

    print(f"등록된 언론사명 수: {len(media_outlets)}개")
    print(f"언론사명 샘플: {media_outlets[:10]}")
    print(f"\n기본 불용어 수: {len(stopwords)}개")
    print(f"기본 불용어: {stopwords}")

    # 중요 언론사 포함 확인
    important_media = ["조선일보", "한겨레", "중앙일보", "경향신문", "KBS", "MBC", "JTBC"]
    missing = [m for m in important_media if m not in media_outlets]

    if missing:
        print(f"\n⚠️ 누락된 주요 언론사: {missing}")
    else:
        print(f"\n✅ 주요 언론사 모두 포함됨")


if __name__ == "__main__":
    print("=" * 70)
    print("언론사명 필터링 구현 상태 검증")
    print("=" * 70)

    # 1. 설정 파일 확인
    verify_config()

    # 2. 텍스트 전처리기 테스트
    test_text_preprocessor()

    # 3. 프레임 추출기 테스트
    test_frame_extractor()

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)