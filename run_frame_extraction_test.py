#!/usr/bin/env python3
"""
실제 데이터로 프레임 추출 테스트
언론사명이 키워드에서 제외되는지 확인
"""

import json
import yaml
from pathlib import Path
from src.unsupervised.frame_extractor import extract_frames_from_json


def check_media_in_keywords():
    """추출된 프레임의 키워드에 언론사명이 있는지 확인"""

    # 프레임 추출 실행
    print("프레임 추출 시작...")
    try:
        extractor, topics, probs, frames = extract_frames_from_json(
            "data/input/articles.json",
            output_dir="results_test"
        )
    except Exception as e:
        print(f"프레임 추출 중 오류 발생: {e}")
        return

    # 언론사명 리스트 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    media_outlets = set(config.get("preprocessing", {}).get("media_outlets", []))

    # 실제 데이터에 있는 언론사
    with open("data/input/articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    actual_media = set([a['media_outlet'] for a in data['articles']])

    print("\n" + "=" * 70)
    print("언론사명 필터링 검증 결과")
    print("=" * 70)
    print(f"\n실제 데이터의 언론사: {actual_media}")
    print(f"\n필터링 대상 언론사명: {len(media_outlets)}개")

    # 각 프레임 검사
    found_media_in_keywords = False

    for frame in frames:
        print(f"\n프레임 {frame['frame_id']}:")
        print(f"  문서 수: {frame['size']}")
        print(f"  상위 키워드: {', '.join(frame['keywords'][:10])}")

        # 언론사명이 키워드에 있는지 체크
        media_in_keywords = []
        for keyword in frame['keywords']:
            # 빈 문자열 제외
            if not keyword or keyword.strip() == "":
                continue

            # 정확히 언론사명인 경우
            if keyword in media_outlets:
                media_in_keywords.append(keyword)
                continue

            # 언론사명으로 시작하는 경우 (조사가 붙은 경우)
            for media in media_outlets:
                if keyword.startswith(media):
                    media_in_keywords.append(f"{keyword}(언론사: {media})")
                    break

        if media_in_keywords:
            print(f"  ⚠️ 경고: 언론사명 또는 관련 단어가 키워드에 포함됨: {media_in_keywords}")
            found_media_in_keywords = True
        else:
            print(f"  ✅ 언론사명이 키워드에서 성공적으로 제외됨")

    print("\n" + "=" * 70)
    if found_media_in_keywords:
        print("⚠️ 일부 프레임에서 언론사명이 발견되었습니다. 추가 필터링이 필요합니다.")
    else:
        print("✅ 모든 프레임에서 언론사명이 성공적으로 제외되었습니다!")
    print("=" * 70)

    # 결과 파일 확인
    result_path = Path("results_test")
    if result_path.exists():
        print(f"\n결과 파일 저장 위치: {result_path}")
        print("  - frames.json: 프레임 정보")
        print("  - article_frames.json: 기사별 프레임 할당")


if __name__ == "__main__":
    check_media_in_keywords()