#!/usr/bin/env python3
"""
프레임 품질 분석 도구
프레임이 제대로 구분되었는지 진단합니다.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter


def analyze_frame_quality():
    """프레임 품질 분석"""

    print("=" * 70)
    print(" 프레임 품질 진단 ".center(70))
    print("=" * 70)

    # 데이터 로드
    with open("results/analysis/frame_interpretation.json", "r", encoding="utf-8") as f:
        interpretation = json.load(f)

    frames = interpretation["frame_interpretations"]

    print(f"\n총 프레임 수: {len(frames)}")
    print(f"총 기사 수: {interpretation['summary']['total_articles']}")

    # 각 프레임 분석
    print("\n" + "=" * 70)
    print(" 프레임별 상세 분석 ".center(70))
    print("=" * 70)

    issues = []

    for frame in frames:
        frame_id = frame["frame_id"]
        frame_name = frame["frame_name"]
        char = frame["characteristics"]

        print(f"\n[프레임 {frame_id}: {frame_name}]")
        print(f"  기사 수: {char['n_articles']}")
        print(f"  키워드: {', '.join(char['keywords'][:5])}")

        # 편향도 통계
        bias_stats = char["bias_stats"]
        print(f"\n  📊 편향도 통계:")
        print(f"    평균: {bias_stats['mean']:.3f}")
        print(f"    표준편차: {bias_stats['std']:.3f}")
        print(f"    범위: {bias_stats['min']:.2f} ~ {bias_stats['max']:.2f}")
        print(f"    일관성: {char['consistency']}")

        # 편향 분포
        bias_dist = char["bias_distribution"]
        print(f"\n  🎯 편향 분포:")
        for label, count in bias_dist.items():
            pct = count / char['n_articles'] * 100
            print(f"    {label}: {count}개 ({pct:.1f}%)")

        # 언론사 분포
        media_dist = char["media_distribution"]
        top_media = list(media_dist.items())[:3]
        print(f"\n  📰 주요 언론사:")
        for media, count in top_media:
            print(f"    {media}: {count}개")

        # 문제 진단
        print(f"\n  ⚠️ 진단:")

        # 1. 키워드 의미성 체크
        if any(kw.startswith("keyword_") for kw in char["keywords"][:5]):
            issue = f"프레임 {frame_id}: 키워드가 의미 없음 (형태소 분석 실패 가능성)"
            print(f"    ❌ {issue}")
            issues.append(issue)
        else:
            print(f"    ✅ 키워드가 의미 있음")

        # 2. 일관성 체크
        if bias_stats["std"] > 0.5:
            issue = f"프레임 {frame_id}: 일관성 매우 낮음 (std={bias_stats['std']:.3f})"
            print(f"    ❌ {issue}")
            issues.append(issue)
        elif bias_stats["std"] > 0.4:
            print(f"    ⚠️ 일관성 다소 낮음 (std={bias_stats['std']:.3f})")
        else:
            print(f"    ✅ 일관성 양호")

        # 3. 편향 분포 체크 (균등 분산 여부)
        max_group = max(bias_dist.values())
        total = sum(bias_dist.values())
        dominance = max_group / total

        if dominance < 0.4:  # 최대 그룹이 40% 미만
            issue = f"프레임 {frame_id}: 편향이 균등 분산됨 (프레임 구분 실패)"
            print(f"    ❌ {issue}")
            issues.append(issue)
        elif dominance < 0.5:
            print(f"    ⚠️ 편향 분포가 다소 분산됨 (지배 그룹 {dominance*100:.1f}%)")
        else:
            print(f"    ✅ 명확한 편향 성향 ({dominance*100:.1f}%)")

        # 4. 편향도 범위 체크
        bias_range = bias_stats["max"] - bias_stats["min"]
        if bias_range > 1.2:
            issue = f"프레임 {frame_id}: 편향도 범위가 너무 큼 ({bias_range:.2f})"
            print(f"    ❌ {issue}")
            issues.append(issue)
        elif bias_range > 0.8:
            print(f"    ⚠️ 편향도 범위가 다소 큼 ({bias_range:.2f})")
        else:
            print(f"    ✅ 편향도 범위 양호 ({bias_range:.2f})")

    # 전체 요약
    print("\n" + "=" * 70)
    print(" 전체 진단 요약 ".center(70))
    print("=" * 70)

    if issues:
        print(f"\n⚠️ 발견된 문제: {len(issues)}개\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

        print("\n" + "=" * 70)
        print(" 권장 조치 ".center(70))
        print("=" * 70)

        if any("키워드" in issue for issue in issues):
            print("\n1. ⚠️ 형태소 분석 문제")
            print("   → Mecab 설치 확인:")
            print("     python -c \"from konlpy.tag import Mecab; print(Mecab().morphs('최저임금'))\"")
            print("\n   macOS:")
            print("     brew install mecab mecab-ko mecab-ko-dic")
            print("\n   Ubuntu/Colab:")
            print("     !apt-get install -y mecab libmecab-dev mecab-ko mecab-ko-dic")

        if any("일관성" in issue or "분산" in issue for issue in issues):
            print("\n2. ⚠️ 프레임 구분 문제")
            print("   → config.yaml 수정:")
            print("     unsupervised:")
            print("       min_topic_size: 10  # 5 → 10")
            print("       nr_topics: 10       # auto → 10")
            print("       max_df: 0.7         # 0.8 → 0.7")
            print("\n   → 파이프라인 재실행:")
            print("     python src/pipeline.py")

        if any("범위" in issue for issue in issues):
            print("\n3. ⚠️ 데이터 품질 문제")
            print("   → 실제 뉴스 데이터 사용 권장 (최소 500개)")
            print("   → 또는 편향도 기반 후처리 추가")

    else:
        print("\n✅ 모든 프레임이 양호한 품질을 보입니다!")
        print("\n프레임 특성:")
        avg_std = np.mean([f["characteristics"]["bias_stats"]["std"] for f in frames])
        print(f"  - 평균 표준편차: {avg_std:.3f}")
        print(f"  - 대부분 프레임이 명확한 편향 성향을 가짐")
        print(f"  - 키워드가 의미 있음")

    print("\n" + "=" * 70)

    # 상세 리포트 저장
    report_path = Path("results/analysis/quality_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("프레임 품질 진단 리포트\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"총 프레임 수: {len(frames)}\n")
        f.write(f"총 기사 수: {interpretation['summary']['total_articles']}\n\n")

        if issues:
            f.write(f"발견된 문제: {len(issues)}개\n\n")
            for i, issue in enumerate(issues, 1):
                f.write(f"{i}. {issue}\n")
        else:
            f.write("✅ 모든 프레임이 양호한 품질을 보입니다!\n")

    print(f"\n상세 리포트 저장: {report_path}")


if __name__ == "__main__":
    if not Path("results/analysis/frame_interpretation.json").exists():
        print("⚠️ 프레임 해석 파일이 없습니다.")
        print("먼저 파이프라인을 실행하세요: python src/pipeline.py")
    else:
        analyze_frame_quality()
