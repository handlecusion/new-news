#!/usr/bin/env python3
"""
프레임 해석 도구
기존 프레임 추출 결과를 바탕으로 각 프레임의 특성을 분석합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.frame_interpreter import interpret_frames
from src import config


def main():
    """프레임 해석 실행"""

    # 필요한 파일 확인
    input_path = config.get_input_path()
    required_files = [
        input_path,
        "results/frames.json",
        "results/article_frames.json"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("⚠️ 다음 파일들이 필요합니다:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n먼저 python src/pipeline.py를 실행하여 프레임을 추출하세요.")
        return

    print("=" * 60)
    print("프레임 해석 도구")
    print("=" * 60)
    print("\n각 프레임의 특성과 대표 문장을 분석합니다.")
    print("프레임이 왜 이렇게 구분되었는지 이해할 수 있습니다.\n")

    # 프레임 해석 실행
    interpreter, report = interpret_frames()

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print("\n결과 파일:")
    print("  📖 results/analysis/frame_interpretation.json")
    print("\n이 파일에서 다음 정보를 확인할 수 있습니다:")
    print("  • 각 프레임의 주요 키워드")
    print("  • 프레임 성향 (진보/중도/보수)")
    print("  • 프레임 구분 이유")
    print("  • 대표 기사와 핵심 문장 예시")
    print("  • 언론사별 분포")


if __name__ == "__main__":
    main()
