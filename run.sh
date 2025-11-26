#!/bin/bash

# 뉴스 프레임 분석 시스템 실행 스크립트

echo "=================================================="
echo "     뉴스 프레임 분석 시스템"
echo "=================================================="
echo ""

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화..."
source venv/bin/activate

# 의존성 설치 확인
echo "📋 의존성 확인 중..."
pip install -q --upgrade pip

# 필수 패키지만 설치 (선택적)
echo "📥 필수 패키지 설치 중..."
pip install -q numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm

# config.yaml에서 입력 파일 경로 읽기
INPUT_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['data']['input_path'])" 2>/dev/null || echo "data/input/articles.json")

# 샘플 데이터 확인
if [ ! -f "$INPUT_PATH" ]; then
    echo "📝 입력 파일이 없습니다: $INPUT_PATH"
    echo "📝 샘플 데이터 생성 중..."
    python src/generate_sample_data.py
fi

# 메인 파이프라인 실행
echo ""
echo "🚀 분석 파이프라인 시작..."
echo ""
python src/pipeline.py "$@"

echo ""
echo "✅ 완료!"
echo ""
echo "결과 확인:"
echo "  📊 results/dashboard.html - 메인 대시보드"
echo "  🔍 results/frame_explorer.html - 프레임별 기사 탐색"
echo "  📈 results/figures/ - 시각화 결과"
echo ""

# 대시보드 열기 (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "브라우저에서 대시보드를 여시겠습니까? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        open results/dashboard.html
    fi
fi