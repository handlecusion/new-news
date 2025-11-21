#!/usr/bin/env python3
"""
샘플 뉴스 데이터 생성기
최저임금 인상 관련 샘플 기사 데이터를 생성합니다.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

# 언론사별 정치적 성향 (편향도)
MEDIA_BIAS = {
    "한겨레": -0.7,
    "경향신문": -0.6,
    "오마이뉴스": -0.8,
    "프레시안": -0.75,
    "JTBC": -0.4,
    "MBC": -0.3,
    "KBS": -0.1,
    "SBS": 0.0,
    "YTN": 0.1,
    "연합뉴스": 0.0,
    "중앙일보": 0.4,
    "동아일보": 0.5,
    "조선일보": 0.7,
    "매일경제": 0.6,
    "한국경제": 0.65,
}

# 프레임별 제목 템플릿
FRAME_TITLES = {
    "노동자_생계": [
        "최저임금 {amount}원, 노동자 생계 {sentiment}",
        "'{amount}원' 최저임금, 서민 생활 {sentiment}",
        "최저임금 인상, 노동자 삶의 질 {sentiment}",
    ],
    "소상공인_부담": [
        "최저임금 {amount}원, 소상공인 {sentiment}",
        "자영업자들 '최저임금 부담 {sentiment}'",
        "소상공인 인건비 부담 {sentiment}",
    ],
    "고용_영향": [
        "최저임금 인상, 고용시장 {sentiment}",
        "일자리 {sentiment}, 최저임금 영향 분석",
        "최저임금과 고용률 {sentiment}",
    ],
    "경제_효과": [
        "최저임금 {amount}원, 경제 {sentiment}",
        "내수 경제 {sentiment}, 최저임금 효과",
        "경제 성장과 최저임금 {sentiment}",
    ],
}

# 프레임별 본문 템플릿
FRAME_CONTENTS = {
    "노동자_생계": {
        "진보": [
            "최저임금 인상은 노동자들의 기본적인 생계를 보장하는 최소한의 조치다. 물가 상승률을 고려하면 실질 임금은 여전히 부족한 수준이며, 노동자들의 삶의 질 개선을 위해서는 추가적인 인상이 필요하다.",
            "이번 최저임금 인상으로 저임금 노동자들의 생활 안정에 도움이 될 것으로 기대된다. 하지만 여전히 OECD 평균에 미치지 못하는 수준으로, 노동자의 인간다운 삶을 위해서는 지속적인 개선이 요구된다.",
        ],
        "보수": [
            "급격한 최저임금 인상이 오히려 저임금 노동자들의 일자리를 위협할 수 있다는 우려가 제기되고 있다. 시장 원리를 무시한 인위적 임금 인상은 부작용을 초래할 수 있다.",
            "최저임금 인상이 모든 노동자에게 도움이 되는 것은 아니다. 일자리를 잃거나 근로시간이 줄어드는 노동자들도 발생할 수 있어 신중한 접근이 필요하다.",
        ],
    },
    "소상공인_부담": {
        "진보": [
            "정부의 소상공인 지원 정책과 함께 시행되는 최저임금 인상은 충분히 감당 가능한 수준이다. 일부 어려움은 있겠지만, 노동자의 구매력 증가가 결국 소상공인에게도 도움이 될 것이다.",
            "소상공인의 어려움을 최저임금 탓으로만 돌리는 것은 문제의 본질을 회피하는 것이다. 임대료, 카드 수수료 등 다른 구조적 문제 해결이 우선되어야 한다.",
        ],
        "보수": [
            "영세 자영업자들은 최저임금 인상으로 인한 인건비 부담 증가로 큰 어려움을 겪고 있다. 많은 소상공인들이 직원을 줄이거나 폐업을 고려하는 상황이다.",
            "최저임금의 급격한 인상은 소상공인과 자영업자들에게 과도한 부담을 주고 있다. 이들의 경영 악화는 결국 경제 전체에 악영향을 미칠 수 있다.",
        ],
    },
    "고용_영향": {
        "진보": [
            "최저임금 인상이 고용에 미치는 부정적 영향은 제한적이라는 연구 결과가 나왔다. 오히려 노동자의 생산성 향상과 이직률 감소 등 긍정적 효과가 관찰되고 있다.",
            "고용 감소를 우려하는 목소리가 있지만, 실제 데이터를 보면 최저임금 인상 이후에도 고용은 꾸준히 증가하고 있다. 기업들의 과도한 우려는 기우에 불과하다.",
        ],
        "보수": [
            "최저임금 인상으로 인한 고용 감소가 현실화되고 있다. 특히 청년층과 고령층의 일자리가 크게 줄어들어 취약계층이 더 큰 피해를 보고 있다.",
            "인건비 부담 증가로 기업들이 신규 채용을 기피하고 있다. 자동화와 무인화 추세가 가속화되면서 일자리 자체가 사라지는 현상이 나타나고 있다.",
        ],
    },
    "경제_효과": {
        "진보": [
            "최저임금 인상은 내수 경제 활성화에 기여할 것으로 예상된다. 저소득층의 소비 여력이 늘어나면서 경제 전체에 긍정적인 파급 효과를 가져올 것이다.",
            "소득 주도 성장의 핵심인 최저임금 인상이 경제 선순환 구조를 만들어낼 것이다. 노동자의 소득 증가가 소비 증가로 이어져 기업 매출 상승에도 도움이 될 것이다.",
        ],
        "보수": [
            "최저임금의 급격한 인상이 기업 경쟁력을 약화시키고 있다. 인건비 부담 증가로 투자 여력이 줄어들어 경제 성장에 악영향을 미칠 우려가 크다.",
            "시장 원리를 무시한 인위적 임금 인상은 경제 전체의 효율성을 떨어뜨린다. 기업들의 해외 이전과 투자 위축으로 경제 활력이 저하되고 있다.",
        ],
    },
}


def generate_article(
    article_id: str,
    media_outlet: str,
    bias_score: float,
    frame: str,
    date: str,
) -> Dict:
    """단일 기사 생성"""

    # 편향도에 따른 성향 결정
    if bias_score < -0.3:
        sentiment_type = "진보"
        sentiment_words = ["개선", "기대", "희망", "필요", "당연"]
    elif bias_score > 0.3:
        sentiment_type = "보수"
        sentiment_words = ["우려", "부담", "위험", "문제", "과도"]
    else:
        sentiment_type = "중도"
        sentiment_words = ["변화", "영향", "관찰", "분석", "평가"]

    # 제목 생성
    title_template = random.choice(FRAME_TITLES[frame])
    title = title_template.format(
        amount="9,860",
        sentiment=random.choice(sentiment_words)
    )

    # 본문 생성
    if sentiment_type in ["진보", "보수"]:
        content = random.choice(FRAME_CONTENTS[frame][sentiment_type])
    else:  # 중도는 진보/보수 내용을 섞어서 사용
        all_contents = (
            FRAME_CONTENTS[frame].get("진보", []) +
            FRAME_CONTENTS[frame].get("보수", [])
        )
        content = random.choice(all_contents) if all_contents else "최저임금 인상에 대한 다양한 의견이 제시되고 있다."

    # 추가 문장 생성
    additional_sentences = [
        f"2024년 최저임금은 시간당 9,860원으로 결정되었다.",
        f"{media_outlet}은 이번 최저임금 결정에 대해 심층 분석을 진행했다.",
        f"관련 부처는 최저임금 인상의 영향을 면밀히 모니터링할 예정이다.",
        f"전문가들은 다양한 의견을 제시하고 있다.",
    ]

    content = content + " " + " ".join(random.sample(additional_sentences, 2))

    return {
        "article_id": article_id,
        "media_outlet": media_outlet,
        "bias_score": bias_score,
        "title": title,
        "content": content,
        "published_date": date,
        "url": f"https://example.com/{article_id}",
    }


def generate_sample_data(num_articles: int = 150) -> Dict:
    """샘플 데이터셋 생성"""

    articles = []
    media_outlets = list(MEDIA_BIAS.keys())
    frames = list(FRAME_TITLES.keys())

    # 시작 날짜
    start_date = datetime(2024, 1, 1)

    for i in range(num_articles):
        # 언론사 선택
        media = random.choice(media_outlets)
        bias = MEDIA_BIAS[media]

        # 프레임 선택 (편향도에 따라 가중치 부여)
        if bias < -0.3:  # 진보
            frame_weights = [0.4, 0.1, 0.3, 0.2]  # 노동자_생계 강조
        elif bias > 0.3:  # 보수
            frame_weights = [0.1, 0.4, 0.3, 0.2]  # 소상공인_부담 강조
        else:  # 중도
            frame_weights = [0.25, 0.25, 0.25, 0.25]  # 균등

        frame = random.choices(frames, weights=frame_weights)[0]

        # 날짜 생성
        days_offset = random.randint(0, 30)
        pub_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

        # 기사 ID 생성
        article_id = f"{pub_date.replace('-', '')}_{media.replace(' ', '_')}_{i:03d}"

        # 기사 생성
        article = generate_article(article_id, media, bias, frame, pub_date)
        articles.append(article)

    # 최종 데이터 구조
    data = {
        "metadata": {
            "issue": "최저임금 인상",
            "collection_period": "2024-01-01 ~ 2024-01-31",
            "total_articles": len(articles),
        },
        "articles": articles,
    }

    return data


def main():
    """메인 실행 함수"""
    print("샘플 데이터 생성 중...")

    # 150개 기사 생성
    sample_data = generate_sample_data(150)

    # JSON 파일로 저장
    output_path = "data/input/articles.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(sample_data['articles'])}개 샘플 기사 생성 완료")
    print(f"✓ 저장 위치: {output_path}")

    # 통계 출력
    media_counts = {}
    frame_counts = {}

    for article in sample_data["articles"]:
        media = article["media_outlet"]
        media_counts[media] = media_counts.get(media, 0) + 1

        # 제목에서 프레임 추정 (간단한 키워드 매칭)
        title = article["title"]
        if "노동자" in title or "생계" in title:
            frame = "노동자_생계"
        elif "소상공인" in title or "자영업" in title:
            frame = "소상공인_부담"
        elif "고용" in title or "일자리" in title:
            frame = "고용_영향"
        else:
            frame = "경제_효과"

        frame_counts[frame] = frame_counts.get(frame, 0) + 1

    print("\n언론사별 기사 수:")
    for media, count in sorted(media_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {media}: {count}개")

    print("\n프레임별 기사 수 (추정):")
    for frame, count in sorted(frame_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {frame}: {count}개")


if __name__ == "__main__":
    main()