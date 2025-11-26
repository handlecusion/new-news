import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup as bs
import requests
import re

# 환경 변수 로드
load_dotenv()


class NaverNewsCrawler:
    """네이버 뉴스 API를 이용한 뉴스 크롤러"""

    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.base_url = "https://openapi.naver.com/v1/search/news.json"

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 .env 파일에 설정해주세요."
            )

    def search_news(
        self, query: str, display: int = 100, start: int = 1, sort: str = "date"
    ) -> Dict:
        """
        네이버 뉴스 API를 통해 뉴스 검색

        Args:
            query: 검색어 (UTF-8 인코딩 필요)
            display: 한 번에 표시할 검색 결과 개수 (기본값: 100, 최대: 100)
            start: 검색 시작 위치 (기본값: 1, 최대: 1000)
            sort: 정렬 방법 (sim: 정확도순, date: 날짜순)

        Returns:
            Dict: API 응답 결과
        """
        # HTTP 요청 헤더 설정
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

        # 쿼리 파라미터 설정
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort,
        }

        try:
            response = requests.get(
                self.base_url, headers=headers, params=params, timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error Code: {response.status_code}")
                return None
        except Exception as e:
            print(f"API 요청 중 오류 발생: {str(e)}")
            return None

    def convert_to_article_format(
        self,
        items: List[Dict],
        issue: str,
        media_bias_map: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        네이버 뉴스 API 응답을 articles.json 형식으로 변환

        Args:
            items: 네이버 뉴스 API의 items 리스트
            issue: 이슈/주제명
            media_bias_map: 언론사별 편향 점수 매핑 (Optional)

        Returns:
            Dict: articles.json 형식의 데이터
        """
        if media_bias_map is None:
            # 기본 편향 점수 설정 (필요시 수정)
            media_bias_map = {
                "조선일보": 0.7,
                "중앙일보": 0.4,
                "동아일보": 0.5,
                "한국경제": 0.65,
                "매일경제": 0.6,
                "연합뉴스": 0.0,
                "KBS": -0.1,
                "MBC": -0.3,
                "SBS": 0.0,
                "JTBC": -0.4,
                "YTN": 0.1,
                "한겨레": -0.7,
                "경향신문": -0.6,
                "오마이뉴스": -0.8,
                "프레시안": -0.75,
            }

        articles = []
        media_counter = {}  # 언론사별 카운터

        for idx, item in enumerate(items):
            # HTML 태그 제거
            title = self._remove_html_tags(item.get("title", ""))
            description = self._remove_html_tags(item.get("description", ""))

            # 언론사 추출 (originallink에서 도메인 추출)
            media_outlet = self._extract_media_outlet(item.get("originallink", ""))
            if media_outlet == "기타":
                continue

            # 날짜 변환 (RFC 형식 -> YYYY-MM-DD)
            pub_date = self._convert_date(item.get("pubDate", ""))

            # 편향 점수 가져오기
            bias_score = media_bias_map.get(media_outlet, 0.0)

            # 언론사별 카운터 증가
            if media_outlet not in media_counter:
                media_counter[media_outlet] = 0
            else:
                media_counter[media_outlet] += 1

            # article_id 생성 (언론사별 카운터 사용)
            article_id = f"{''.join(pub_date.split('-'))}_{media_outlet}_{media_counter[media_outlet]:03d}"

            url = item.get("link", "")
            body = None
            # print(f"parser: {url}")
            body = self.get_description(url)
            if body is not None:
                description = body

            article = {
                "article_id": article_id,
                "media_outlet": media_outlet,
                "bias_score": bias_score,
                "title": title,
                "content": description,  # API는 요약문만 제공
                "published_date": pub_date,
                "url": url,
            }

            articles.append(article)

        # 메타데이터 생성
        metadata = {
            "issue": issue,
            "collection_period": f"{articles[0]['published_date']} ~ {articles[-1]['published_date']}"
            if articles
            else "N/A",
            "total_articles": len(articles),
        }

        return {"metadata": metadata, "articles": articles}

    def _remove_html_tags(self, text: str) -> str:
        """HTML 태그 제거"""
        import re

        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

    def _extract_media_outlet(self, url: str) -> str:
        """URL에서 언론사명 추출"""
        # 도메인별 언론사 매핑
        domain_map = {
            "chosun.com": "조선일보",
            "joongang.co.kr": "중앙일보",
            "donga.com": "동아일보",
            "hankyung.com": "한국경제",
            "mk.co.kr": "매일경제",
            "yna.co.kr": "연합뉴스",
            "yonhapnews.co.kr": "연합뉴스",
            "kbs.co.kr": "KBS",
            "imbc.com": "MBC",
            "sbs.co.kr": "SBS",
            "jtbc.co.kr": "JTBC",
            "ytn.co.kr": "YTN",
            "hani.co.kr": "한겨레",
            "khan.co.kr": "경향신문",
            "ohmynews.com": "오마이뉴스",
            "pressian.com": "프레시안",
        }

        for domain, outlet in domain_map.items():
            if domain in url:
                return outlet

        return "기타"

    def _convert_date(self, rfc_date: str) -> str:
        """
        RFC 날짜 형식을 YYYY-MM-DD로 변환
        예: Mon, 26 Sep 2016 07:50:00 +0900 -> 2016-09-26
        """
        try:
            from datetime import datetime

            # RFC 2822 형식 파싱
            dt = datetime.strptime(rfc_date, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except:
            return datetime.now().strftime("%Y-%m-%d")

    def crawl_and_save(
        self,
        query: str,
        output_path: str = "data/input/articles.json",
        max_results: int = 100,
        sort: str = "date",
    ):
        """
        뉴스를 크롤링하고 JSON 파일로 저장

        Args:
            query: 검색어
            output_path: 저장할 파일 경로
            max_results: 최대 결과 개수 (최대 1000개)
            sort: 정렬 방법 (sim 또는 date)
            target_media: 수집할 언론사 리스트 (None이면 '기타' 제외한 전체)
        """
        all_items = []
        display = 100  # 한 번에 최대 100개
        collected_count = 0

        print(f"{'=' * 60}")
        print(f"검색어: {query}")
        print(f"대상 언론사: 전체 (기타 제외)")
        print(f"목표 개수: {max_results}개")
        print(f"{'=' * 60}\n")

        # 필요한 만큼 반복 요청 (목표 개수에 도달할 때까지)
        start = 1
        while collected_count < max_results and start <= 1000:
            print(
                f"API 검색 중... (start: {start}, 수집: {collected_count}/{max_results})"
            )
            result = self.search_news(query, display=display, start=start, sort=sort)

            if result and "items" in result:
                items = result["items"]

                # 언론사 필터링
                # target_media가 없으면 '기타'가 아닌 것만
                filtered_items = []
                for item in items:
                    media = self._extract_media_outlet(item.get("originallink", ""))
                    link = item.get("link", "")
                    if media == "기타" or not link.startswith(
                        "https://n.news.naver.com"
                    ):
                        continue
                    filtered_items.append(item)
                items = filtered_items

                all_items.extend(items)
                collected_count += len(items)

                # 더 이상 결과가 없으면 중단
                if len(result["items"]) < display:
                    break

                # 목표 개수에 도달하면 중단
                if collected_count >= max_results:
                    all_items = all_items[:max_results]
                    break

                start += display
            else:
                break

        if not all_items:
            print("검색 결과가 없습니다.")
            return

        print(f"\n총 {len(all_items)}개의 기사를 수집했습니다.")
        print("본문 크롤링 및 변환 중...\n")

        # articles.json 형식으로 변환
        article_data = self.convert_to_article_format(all_items, issue=query)

        # JSON 파일로 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(
            f"✓ {len(article_data['articles'])}개의 기사를 {output_path}에 저장했습니다."
        )
        print(f"✓ 수집 기간: {article_data['metadata']['collection_period']}")
        print(f"{'=' * 60}")

    def get_description(self, url) -> str | None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = bs(response.text, "html.parser")
        body = soup.select_one("#newsct_article")
        if body:
            # TODO: article body preprocess
            # multiline = re.compile("\n+")
            body = re.sub(r"\n+", "\n", str(body.text.strip()))
            return body
        return None


def main():
    """사용 예제"""
    crawler = NaverNewsCrawler()

    # 뉴스 크롤링 및 저장
    query = "최저시급 인상"  # 검색어
    crawler.crawl_and_save(
        query=query,
        output_path="data/input/articles_naver.json",
        max_results=200,  # 최대 150개 수집
        sort="date",  # 날짜순 정렬
    )


if __name__ == "__main__":
    main()
