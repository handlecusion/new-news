"""
프레임 시각화 모듈
추출된 프레임을 다양한 방식으로 시각화합니다.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Optional, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정"""
    try:
        # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        if Path(font_path).exists():
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc("font", family=font_prop.get_name())
        else:
            # Windows
            plt.rc("font", family="Malgun Gothic")
    except:
        print("⚠️ 한글 폰트 설정 실패. 한글이 깨질 수 있습니다.")

    plt.rc("axes", unicode_minus=False)  # 마이너스 기호 깨짐 방지


class FrameVisualizer:
    """프레임 시각화 클래스"""

    def __init__(self, topic_model=None, set_font: bool = True):
        """
        Args:
            topic_model: BERTopic 모델 인스턴스
            set_font: 한글 폰트 설정 여부
        """
        self.topic_model = topic_model
        if set_font:
            set_korean_font()

    def visualize_topics(self, save_path: Optional[str] = None):
        """
        토픽 간 관계 시각화 (2D 맵)

        Args:
            save_path: 저장 경로
        """
        if not self.topic_model:
            print("⚠️ BERTopic 모델이 필요합니다.")
            return None

        try:
            fig = self.topic_model.visualize_topics()
            if save_path:
                fig.write_html(save_path)
                print(f"✓ 토픽 맵 저장: {save_path}")
            return fig
        except Exception as e:
            print(f"⚠️ 토픽 시각화 실패: {e}")
            return None

    def visualize_barchart(
        self, top_n_topics: int = 8, n_words: int = 10, save_path: Optional[str] = None
    ):
        """
        토픽별 키워드 바차트

        Args:
            top_n_topics: 상위 토픽 수
            n_words: 토픽당 키워드 수
            save_path: 저장 경로
        """
        if not self.topic_model:
            print("⚠️ BERTopic 모델이 필요합니다.")
            return None

        try:
            fig = self.topic_model.visualize_barchart(
                top_n_topics=top_n_topics, n_words=n_words
            )
            if save_path:
                fig.write_html(save_path)
                print(f"✓ 키워드 바차트 저장: {save_path}")
            return fig
        except Exception as e:
            print(f"⚠️ 바차트 시각화 실패: {e}")
            return None

    def visualize_hierarchy(
        self, save_path: Optional[str] = None
    ):
        """
        토픽 계층 구조 시각화

        Args:
            save_path: 저장 경로
        """
        if not self.topic_model:
            print("⚠️ BERTopic 모델이 필요합니다.")
            return None

        try:
            hierarchical_topics = self.topic_model.hierarchical_topics(
                self.topic_model.original_topics_
            )
            fig = self.topic_model.visualize_hierarchy(
                hierarchical_topics=hierarchical_topics
            )
            if save_path:
                fig.write_html(save_path)
                print(f"✓ 계층 구조 저장: {save_path}")
            return fig
        except Exception as e:
            print(f"⚠️ 계층 구조 시각화 실패: {e}")
            return None

    def create_frame_distribution(
        self,
        articles: List[Dict],
        topics: np.ndarray,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 8),
    ):
        """
        언론사별 프레임 분포 히트맵

        Args:
            articles: 기사 리스트
            topics: 토픽 할당
            save_path: 저장 경로
            figsize: 그림 크기
        """
        # 데이터프레임 생성
        df = pd.DataFrame({
            "media": [a["media_outlet"] for a in articles],
            "bias": [a["bias_score"] for a in articles],
            "frame": topics,
        })

        # outlier 제외
        df = df[df["frame"] >= 0]

        # 피벗 테이블 생성
        pivot = df.pivot_table(
            index="media", columns="frame", aggfunc="size", fill_value=0
        )

        # 언론사를 편향도 순으로 정렬
        media_bias = df.groupby("media")["bias"].mean().sort_values()
        pivot = pivot.reindex(media_bias.index)

        # 히트맵 그리기
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "기사 수"},
            linewidths=0.5,
        )

        plt.title("언론사별 프레임 분포", fontsize=16, fontweight="bold")
        plt.xlabel("프레임 ID", fontsize=12)
        plt.ylabel("언론사", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 프레임 분포 히트맵 저장: {save_path}")
            plt.close()
        else:
            plt.show()

        return pivot

    def visualize_frame_bias_correlation(
        self,
        articles: List[Dict],
        topics: np.ndarray,
        frame_names: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (16, 10),
    ):
        """
        프레임과 편향도의 관계 시각화

        Args:
            articles: 기사 리스트
            topics: 토픽 할당
            frame_names: 프레임 이름 매핑
            save_path: 저장 경로
            figsize: 그림 크기
        """
        # 데이터프레임 생성
        df = pd.DataFrame({
            "media": [a["media_outlet"] for a in articles],
            "bias": [a["bias_score"] for a in articles],
            "frame": topics,
            "title": [a["title"] for a in articles],
        })

        # outlier 제외
        df = df[df["frame"] >= 0]

        # 프레임 이름 적용
        if frame_names:
            df["frame_name"] = df["frame"].map(frame_names)
        else:
            df["frame_name"] = df["frame"].apply(lambda x: f"프레임 {x}")

        # 그림 생성
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 프레임별 편향도 박스플롯
        frame_order = df.groupby("frame")["bias"].median().sort_values().index
        df_ordered = df.set_index("frame").loc[frame_order].reset_index()

        sns.boxplot(data=df_ordered, x="frame", y="bias", ax=axes[0, 0])
        axes[0, 0].set_title("프레임별 편향도 분포", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("프레임 ID")
        axes[0, 0].set_ylabel("편향도")
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].axhline(y=-0.3, color="blue", linestyle=":", alpha=0.3)
        axes[0, 0].axhline(y=0.3, color="blue", linestyle=":", alpha=0.3)

        # 2. 프레임별 기사 수
        frame_counts = df["frame"].value_counts().sort_index()
        axes[0, 1].bar(frame_counts.index, frame_counts.values, color="skyblue")
        axes[0, 1].set_title("프레임별 기사 수", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("프레임 ID")
        axes[0, 1].set_ylabel("기사 수")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 편향 그룹별 프레임 분포
        df["bias_group"] = pd.cut(
            df["bias"],
            bins=[-1, -0.3, 0.3, 1],
            labels=["진보", "중도", "보수"],
        )

        frame_bias_pivot = pd.crosstab(
            df["frame"], df["bias_group"], normalize="columns"
        )

        frame_bias_pivot.T.plot(kind="bar", stacked=True, ax=axes[1, 0], legend=False)
        axes[1, 0].set_title("편향 그룹별 프레임 분포", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("편향 그룹")
        axes[1, 0].set_ylabel("비율")
        axes[1, 0].legend(
            title="프레임", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8
        )
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

        # 4. 프레임-편향도 산점도
        for frame_id in df["frame"].unique():
            frame_data = df[df["frame"] == frame_id]
            axes[1, 1].scatter(
                frame_data["bias"],
                [frame_id] * len(frame_data),
                alpha=0.6,
                s=50,
                label=f"프레임 {frame_id}",
            )

        axes[1, 1].set_title("프레임-편향도 산점도", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("편향도")
        axes[1, 1].set_ylabel("프레임 ID")
        axes[1, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].axvline(x=-0.3, color="blue", linestyle=":", alpha=0.3)
        axes[1, 1].axvline(x=0.3, color="blue", linestyle=":", alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("프레임-편향도 관계 분석", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 프레임-편향도 관계 시각화 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def visualize_frame_keywords(
        self,
        frames: List[Dict],
        top_n: int = 5,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 8),
    ):
        """
        프레임별 주요 키워드 시각화

        Args:
            frames: 프레임 정보 리스트
            top_n: 상위 키워드 수
            save_path: 저장 경로
            figsize: 그림 크기
        """
        n_frames = len(frames)
        if n_frames == 0:
            print("⚠️ 시각화할 프레임이 없습니다.")
            return

        # 서브플롯 설정
        n_cols = min(3, n_frames)
        n_rows = (n_frames - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, frame in enumerate(frames):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]

            # 상위 키워드와 스코어
            keywords = frame["keywords"][:top_n]
            scores = frame["keyword_scores"][:top_n]

            # 바차트 그리기
            ax.barh(range(len(keywords)), scores, color="steelblue")
            ax.set_yticks(range(len(keywords)))
            ax.set_yticklabels(keywords)
            ax.set_xlabel("중요도")
            ax.set_title(
                f"프레임 {frame['frame_id']}: {frame.get('suggested_name', '')}",
                fontsize=10,
            )
            ax.invert_yaxis()

        # 빈 서브플롯 숨기기
        for idx in range(n_frames, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")

        plt.suptitle("프레임별 주요 키워드", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 프레임 키워드 시각화 저장: {save_path}")
            plt.close()
        else:
            plt.show()


def visualize_results(
    json_path: str = "data/input/articles.json",
    frames_path: str = "results/frames.json",
    article_frames_path: str = "results/article_frames.json",
    output_dir: str = "results/figures",
):
    """
    저장된 결과를 시각화 (테스트/실행용)

    Args:
        json_path: 원본 데이터 경로
        frames_path: 프레임 정보 경로
        article_frames_path: 기사별 프레임 경로
        output_dir: 출력 디렉토리
    """
    # 데이터 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"]

    with open(frames_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    with open(article_frames_path, "r", encoding="utf-8") as f:
        article_frames = json.load(f)

    # 토픽 배열 생성
    topics = np.array([af["assigned_frame"] for af in article_frames])

    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 시각화 생성
    visualizer = FrameVisualizer()

    # 1. 언론사별 프레임 분포
    visualizer.create_frame_distribution(
        articles, topics, save_path=output_dir / "media_frame_heatmap.png"
    )

    # 2. 프레임-편향도 관계
    frame_names = {f["frame_id"]: f.get("suggested_name", f"프레임_{f['frame_id']}") for f in frames}
    visualizer.visualize_frame_bias_correlation(
        articles, topics, frame_names, save_path=output_dir / "frame_bias_analysis.png"
    )

    # 3. 프레임별 키워드
    visualizer.visualize_frame_keywords(
        frames, top_n=8, save_path=output_dir / "frame_keywords.png"
    )

    print("\n✓ 모든 시각화 완료")


if __name__ == "__main__":
    # 결과 파일이 있으면 시각화
    if Path("results/frames.json").exists():
        visualize_results()
    else:
        print("⚠️ 먼저 frame_extractor.py를 실행하여 프레임을 추출하세요.")