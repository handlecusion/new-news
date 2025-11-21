"""
인터랙티브 대시보드 모듈
Plotly를 사용하여 프레임 분석 결과를 인터랙티브하게 시각화합니다.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from pathlib import Path


class InteractiveDashboard:
    """인터랙티브 대시보드 클래스"""

    def __init__(
        self,
        articles: List[Dict],
        frames: List[Dict],
        frame_assignments: np.ndarray,
        frame_probs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            articles: 기사 리스트
            frames: 프레임 정보
            frame_assignments: 프레임 할당
            frame_probs: 프레임 확률
        """
        self.articles = articles
        self.frames = frames
        self.frame_assignments = frame_assignments
        self.frame_probs = frame_probs

        # 데이터프레임 생성
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """대시보드용 데이터프레임 생성"""
        df_data = {
            "article_id": [],
            "media": [],
            "bias_score": [],
            "bias_label": [],
            "frame": [],
            "title": [],
            "content_preview": [],
        }

        for i, article in enumerate(self.articles):
            df_data["article_id"].append(article.get("article_id", f"article_{i}"))
            df_data["media"].append(article["media_outlet"])
            df_data["bias_score"].append(article["bias_score"])

            # 편향 레이블
            if article["bias_score"] < -0.3:
                bias_label = "진보"
            elif article["bias_score"] > 0.3:
                bias_label = "보수"
            else:
                bias_label = "중도"
            df_data["bias_label"].append(bias_label)

            df_data["frame"].append(self.frame_assignments[i])
            df_data["title"].append(article["title"])
            df_data["content_preview"].append(article.get("content", "")[:200] + "...")

        df = pd.DataFrame(df_data)

        # 프레임 이름 추가
        frame_names = {
            f["frame_id"]: f.get("suggested_name", f"프레임 {f['frame_id']}")
            for f in self.frames
        }
        df["frame_name"] = df["frame"].map(frame_names).fillna("Outlier")

        return df

    def create_main_dashboard(self, save_path: Optional[str] = None):
        """
        메인 대시보드 생성

        Args:
            save_path: HTML 파일 저장 경로
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "프레임 분포",
                "편향도 분포",
                "프레임-편향도 관계",
                "언론사별 프레임 사용",
                "프레임별 편향도 분포",
                "언론사 편향도",
                "프레임 크기",
                "편향 그룹별 프레임",
                "시간별 프레임 추이"
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        # outlier 제외
        valid_df = self.df[self.df["frame"] >= 0]

        # 1. 프레임 분포
        frame_counts = valid_df["frame_name"].value_counts()
        fig.add_trace(
            go.Bar(
                x=frame_counts.index,
                y=frame_counts.values,
                name="프레임 빈도",
                marker_color="lightblue",
                hovertemplate="프레임: %{x}<br>기사 수: %{y}<extra></extra>",
            ),
            row=1, col=1
        )

        # 2. 편향도 분포
        fig.add_trace(
            go.Histogram(
                x=self.df["bias_score"],
                nbinsx=20,
                name="편향도",
                marker_color="lightgreen",
                hovertemplate="편향도: %{x}<br>빈도: %{y}<extra></extra>",
            ),
            row=1, col=2
        )

        # 3. 프레임-편향도 산점도
        for frame_id in valid_df["frame"].unique():
            frame_data = valid_df[valid_df["frame"] == frame_id]
            frame_name = frame_data["frame_name"].iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=frame_data["bias_score"],
                    y=[frame_id] * len(frame_data),
                    mode="markers",
                    name=frame_name,
                    text=frame_data["title"],
                    hovertemplate="<b>%{text}</b><br>편향도: %{x:.2f}<extra></extra>",
                    marker=dict(size=8, opacity=0.6),
                ),
                row=1, col=3
            )

        # 4. 언론사별 프레임 사용
        media_frame = valid_df.groupby(["media", "frame_name"]).size().unstack(fill_value=0)
        for frame_name in media_frame.columns:
            fig.add_trace(
                go.Bar(
                    x=media_frame.index,
                    y=media_frame[frame_name],
                    name=frame_name,
                    hovertemplate="언론사: %{x}<br>기사 수: %{y}<extra></extra>",
                ),
                row=2, col=1
            )

        # 5. 프레임별 편향도 박스플롯
        for frame_id in sorted(valid_df["frame"].unique()):
            frame_data = valid_df[valid_df["frame"] == frame_id]
            frame_name = frame_data["frame_name"].iloc[0]

            fig.add_trace(
                go.Box(
                    y=frame_data["bias_score"],
                    name=frame_name,
                    boxmean=True,
                    hovertemplate="편향도: %{y:.2f}<extra></extra>",
                ),
                row=2, col=2
            )

        # 6. 언론사 편향도
        media_bias = self.df.groupby("media")["bias_score"].mean().sort_values()
        fig.add_trace(
            go.Scatter(
                x=media_bias.values,
                y=media_bias.index,
                mode="markers+text",
                text=media_bias.index,
                textposition="middle left",
                marker=dict(
                    size=10,
                    color=media_bias.values,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="편향도", x=0.65, len=0.25),
                ),
                hovertemplate="언론사: %{y}<br>평균 편향도: %{x:.2f}<extra></extra>",
            ),
            row=2, col=3
        )

        # 7. 프레임 크기 (파이 차트)
        frame_sizes = valid_df["frame_name"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=frame_sizes.index,
                values=frame_sizes.values,
                hovertemplate="프레임: %{label}<br>기사 수: %{value}<br>비율: %{percent}<extra></extra>",
            ),
            row=3, col=1
        )

        # 8. 편향 그룹별 프레임 분포
        bias_frame = valid_df.groupby(["bias_label", "frame_name"]).size().unstack(fill_value=0)
        for bias_label in bias_frame.index:
            fig.add_trace(
                go.Bar(
                    x=bias_frame.columns,
                    y=bias_frame.loc[bias_label],
                    name=bias_label,
                    hovertemplate="프레임: %{x}<br>기사 수: %{y}<extra></extra>",
                ),
                row=3, col=2
            )

        # 9. 날짜별 프레임 (더미 데이터)
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        for frame_id in valid_df["frame"].unique()[:3]:  # 상위 3개 프레임만
            frame_name = valid_df[valid_df["frame"] == frame_id]["frame_name"].iloc[0]
            daily_counts = np.random.poisson(5, 30)  # 더미 데이터

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=daily_counts,
                    mode="lines+markers",
                    name=frame_name,
                    hovertemplate="날짜: %{x}<br>기사 수: %{y}<extra></extra>",
                ),
                row=3, col=3
            )

        # 레이아웃 업데이트
        fig.update_layout(
            title={
                "text": "뉴스 프레임 분석 대시보드",
                "font": {"size": 20},
                "x": 0.5,
                "xanchor": "center",
            },
            showlegend=True,
            height=1000,
            hovermode="closest",
        )

        # x, y축 라벨
        fig.update_xaxes(title_text="프레임", row=1, col=1)
        fig.update_xaxes(title_text="편향도", row=1, col=2)
        fig.update_xaxes(title_text="편향도", row=1, col=3)
        fig.update_xaxes(title_text="언론사", row=2, col=1)
        fig.update_xaxes(title_text="프레임", row=2, col=2)
        fig.update_xaxes(title_text="편향도", row=2, col=3)
        fig.update_xaxes(title_text="프레임", row=3, col=2)
        fig.update_xaxes(title_text="날짜", row=3, col=3)

        fig.update_yaxes(title_text="기사 수", row=1, col=1)
        fig.update_yaxes(title_text="빈도", row=1, col=2)
        fig.update_yaxes(title_text="프레임 ID", row=1, col=3)
        fig.update_yaxes(title_text="기사 수", row=2, col=1)
        fig.update_yaxes(title_text="편향도", row=2, col=2)
        fig.update_yaxes(title_text="언론사", row=2, col=3)
        fig.update_yaxes(title_text="기사 수", row=3, col=2)
        fig.update_yaxes(title_text="기사 수", row=3, col=3)

        # 저장
        if save_path:
            fig.write_html(save_path)
            print(f"✓ 대시보드 저장: {save_path}")

        return fig

    def create_frame_explorer(self, save_path: Optional[str] = None):
        """
        프레임별 기사 탐색기 생성

        Args:
            save_path: HTML 파일 저장 경로
        """
        valid_df = self.df[self.df["frame"] >= 0]

        # 프레임 정보 추가
        frame_info = {f["frame_id"]: f for f in self.frames}

        # 테이블 데이터 준비
        table_data = []
        for _, row in valid_df.iterrows():
            frame_id = row["frame"]
            if frame_id in frame_info:
                keywords = ", ".join(frame_info[frame_id].get("keywords", [])[:5])
            else:
                keywords = ""

            table_data.append([
                row["frame_name"],
                keywords,
                row["media"],
                f"{row['bias_score']:.2f}",
                row["bias_label"],
                row["title"],
            ])

        # 인터랙티브 테이블 생성
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["프레임", "주요 키워드", "언론사", "편향도", "편향 그룹", "제목"],
                fill_color="paleturquoise",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=list(zip(*table_data)),  # 전치
                fill_color="lavender",
                align="left",
                font=dict(size=11),
                height=30,
            )
        )])

        fig.update_layout(
            title={
                "text": "프레임별 기사 탐색기",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            height=800,
        )

        # 저장
        if save_path:
            fig.write_html(save_path)
            print(f"✓ 프레임 탐색기 저장: {save_path}")

        return fig

    def create_frame_network(self, save_path: Optional[str] = None):
        """
        프레임 네트워크 그래프 생성 (프레임 간 관계)

        Args:
            save_path: HTML 파일 저장 경로
        """
        valid_df = self.df[self.df["frame"] >= 0]

        # 언론사별 프레임 사용 매트릭스
        media_frame = valid_df.pivot_table(
            index="media",
            columns="frame",
            aggfunc="size",
            fill_value=0
        )

        # 프레임 간 상관관계 계산 (언론사 사용 패턴 기반)
        frame_corr = media_frame.corr()

        # 네트워크 노드와 엣지 생성
        nodes = []
        edges = []

        # 노드 생성
        for frame_id in frame_corr.index:
            frame_info = next((f for f in self.frames if f["frame_id"] == frame_id), {})
            node_size = valid_df[valid_df["frame"] == frame_id].shape[0]

            nodes.append({
                "id": frame_id,
                "label": frame_info.get("suggested_name", f"프레임 {frame_id}"),
                "size": node_size,
                "keywords": ", ".join(frame_info.get("keywords", [])[:3]),
            })

        # 엣지 생성 (상관계수 > 0.3인 경우만)
        for i, frame1 in enumerate(frame_corr.index):
            for j, frame2 in enumerate(frame_corr.columns):
                if i < j and abs(frame_corr.iloc[i, j]) > 0.3:
                    edges.append({
                        "source": frame1,
                        "target": frame2,
                        "weight": abs(frame_corr.iloc[i, j]),
                    })

        # Plotly 네트워크 그래프
        edge_trace = []
        for edge in edges:
            source_node = next(n for n in nodes if n["id"] == edge["source"])
            target_node = next(n for n in nodes if n["id"] == edge["target"])

            # 간단한 레이아웃 (원형 배치)
            angle_source = 2 * np.pi * edge["source"] / len(nodes)
            angle_target = 2 * np.pi * edge["target"] / len(nodes)

            x0, y0 = np.cos(angle_source), np.sin(angle_source)
            x1, y1 = np.cos(angle_target), np.sin(angle_target)

            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=edge["weight"] * 2, color="gray"),
                    hoverinfo="none",
                )
            )

        node_trace = go.Scatter(
            x=[np.cos(2 * np.pi * n["id"] / len(nodes)) for n in nodes],
            y=[np.sin(2 * np.pi * n["id"] / len(nodes)) for n in nodes],
            mode="markers+text",
            text=[n["label"] for n in nodes],
            textposition="top center",
            marker=dict(
                size=[n["size"] for n in nodes],
                color=[n["id"] for n in nodes],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="프레임 ID"),
            ),
            hovertemplate="<b>%{text}</b><br>크기: %{marker.size}<extra></extra>",
        )

        fig = go.Figure(data=edge_trace + [node_trace])

        fig.update_layout(
            title={
                "text": "프레임 네트워크 (언론사 사용 패턴 기반)",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )

        # 저장
        if save_path:
            fig.write_html(save_path)
            print(f"✓ 프레임 네트워크 저장: {save_path}")

        return fig

    def create_bias_timeline(self, save_path: Optional[str] = None):
        """
        편향도 타임라인 생성

        Args:
            save_path: HTML 파일 저장 경로
        """
        # 날짜 정보가 있는 경우에만 실행 (현재는 더미 데이터)
        dates = pd.date_range(start="2024-01-01", periods=len(self.df), freq="6H")
        self.df["date"] = dates[:len(self.df)]

        # 일별 편향도 평균
        daily_bias = self.df.groupby(self.df["date"].dt.date).agg({
            "bias_score": ["mean", "std", "count"]
        })

        fig = go.Figure()

        # 평균 편향도 라인
        fig.add_trace(
            go.Scatter(
                x=daily_bias.index,
                y=daily_bias[("bias_score", "mean")],
                mode="lines+markers",
                name="평균 편향도",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                hovertemplate="날짜: %{x}<br>평균 편향도: %{y:.3f}<extra></extra>",
            )
        )

        # 표준편차 영역
        fig.add_trace(
            go.Scatter(
                x=daily_bias.index,
                y=daily_bias[("bias_score", "mean")] + daily_bias[("bias_score", "std")],
                mode="lines",
                name="표준편차",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_bias.index,
                y=daily_bias[("bias_score", "mean")] - daily_bias[("bias_score", "std")],
                mode="lines",
                name="표준편차",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 100, 200, 0.2)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 중립선
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=-0.3, line_dash="dot", line_color="gray", opacity=0.3)
        fig.add_hline(y=0.3, line_dash="dot", line_color="gray", opacity=0.3)

        fig.update_layout(
            title={
                "text": "편향도 타임라인",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="날짜",
            yaxis_title="편향도",
            height=400,
            hovermode="x unified",
        )

        # 저장
        if save_path:
            fig.write_html(save_path)
            print(f"✓ 편향도 타임라인 저장: {save_path}")

        return fig


def create_full_dashboard(
    articles_path: str = "data/input/articles.json",
    frames_path: str = "results/frames.json",
    article_frames_path: str = "results/article_frames.json",
    output_dir: str = "results",
):
    """
    전체 대시보드 생성 (테스트/실행용)

    Args:
        articles_path: 기사 데이터 경로
        frames_path: 프레임 정보 경로
        article_frames_path: 기사별 프레임 경로
        output_dir: 출력 디렉토리
    """
    # 데이터 로드
    with open(articles_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"]

    with open(frames_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    with open(article_frames_path, "r", encoding="utf-8") as f:
        article_frames = json.load(f)

    # 프레임 할당 배열 생성
    frame_assignments = np.array([af["assigned_frame"] for af in article_frames])

    # 프레임 확률 (있다면)
    frame_probs = None
    if article_frames[0].get("frame_probabilities"):
        frame_probs = np.array([af["frame_probabilities"] for af in article_frames])

    # 대시보드 생성
    dashboard = InteractiveDashboard(articles, frames, frame_assignments, frame_probs)

    # 출력 디렉토리
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 메인 대시보드
    dashboard.create_main_dashboard(save_path=output_dir / "dashboard.html")

    # 프레임 탐색기
    dashboard.create_frame_explorer(save_path=output_dir / "frame_explorer.html")

    # 프레임 네트워크
    dashboard.create_frame_network(save_path=output_dir / "frame_network.html")

    # 편향도 타임라인
    dashboard.create_bias_timeline(save_path=output_dir / "bias_timeline.html")

    print("\n✓ 모든 대시보드 생성 완료")
    print("  - dashboard.html: 메인 대시보드")
    print("  - frame_explorer.html: 프레임별 기사 탐색")
    print("  - frame_network.html: 프레임 관계 네트워크")
    print("  - bias_timeline.html: 편향도 타임라인")

    return dashboard


if __name__ == "__main__":
    # 필요한 파일이 있으면 대시보드 생성
    if (
        Path("data/input/articles.json").exists() and
        Path("results/frames.json").exists() and
        Path("results/article_frames.json").exists()
    ):
        create_full_dashboard()
    else:
        print("⚠️ 필요한 파일을 찾을 수 없습니다.")
        print("  1. generate_sample_data.py 실행")
        print("  2. frame_extractor.py 실행")
        print("  이후 이 스크립트를 실행하세요.")