"""
프레임-편향도 상관관계 분석 모듈
발견된 프레임과 편향도 간의 통계적 관계를 분석합니다.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway
from typing import List, Dict, Optional, Tuple, Any
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
            import matplotlib.font_manager as fm
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc("font", family=font_prop.get_name())
        else:
            # Windows
            plt.rc("font", family="Malgun Gothic")
    except:
        pass
    plt.rc("axes", unicode_minus=False)


class IntegratedAnalyzer:
    """프레임-편향도 통합 분석 클래스"""

    def __init__(
        self,
        articles: List[Dict],
        frame_assignments: np.ndarray,
        frame_probs: Optional[np.ndarray] = None,
        frame_info: Optional[List[Dict]] = None,
    ):
        """
        Args:
            articles: 기사 리스트
            frame_assignments: 프레임 할당
            frame_probs: 프레임 확률 분포
            frame_info: 프레임 정보 (키워드 등)
        """
        self.articles = articles
        self.frame_assignments = frame_assignments
        self.frame_probs = frame_probs
        self.frame_info = frame_info

        # 데이터프레임 생성
        self.df = self._create_dataframe()

        # 한글 폰트 설정
        set_korean_font()

    def _create_dataframe(self) -> pd.DataFrame:
        """분석용 데이터프레임 생성"""
        df_data = {
            "article_id": [],
            "media": [],
            "bias_score": [],
            "bias_label": [],
            "frame": [],
            "title": [],
        }

        for i, article in enumerate(self.articles):
            df_data["article_id"].append(article.get("article_id", f"article_{i}"))
            df_data["media"].append(article["media_outlet"])
            df_data["bias_score"].append(article["bias_score"])
            df_data["bias_label"].append(self._bias_to_label(article["bias_score"]))
            df_data["frame"].append(self.frame_assignments[i])
            df_data["title"].append(article["title"])

        df = pd.DataFrame(df_data)

        # 프레임 확률 추가
        if self.frame_probs is not None:
            for i in range(self.frame_probs.shape[1]):
                df[f"frame_{i}_prob"] = self.frame_probs[:, i]

        return df

    @staticmethod
    def _bias_to_label(score: float) -> str:
        """편향도 점수를 레이블로 변환"""
        if score < -0.3:
            return "진보"
        elif score > 0.3:
            return "보수"
        else:
            return "중도"

    def analyze_frame_bias_correlation(self, verbose: bool = True) -> pd.DataFrame:
        """
        프레임과 편향도 간 상관관계 분석

        Args:
            verbose: 출력 여부

        Returns:
            상관관계 결과 데이터프레임
        """
        results = []

        # 전체 상관관계 (프레임 ID vs 편향도)
        valid_df = self.df[self.df["frame"] >= 0]  # outlier 제외

        if len(valid_df) > 0:
            corr_coef, p_value = spearmanr(valid_df["frame"], valid_df["bias_score"])
            results.append({
                "analysis": "전체 프레임-편향도 상관관계",
                "statistic": corr_coef,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })

        # 각 프레임별 편향도 평균
        frame_bias_means = valid_df.groupby("frame")["bias_score"].agg(["mean", "std", "count"])

        if verbose:
            print("\n=== 프레임별 편향도 통계 ===")
            print(frame_bias_means.to_string())

        # 프레임 확률별 상관관계
        if self.frame_probs is not None:
            frame_prob_cols = [col for col in self.df.columns if col.startswith("frame_") and col.endswith("_prob")]

            for col in frame_prob_cols:
                frame_id = col.split("_")[1]
                corr_coef, p_value = pearsonr(self.df["bias_score"], self.df[col])

                results.append({
                    "analysis": f"프레임 {frame_id} 확률-편향도 상관관계",
                    "statistic": corr_coef,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                })

        results_df = pd.DataFrame(results)

        if verbose and len(results_df) > 0:
            print("\n=== 상관관계 분석 결과 ===")
            print(results_df.to_string(index=False))

        return results_df

    def chi_square_test(self, verbose: bool = True) -> Dict:
        """
        카이제곱 검정: 프레임과 편향 그룹 간 독립성 검정

        Args:
            verbose: 출력 여부

        Returns:
            검정 결과 딕셔너리
        """
        valid_df = self.df[self.df["frame"] >= 0]

        # 교차표 생성
        contingency_table = pd.crosstab(valid_df["bias_label"], valid_df["frame"])

        # 카이제곱 검정
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        result = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < 0.05,
            "contingency_table": contingency_table,
        }

        if verbose:
            print("\n=== 카이제곱 독립성 검정 ===")
            print(f"카이제곱 통계량: {chi2:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"자유도: {dof}")
            print(f"귀무가설 기각: {result['significant']} (α=0.05)")
            print("\n교차표 (관측값):")
            print(contingency_table)
            print("\n교차표 (비율):")
            print(pd.crosstab(valid_df["bias_label"], valid_df["frame"], normalize="columns"))

        return result

    def anova_test(self, verbose: bool = True) -> Dict:
        """
        ANOVA 검정: 프레임별 편향도 평균 차이 검정

        Args:
            verbose: 출력 여부

        Returns:
            검정 결과 딕셔너리
        """
        valid_df = self.df[self.df["frame"] >= 0]

        # 프레임별 편향도 그룹 생성
        frame_groups = [group["bias_score"].values for _, group in valid_df.groupby("frame")]

        if len(frame_groups) < 2:
            print("⚠️ ANOVA를 위한 충분한 그룹이 없습니다.")
            return {}

        # F-검정
        f_stat, p_value = f_oneway(*frame_groups)

        result = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_groups": len(frame_groups),
        }

        if verbose:
            print("\n=== ANOVA 검정 (프레임별 편향도 평균 차이) ===")
            print(f"F 통계량: {f_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"그룹 수: {result['n_groups']}")
            print(f"귀무가설 기각: {result['significant']} (α=0.05)")

        return result

    def analyze_media_frame_preference(self, verbose: bool = True) -> pd.DataFrame:
        """
        언론사별 프레임 선호도 분석

        Args:
            verbose: 출력 여부

        Returns:
            언론사별 프레임 사용 비율 데이터프레임
        """
        valid_df = self.df[self.df["frame"] >= 0]

        # 언론사별 프레임 사용 비율
        media_frame = pd.crosstab(
            valid_df["media"],
            valid_df["frame"],
            normalize="index"
        )

        # 언론사 편향도 순으로 정렬
        media_bias = self.df.groupby("media")["bias_score"].mean().sort_values()
        media_frame = media_frame.reindex(media_bias.index)

        if verbose:
            print("\n=== 언론사별 프레임 사용 비율 ===")
            print(media_frame.round(3).to_string())

        return media_frame

    def find_discriminative_frames(self, threshold: float = 0.3) -> List[Dict]:
        """
        편향을 구분하는 특징적인 프레임 찾기

        Args:
            threshold: 편향도 차이 임계값

        Returns:
            특징적인 프레임 정보 리스트
        """
        valid_df = self.df[self.df["frame"] >= 0]
        discriminative_frames = []

        for frame_id in valid_df["frame"].unique():
            frame_data = valid_df[valid_df["frame"] == frame_id]

            # 편향 그룹별 비율
            bias_dist = frame_data["bias_label"].value_counts(normalize=True)

            # 평균 편향도
            mean_bias = frame_data["bias_score"].mean()
            std_bias = frame_data["bias_score"].std()

            # 특징 판단
            if abs(mean_bias) > threshold:
                frame_type = "진보 프레임" if mean_bias < 0 else "보수 프레임"

                frame_info_dict = {
                    "frame_id": int(frame_id),
                    "frame_type": frame_type,
                    "mean_bias": mean_bias,
                    "std_bias": std_bias,
                    "n_articles": len(frame_data),
                    "bias_distribution": bias_dist.to_dict(),
                }

                # 프레임 키워드 추가
                if self.frame_info:
                    frame_meta = next((f for f in self.frame_info if f["frame_id"] == frame_id), None)
                    if frame_meta:
                        frame_info_dict["keywords"] = frame_meta.get("keywords", [])[:5]
                        frame_info_dict["suggested_name"] = frame_meta.get("suggested_name", "")

                discriminative_frames.append(frame_info_dict)

        # 편향도 절댓값으로 정렬
        discriminative_frames.sort(key=lambda x: abs(x["mean_bias"]), reverse=True)

        print(f"\n=== 특징적인 프레임 (|편향도| > {threshold}) ===")
        for frame in discriminative_frames:
            print(f"\n프레임 {frame['frame_id']}: {frame['frame_type']}")
            print(f"  평균 편향도: {frame['mean_bias']:.3f} (±{frame['std_bias']:.3f})")
            print(f"  기사 수: {frame['n_articles']}")
            if "keywords" in frame:
                print(f"  주요 키워드: {', '.join(frame['keywords'])}")

        return discriminative_frames

    def create_comprehensive_report(self, save_path: Optional[str] = None) -> Dict:
        """
        종합 분석 리포트 생성

        Args:
            save_path: 저장 경로

        Returns:
            분석 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("프레임-편향도 통합 분석 리포트")
        print("=" * 60)

        report = {
            "data_summary": {
                "n_articles": len(self.df),
                "n_frames": len(self.df[self.df["frame"] >= 0]["frame"].unique()),
                "n_outliers": len(self.df[self.df["frame"] == -1]),
                "bias_distribution": self.df["bias_label"].value_counts().to_dict(),
            },
            "correlation_analysis": self.analyze_frame_bias_correlation(verbose=False),
            "chi_square_test": self.chi_square_test(verbose=False),
            "anova_test": self.anova_test(verbose=False),
            "discriminative_frames": self.find_discriminative_frames(),
            "media_frame_preference": self.analyze_media_frame_preference(verbose=False),
        }

        # 주요 발견사항
        findings = []

        # 상관관계 유의성
        if len(report["correlation_analysis"]) > 0:
            significant_corr = report["correlation_analysis"][
                report["correlation_analysis"]["significant"]
            ]
            if len(significant_corr) > 0:
                findings.append(f"✓ {len(significant_corr)}개의 유의미한 프레임-편향도 상관관계 발견")

        # 카이제곱 검정
        if report["chi_square_test"].get("significant"):
            findings.append("✓ 프레임과 편향 그룹 간 유의미한 연관성 확인 (카이제곱 검정)")

        # ANOVA 검정
        if report["anova_test"].get("significant"):
            findings.append("✓ 프레임별 편향도 평균에 유의미한 차이 존재 (ANOVA)")

        # 특징적 프레임
        if report["discriminative_frames"]:
            n_progressive = sum(1 for f in report["discriminative_frames"] if f["frame_type"] == "진보 프레임")
            n_conservative = sum(1 for f in report["discriminative_frames"] if f["frame_type"] == "보수 프레임")
            findings.append(f"✓ 진보 성향 프레임 {n_progressive}개, 보수 성향 프레임 {n_conservative}개 발견")

        report["key_findings"] = findings

        print("\n=== 주요 발견사항 ===")
        for finding in findings:
            print(finding)

        # 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                # numpy/pandas 객체를 JSON 직렬화 가능한 형태로 변환
                json_report = self._prepare_for_json(report)
                json.dump(json_report, f, ensure_ascii=False, indent=2)

            print(f"\n✓ 분석 리포트 저장: {save_path}")

        return report

    def _prepare_for_json(self, obj: Any) -> Any:
        """JSON 직렬화를 위한 데이터 변환"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj

    def visualize_comprehensive_analysis(
        self,
        save_dir: Optional[str] = None,
        figsize: tuple = (18, 12)
    ):
        """
        종합 분석 시각화

        Args:
            save_dir: 저장 디렉토리
            figsize: 그림 크기
        """
        valid_df = self.df[self.df["frame"] >= 0]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 프레임별 편향도 분포 (박스플롯)
        ax1 = fig.add_subplot(gs[0, :2])
        frame_order = valid_df.groupby("frame")["bias_score"].median().sort_values().index
        valid_df_ordered = valid_df.set_index("frame").loc[frame_order].reset_index()
        sns.boxplot(data=valid_df_ordered, x="frame", y="bias_score", ax=ax1)
        ax1.set_title("프레임별 편향도 분포", fontsize=14, fontweight="bold")
        ax1.set_xlabel("프레임 ID")
        ax1.set_ylabel("편향도")
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(y=-0.3, color="blue", linestyle=":", alpha=0.3)
        ax1.axhline(y=0.3, color="blue", linestyle=":", alpha=0.3)

        # 2. 편향 그룹별 프레임 분포
        ax2 = fig.add_subplot(gs[0, 2])
        bias_frame = pd.crosstab(valid_df["bias_label"], valid_df["frame"], normalize="index")
        bias_frame.T.plot(kind="bar", stacked=True, ax=ax2, legend=False)
        ax2.set_title("편향 그룹별 프레임 비율", fontsize=12, fontweight="bold")
        ax2.set_xlabel("프레임 ID")
        ax2.set_ylabel("비율")
        ax2.legend(title="편향", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # 3. 언론사-프레임 히트맵
        ax3 = fig.add_subplot(gs[1, :])
        media_frame = pd.crosstab(valid_df["media"], valid_df["frame"])
        media_bias = self.df.groupby("media")["bias_score"].mean().sort_values()
        media_frame = media_frame.reindex(media_bias.index)

        sns.heatmap(
            media_frame,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "기사 수"},
            ax=ax3,
            linewidths=0.5
        )
        ax3.set_title("언론사별 프레임 사용 빈도", fontsize=14, fontweight="bold")
        ax3.set_xlabel("프레임 ID")
        ax3.set_ylabel("언론사")

        # 4. 프레임-편향도 산점도
        ax4 = fig.add_subplot(gs[2, 0])
        for frame_id in valid_df["frame"].unique():
            frame_data = valid_df[valid_df["frame"] == frame_id]
            ax4.scatter(
                frame_data["bias_score"],
                [frame_id] * len(frame_data),
                alpha=0.6,
                s=30
            )
        ax4.set_title("프레임-편향도 산점도", fontsize=12, fontweight="bold")
        ax4.set_xlabel("편향도")
        ax4.set_ylabel("프레임 ID")
        ax4.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax4.grid(True, alpha=0.3)

        # 5. 프레임 크기 분포
        ax5 = fig.add_subplot(gs[2, 1])
        frame_counts = valid_df["frame"].value_counts().sort_index()
        ax5.bar(frame_counts.index, frame_counts.values, color="steelblue")
        ax5.set_title("프레임별 기사 수", fontsize=12, fontweight="bold")
        ax5.set_xlabel("프레임 ID")
        ax5.set_ylabel("기사 수")
        ax5.grid(True, alpha=0.3, axis="y")

        # 6. 편향도 히스토그램
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.hist(self.df["bias_score"], bins=20, edgecolor="black", alpha=0.7)
        ax6.axvline(x=-0.3, color="blue", linestyle="--", alpha=0.5, label="진보/중도")
        ax6.axvline(x=0.3, color="blue", linestyle="--", alpha=0.5, label="중도/보수")
        ax6.set_title("편향도 분포", fontsize=12, fontweight="bold")
        ax6.set_xlabel("편향도")
        ax6.set_ylabel("빈도")
        ax6.legend()

        plt.suptitle("프레임-편향도 통합 분석", fontsize=16, fontweight="bold", y=0.995)

        if save_dir:
            save_path = Path(save_dir) / "comprehensive_analysis.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 종합 분석 시각화 저장: {save_path}")
            plt.close()
        else:
            plt.show()


def run_integrated_analysis(
    articles_path: str = "data/input/articles.json",
    frames_path: str = "results/frames.json",
    article_frames_path: str = "results/article_frames.json",
    output_dir: str = "results/analysis",
):
    """
    통합 분석 실행 (테스트/실행용)

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

    # 분석기 초기화
    analyzer = IntegratedAnalyzer(articles, frame_assignments, frame_probs, frames)

    # 종합 분석
    report = analyzer.create_comprehensive_report(
        save_path=Path(output_dir) / "analysis_report.json"
    )

    # 시각화
    analyzer.visualize_comprehensive_analysis(save_dir=output_dir)

    # 개별 분석
    analyzer.analyze_frame_bias_correlation()
    analyzer.chi_square_test()
    analyzer.anova_test()
    analyzer.analyze_media_frame_preference()
    analyzer.find_discriminative_frames()

    print("\n✓ 통합 분석 완료")
    return analyzer


if __name__ == "__main__":
    # 필요한 파일이 있으면 분석 실행
    if (
        Path("data/input/articles.json").exists() and
        Path("results/frames.json").exists() and
        Path("results/article_frames.json").exists()
    ):
        run_integrated_analysis()
    else:
        print("⚠️ 필요한 파일을 찾을 수 없습니다.")
        print("  1. generate_sample_data.py 실행")
        print("  2. frame_extractor.py 실행")
        print("  이후 이 스크립트를 실행하세요.")