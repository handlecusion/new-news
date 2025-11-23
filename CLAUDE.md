# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean news frame analysis project that uses a hybrid unsupervised/supervised learning approach to:
1. **Unsupervised learning:** Automatically discover frame patterns from news data
2. **Supervised learning:** Predict media outlet bias based on bias scores
3. **Integrated analysis:** Analyze relationships between discovered frames and bias scores

The project analyzes news articles about a single issue (e.g., minimum wage increase) to identify different narrative frames and correlate them with media outlet political bias.

## Key Technology Stack

- **NLP/ML:** BERTopic for frame discovery, KoBERT for bias classification, sentence-transformers for embeddings
- **Korean NLP:** KoNLPy (Mecab) for morphological analysis
- **ML Frameworks:** PyTorch, scikit-learn, transformers
- **Visualization:** matplotlib, seaborn, plotly for interactive dashboards

## Data Format

Input data should be in JSON format at `data/input/articles.json`:

```json
{
  "metadata": {
    "issue": "최저임금 인상",
    "collection_period": "2024-01-01 ~ 2024-01-31",
    "total_articles": 150
  },
  "articles": [
    {
      "article_id": "20240115_hankyoreh_001",
      "media_outlet": "한겨레",
      "bias_score": -0.7,
      "title": "article title",
      "content": "article content...",
      "published_date": "2024-01-15",
      "url": "https://..."
    }
  ]
}
```

Where `bias_score` ranges from -1.0 (progressive) to +1.0 (conservative).

## Project Structure

```
news-frame-bias-analysis/
├── data/
│   ├── input/
│   │   └── articles.json          # Input data
│   └── processed/
│       └── embeddings.npy         # Cached document embeddings
├── src/
│   ├── preprocessing/
│   │   ├── text_preprocessor.py   # Text cleaning and tokenization
│   │   └── embedder.py            # Document embedding generation
│   ├── unsupervised/
│   │   ├── frame_extractor.py     # BERTopic-based frame extraction
│   │   └── visualizer.py          # Frame visualization
│   ├── supervised/
│   │   ├── bias_classifier.py     # KoBERT bias classification
│   │   └── frame_predictor.py     # Frame-based bias prediction
│   ├── analysis/
│   │   ├── correlation.py         # Frame-bias correlation analysis
│   │   ├── frame_interpreter.py   # Frame interpretation & representative examples
│   │   └── dashboard.py           # Interactive dashboard creation
│   └── pipeline.py                # Main execution pipeline
├── notebooks/                     # Jupyter notebooks for exploration
├── models/                        # Trained models
├── results/                       # Analysis results and visualizations
└── requirements.txt
```

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Note: KoNLPy requires additional system dependencies
# macOS: brew install mecab mecab-ko mecab-ko-dic
# Ubuntu: apt-get install python3-dev default-jre
```

### Running the Pipeline
```bash
# Run full analysis pipeline
python src/pipeline.py

# Results will be saved to results/ directory
```

### Frame Interpretation (Understanding Why Frames Are Distinguished)
```bash
# Analyze frame characteristics and extract representative sentences
python interpret_frames.py

# This will generate results/analysis/frame_interpretation.json with:
# - Representative articles for each frame
# - Key sentences that define each frame
# - Explanation of why frames are distinguished
# - Frame characteristics (bias tendency, consistency, etc.)
```

### Running Individual Components
```bash
# Frame extraction only
python -m src.unsupervised.frame_extractor

# Bias classification only
python -m src.supervised.bias_classifier
```

### Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Notebooks are located in notebooks/ directory:
# - 01_data_exploration.ipynb
# - 02_unsupervised_frames.ipynb
# - 03_supervised_bias.ipynb
# - 04_integrated_analysis.ipynb
```

## Architecture Notes

### Two-Path Analysis Pipeline

The pipeline splits into two parallel paths after preprocessing:

1. **Unsupervised Path (Frame Discovery):**
   - Uses BERTopic (BERT + HDBSCAN clustering + c-TF-IDF)
   - Automatically discovers narrative frames from article content
   - Does not use bias labels - purely data-driven
   - Output: Frame clusters with representative keywords

2. **Supervised Path (Bias Prediction):**
   - KoBERT-based sequence classification
   - Trained on existing bias_score labels
   - Predicts progressive/neutral/conservative classification
   - Alternative: Frame distribution as features for bias prediction

3. **Integration:**
   - Correlates discovered frames with bias scores
   - Analyzes which frames are associated with political leanings
   - Generates interactive visualizations and dashboards

### Key Design Decisions

**Why BERTopic for Frame Extraction:**
- Captures semantic/contextual similarity via BERT embeddings
- HDBSCAN automatically determines number of topics (no manual k selection)
- c-TF-IDF provides interpretable keywords per frame
- Built-in interactive visualizations

**Bias Score Categorization:**
- < -0.3: Progressive (label 0)
- -0.3 to 0.3: Neutral (label 1)
- > 0.3: Conservative (label 2)

**Korean Text Processing:**
- Mecab tokenizer for morphological analysis
- Filters for nouns (N*), verbs (V*), and adjectives (VA*)
- Uses `jhgan/ko-sroberta-multitask` for Korean sentence embeddings
- Uses `skt/kobert-base-v1` for bias classification

### Important Implementation Details

**Text Preprocessing:**
- Combines title + content for each article
- Removes HTML tags, normalizes whitespace
- Filters stopwords (있다, 하다, 되다, 이다, 것, etc.)
- Only keeps content words (nouns, verbs, adjectives)

**Frame Interpretation:**
When analyzing discovered frames, consider:
1. What perspective do the keywords represent?
2. Which stakeholders' interests are emphasized?
3. What emotional tone (positive/negative/neutral)?
4. Focus on causes vs. consequences vs. solutions?

**Model Training:**
- Train/test split: 80/20 with stratification on bias labels
- KoBERT max sequence length: 512 tokens
- BERTopic minimum topic size: 5 documents
- All random operations use seed 42 for reproducibility

## Output Files

After running the pipeline, expect these outputs:

- `results/frames.json` - Discovered frame information with keywords
- `results/article_frames.json` - Frame assignments for each article
- `results/analysis/frame_interpretation.json` - Frame interpretation data (JSON)
- `results/analysis/report.json` - Comprehensive correlation analysis report
- `results/figures/topic_map.html` - Interactive topic similarity map
- `results/figures/topic_keywords.html` - Top keywords per frame
- `results/figures/integrated_analysis.png` - Frame-bias correlation plots
- `results/dashboard.html` - Main interactive dashboard
- `results/frame_explorer.html` - Frame-based article browser
- **`results/frame_interpretation.html`** - ⭐ **Frame interpretation dashboard with representative sentences and distinction reasons**
- `models/bias_classifier/` - Trained KoBERT model checkpoints

## Korean Language Considerations

This project processes Korean text, so:
- All text cleaning should preserve Korean characters (Hangul)
- Use Korean-specific tokenizers (Mecab, not NLTK)
- Use Korean pre-trained models (KoBERT, ko-sroberta)
- Stopword lists should include common Korean function words
- Frame names and interpretations should be in Korean

## Development Status

**Current State:** Project structure defined in `news.md`, implementation pending

**Next Steps:**
1. Create directory structure (data/, src/, etc.)
2. Implement preprocessing modules
3. Implement unsupervised frame extraction
4. Implement supervised bias classification
5. Build integration and visualization components
