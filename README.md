# Video SimpleQA: Towards Factuality Evaluation in Large Video Language Models

## 📖 Overview

Video SimpleQA is the first comprehensive benchmark specifically designed for evaluating factual grounding capabilities in Large Video Language Models (LVLMs). Unlike existing video benchmarks that often involve subjective speculation or conflate factual grounding with reasoning skills, Video SimpleQA focuses exclusively on objective factuality evaluation through multi-hop fact-seeking questions.

### Key Features

- **🎯 Knowledge Required**: Questions demand integration of external knowledge beyond video content
- **🔗 Multi-hop Fact-seeking**: Each question involves multiple explicit facts with step-by-step sub-QAs
- **✅ Short-form Definitive Answers**: Unambiguous, universally agreed-upon answers
- **⏰ Temporal Grounding**: Answers rely on temporal segments rather than single frames
- **🌍 Open Domain**: Covers diverse video types across 4 primary and 84 tertiary categories

## 📊 Dataset Statistics

- **Total QA Pairs**: 1,504
- **Unique Videos**: 1,079
- **Multi-hop Questions**: 2/3/4-hop (928/469/107 pairs)
- **Categories**: 4 primary, 15 secondary, 84 tertiary
- **Average Question Length**: 15.64 words
- **Average Answer Length**: 1.28 words

## 📁 Code Structure

```
code/
├── bestofn_evaluation.py          # Best-of-N evaluation strategy
├── metrics_analyzer.py            # Performance metrics analysis
├── multi_round_qa_fscore.py      # Multi-round QA F-score calculation
├── self_refine_evaluation.py     # Self-refinement evaluation
├── video_qa_evaluation_multi_round.py  # Multi-round video QA evaluation
├── video_qa_evaluation.py        # Main video QA evaluation script
├── video_qa_generator.py         # Video QA generation utilities
├── download.py                   # Video downloader from Wikimedia Commons
data/
├── VideoSimpleQA.json        		# Main dataset file
README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install openai asyncio tqdm multiprocessing requests beautifulsoup4
```

**For video downloading, also install FFmpeg:**

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

## 📥 Video Download Setup

### Download Videos from Wikimedia Commons

The videos for VideoSimpleQA can be downloaded using the provided download script:

```bash
python download.py
```

**Output:**

- Downloaded MP4 videos will be saved in `videos/` directory
- Videos are named using the date format: `YYYY-MM-DD.mp4`

## 📋 Execution Examples

### 1. Video QA Generation (`video_qa_generator.py`)

Generate question-answer pairs from video frames and descriptions:

```bash
python video_qa_generator.py
```

**Configuration required:**

- Update `API_KEY` and `BASE_URL` in the script
- Prepare video frames in `../../data/wiki_videos/frames_15/{video_id}/`
- Create `descriptions.json` with video descriptions
- Create `videos_name.txt` with list of video files

**Expected Output:**

- `output.json`: Generated QA pairs with verification status
- `error_output.txt`: Error log for failed processing

### 2. Single Video QA Evaluation (`video_qa_evaluation.py`)

Evaluate a model's performance on video QA tasks:

```bash
python video_qa_evaluation.py \
    --target-model "gpt-4-vision-preview" \
    --grader-model "gpt-4" \
    --frame-num 32 \
    --data-file "VideoSimpleQA.json" \
    --pool-processes 20
```

**Parameters:**

- `--target-model`: Model to be evaluated
- `--grader-model`: Model used for grading responses
- `--frame-num`: Number of frames to use (default: 32)
- `--frames-path`: Path to frames directory (default: `./frames_{frame_num}/`)
- `--data-file`: Path to evaluation dataset
- `--pool-processes`: Number of parallel processes

**Expected Output:**

- `evaluation_results_{model}_{frames}frames.json`: Detailed evaluation results
- `error_log_{model}.txt`: Error log

### 3. Multi-Round QA Evaluation (`video_qa_evaluation_multi_round.py`)

Evaluate multi-round question answering:

```bash
python video_qa_evaluation_multi_round.py \
    --target-model "claude-sonnet4" \
    --grader-model "gpt-4" \
    --frame-num 30 \
    --data-file "./data/test_data.json" \
    --output-file "./results/multi_round_results.json" \
    --processes 20
```

**Parameters:**

- `--frame-num`: Number of frames to use
- `--frames-path`: Path to frames directory
- `--target-model`: Model to evaluate (required)
- `--grader-model`: Grading model (required)
- `--output-file`: Output file path
- `--processes`: Number of parallel processes

**Expected Output:**

- Evaluation results with QA pair grades and Multi-hop evaluation
- Progress tracking and error logging

### 4. F-Score Calculation (`multi_round_qa_fscore.py`)

Calculate F-scores for multi-round QA results:

```bash
python multi_round_qa_fscore.py
```

**Input Requirements:**

- Place evaluation result JSON files in `evaluation_results/` directory
- Files should follow naming pattern: `evaluation_results_*.json`

**Expected Output:**

- `all_models_fscore.json`: F-scores for all models
- Console summary table showing F-scores for each QA round and Multi-hop evaluation

**Example Output:**

```
MODEL F-SCORE SUMMARY (F1 Scores in %)
========================================
Model                          Samples  QA_Pair_1    QA_Pair_2    Multi_hop   
claude-sonnet4                 1504     65.2         58.7         42.3        
gpt-4-vision                   1504     72.1         69.4         55.8        
```

### 5. Metrics Analysis (`metrics_analyzer.py`)

Analyze evaluation metrics with category and round breakdowns:

```bash
python metrics_analyzer.py
```

**Input Requirements:**

- Evaluation result files in `evaluation_results/` directory
- Optional: `category_mapping.json` for category analysis
- Optional: `round_mapping.json` for round analysis

**Expected Output:**

- `all_model_metrics.json`: Comprehensive metrics
- Console tables showing overall, category-wise, and round-wise F-scores

**Features:**

- Overall accuracy, attempt rates, F1 scores
- Category-wise performance breakdown
- Round-wise performance analysis

### 6. Self-Refine Evaluation (`self_refine_evaluation.py`)

Evaluate models using iterative self-refinement:

```bash
python self_refine_evaluation.py
```

**Configuration:**

- Update `API_KEY` and `BASE_URL`
- Configure `MODEL_FRAMES_CONFIG` with model specifications
- Set `data_file` path to your dataset

**Process:**

1. Generate initial answer
2. Provide feedback on the answer
3. Refine answer based on feedback
4. Repeat for up to 3 iterations

**Expected Output:**

- `self_refine_results.json`: Final results with accuracies per iteration
- `self_refine_intermediate_results.json`: Intermediate results for recovery
- `error_log_self_refine.txt`: Error log

### 7. Best-of-N Evaluation (`bestofn_evaluation.py`)

Evaluate using Best-of-N sampling strategy:

```bash
python bestofn_evaluation.py
```

**Configuration:**

- Update `API_KEY` and `BASE_URL`
- Configure `MODEL_FRAMES_MAP` with model-to-frames mapping
- Set dataset file path

**Process:**

1. Generate N candidate answers per question
2. Select best answer using selector model
3. Evaluate accuracy for different N values (1, 2, 4, 8, 16)

**Expected Output:**

- `best_of_n_results.json`: Accuracy results for different N values
- `intermediate_results.json`: Intermediate candidate answers
- `error_log_BoN.txt`: Error log

## License <a name="license"></a>

The project is released under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). They are available strictly for non-commercial research. More guidelines of dataset can be found in [here](./DATA.md#license).


## Opt-Out Approach <a name="opt-out-approach"></a>

We uphold the rights of individuals and copyright holders. If you are featured in any of our video annotations or hold copyright to a video and wish to have its annotation removed from our dataset, please reach out to us. Send an email to mengcaopku@gmail.com with the subject line beginning with *VideoSimpleQA-optout*, or raise an issue with the same title format. We commit to reviewing your request promptly and taking suitable action.