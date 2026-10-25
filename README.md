# Persona matching and emotion attribution: MVP

This project tests whether telling an LLM the real nationality of the person who wrote an emotional story makes the model better at recovering the emotion that person reported. The MVP uses ISEAR stories from two countries, five persona conditions, one local model and a pre-registered analysis.

## Folder contents

| File | What it does |
|---|---|
| `01_prepare_data.ipynb` | Downloads ISEAR, cleans it, picks the two countries, builds personas, flags nationality cues, computes power |
| `02_run_experiment.ipynb` | Runs every story under the five personas and saves the raw replies |
| `03_analyze.ipynb` | Pre-registered tests, diagnostics, validation and all figures |
| `preregistration.md` | The decision rules, written before the run |
| `data/` | Story table, data summary and the cue sample for hand-coding |
| `results/` | Raw replies (`raw.jsonl`), `summary.json`, validation files and `figures/` |

## 0. Setup

```
pip install pandas numpy scipy statsmodels openai tabulate huggingface_hub pyarrow matplotlib
```

The model runs locally and for free with [Ollama](https://ollama.com). After installing it, download the model once in a terminal:

```
ollama pull llama3.2:3b
```

## 1. Prepare the data (`01_prepare_data.ipynb`)

The data come from the source cited in the project sheet, `savalera/isear-from-original` on Hugging Face, downloaded inside the notebook with `hf_hub_download`. Write the download date in the pre-registration.

The notebook removes non-answers (such as "No response."), applies the pre-registered country rule (the two countries with the most valid stories, which gives Zambia and Spain), builds each writer's persona from the codebook (country, sex, age, field of study), flags stories that contain explicit nationality cues, exports `data/validation_cue_sample.csv` for hand-coding, and computes the smallest detectable difference.

The code mappings follow the ISEAR codebook as documented in `enums.py` of the py_isear_dataset repository; the sample row on the project sheet (ID 201072: Zambia, woman, 19, law, guilt) confirms them for the Hugging Face file. `0` is treated as missing for sex and field of study.

## 2. Run the experiment (`02_run_experiment.ipynb`)

In the Settings cell, use:

```python
MODEL = "llama3.2:3b"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
WORKERS = 2
```

Run a pilot with `LIMIT = 20`, check that the replies are single emotion words, then set `LIMIT = None`. The run appends to `results/raw.jsonl` and skips calls already done, so it can be stopped and restarted.

The five conditions, with only the system prompt changing:

| Condition | System prompt (Spanish writer) |
|---|---|
| Neutral | You are a person. |
| Mismatched | You are a person from Zambia. |
| Matched | You are a person from Spain. |
| Full | You are a 20-year-old woman from Spain who studies psychology. |
| Length-matched | You are a person from Spain. Please read this carefully. |

Settings: temperature 0, seed 2026, `max_tokens=10`, answer restricted to the seven ISEAR labels. MVP size: 1,056 stories (Zambia 621, Spain 435) from 188 writers, 5,280 calls.

## 3. Analyse (`03_analyze.ipynb`)

Restart the kernel and run all cells. The notebook:

1. scores every reply against the writer's own label (refusals and unusable replies count as wrong);
2. tests H1 (matched vs mismatched) with a linear probability model, story fixed effects, persona-country effects and errors clustered by writer, plus conditional logit and McNemar as checks;
3. tests H2 (full vs matched, and full vs length-matched);
4. repeats H1 without stories that contain nationality cues;
5. runs the diagnostics: label mix by persona, gain on hard vs easy stories, how often answers change, and the types of invalid replies;
6. exports the validation samples and saves all figures to `results/figures/`.

Decision rule: H1 is YES only if the one-sided p-value is below 0.05 and the effect is at least 3 points. If the upper end of the 90% confidence interval is below 3 points, a meaningful effect is ruled out.

## MVP results (llama3.2:3b, Zambia and Spain)

| Condition | Accuracy |
|---|---|
| Neutral | 63.5% |
| Mismatched | 62.5% |
| Matched | 64.1% |
| Full | 64.4% |
| Length-matched | 62.3% |

- **H1: NO.** Matched minus mismatched is +1.47 points (one-sided p = 0.018, 90% CI 0.32 to 2.62). The effect is real but small, and a 3-point effect is ruled out. Without cue stories: +1.33 points.
- **H2: NO.** Full minus matched is +0.29 points (p = 0.33). One neutral sentence (length-matched) lowered accuracy by 1.8 points.
- **Answers rarely move:** 91% of stories get the same answer with the matched and mismatched persona; matching fixed 37 stories and broke 20.
- **Refusals:** safety refusals occurred 28 times with the matched persona and 38 times with the mismatched one, so part of the gain comes from refusing less, not from reading emotion better.
- **Hard stories:** the gain is +3.9 points on stories the neutral persona gets wrong and +0.3 on those it gets right.
- **Label mix:** every persona, including the neutral one, over-predicts sadness and under-predicts disgust, which points to a model bias rather than a national stereotype.

## Known limitations and next steps

1. **Cut-off replies.** About 70 replies were cut off by `max_tokens=10` before naming an emotion. A future run should use `max_tokens=30` for every condition, which means rerunning all calls after renaming `results/raw.jsonl`.
2. **Hand-coding** of the parser and cue samples is still to be done.
3. **Scope.** One small model and two countries; the full project should add more countries, a larger model and a new pre-registration, for example on whether the matching effect differs by country.

## Testing without a model

`BACKEND = "mock"` in notebook 2 produces random answers so you can check that everything runs. Its numbers mean nothing and must never be reported; delete `results/raw.jsonl` afterwards.
