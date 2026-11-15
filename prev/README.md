# Does persona matching improve emotion attribution in LLMs?

MVP of the final project: ISEAR stories, two countries (Zambia, China), one model, one prompt, five persona conditions.
Design and decision rules are in [preregistration.md](preregistration.md).

## Run order

```powershell
uv sync
$env:OPENAI_API_KEY = "sk-..."        # only needed for notebook 02
```

1. **`01_prepare_data.ipynb`**: downloads ISEAR, removes invalid answers, picks the two countries by rule, builds persona attributes, flags nationality cues, exports `data/validation_cue_sample.csv` and the power numbers. No model is used.
2. **Hand-code the cue sample**: write 1 or 0 in `hand_cue` in `data/validation_cue_sample.csv`.
3. **Fill in the date in `preregistration.md` and commit it.**
4. **`02_run_experiment.ipynb`**:
   - Test the pipeline with `BACKEND = "mock"` (random answers, never report them).
   - Pilot with `LIMIT = 20` and `BACKEND = "api"`; every output should be one label. Fix the prompt now and note it in the pre-registration if needed.
   - Full run with `LIMIT = None` (1,323 stories x 5 conditions = 6,615 calls). It resumes if stopped.
5. **`03_analyze.ipynb`** (set `RUN_NAME` to the results folder). The first run exports `results/<model>/validation_parser_sample_n<rows>.csv`.
6. **Hand-code the parser sample**: write the label you read in `raw_output` (or `invalid`) in `hand_label`, without looking at the story's gold label. Run notebook 03 again: it adds agreement by condition and the PPI-corrected H1 effect.
7. Everything for the slides is in `results/<model>/summary.json`.

## What to report

- Accuracy by condition (clustered SEs), invalid-answer rate by condition.
- H1 and H2 decisions with the effect, one-sided p and 90% CI, and the cue-free sensitivity result.
- The naive and PPI-corrected H1 effect side by side. A large gap means parser errors depend on the condition.
- Parser agreement by condition and cue-dictionary recall and precision by country.
- The power note: the smallest detectable difference is about 2.5 to 3.9 pp, so a null rules out larger effects only.

## Notes

- The Hugging Face download is cached after the first run. If you are offline, set `HF_HUB_OFFLINE=1` to use `data_isear_mirror.csv` (it loses 163 stories that contain a `|`).
- Results are written to `results/<model>/`; the mock backend writes to `results/mock/` so it never mixes with real runs.
