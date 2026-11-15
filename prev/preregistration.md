# Pre-registration: does matching an LLM persona to the writer improve emotion attribution?

Date written: ____-__-__ (fill in and commit **before** running notebook 02 with `BACKEND = "api"`)
Author: Maayeesha Farzana

## Question and hypotheses

If an LLM reads the same ISEAR story under different personas, does matching the persona's nationality to the writer improve its accuracy at recovering the emotion the writer reported?

- **H1 (nationality matching).** For the same story, same model and same prompt, a persona with the writer's own nationality gives higher accuracy than a persona with the other country's nationality.
- **H2 (fuller persona).** A persona with the writer's nationality, sex, age and field of study gives higher accuracy than the nationality-only persona, and higher than a length-matched nationality persona (padded with a neutral sentence to the same number of words).

## Design (the MVP)

- Data: ISEAR (`savalera/isear-from-original`, all 7,666 stories). Invalid answers ("nothing", "never", empty, ...) are removed with the rule in notebook 01.
- Countries: the two with the most valid stories, fixed by this rule before any model output: **Zambia (770 stories, 110 writers)** and **China (553 stories, 79 writers)**. Total 1,323 stories, 189 writers.
- Model: `gpt-4o-mini`, temperature 0, `seed=2026`, max 10 tokens, one prompt template (below).
- Conditions, per story (5 calls): `none`, `matched` (nationality only), `mismatched` (the other country), `full` (nationality + sex + age + field), `length_matched`. With two countries there is exactly one mismatched nationality; this is a limitation (see below).
- Prompt: the persona is the system message; the user message is `Here is a situation you experienced: "<story>" What is the main emotion you felt in this situation? Answer with exactly one word from this list: joy, fear, anger, sadness, disgust, shame, guilt.`
- Outcome: the model's answer is correct if the one label it returns equals the writer's reported ISEAR emotion. Refusals, several labels, or labels outside the list are scored **incorrect** and their rate is reported separately.

## Decision rules (copied into notebook 03; not to be changed after seeing results)

- Primary model for H1: linear probability model with story fixed effects and a persona-country dummy, standard errors clustered by writer. The matched coefficient is the matching effect (an interaction of persona country and source country, net of both main effects).
- One-sided test, alpha = 0.05. Smallest effect of interest (SESOI) = 3 percentage points.
- **H1 is a YES** only if p < 0.05 and the effect is at least 3 pp. Otherwise NO. If the 90% CI upper bound is below 3 pp, the result is reported as a null that rules out an effect of that size.
- **H2 is a YES** only if full beats matched by at least 3 pp (p < 0.05) **and** full beats the length-matched control (p < 0.05). Stories with an incomplete persona (missing sex, age or field) are dropped from H2.
- Checks, reported but not decisive: McNemar test on story pairs, conditional logit.

## Power

With 1,323 stories, 189 writers, a design effect for writer clustering, and a one-sided test at 80% power, the minimum detectable difference is about **2.5 to 3.9 pp** (discordance 10-20%, ICC 5-10%; notebook 01, `data/data_summary.json`). The SESOI of 3 pp sits inside that range, so a small effect may be missed and a null is only informative about larger effects.

## Measurement validation

Two automated measurements could have errors that go with the predictor, so both are checked by hand:

1. **Output parser** (raw answer to label). Hand-code 50 outputs per condition (blind to the gold label) in `results/<model>/validation_parser_sample_n*.csv`. Report agreement by condition, and report the **prediction-powered inference (PPI)** corrected H1 effect and CI (Angelopoulos et al., 2023) next to the naive one. If they disagree, the PPI estimate is the one reported.
2. **Nationality-cue dictionary** (flags country, currency, city and nationality words). Hand-code 60 stories per country in `data/validation_cue_sample.csv` (the dictionary's own flag is hidden). Report recall and precision by country. H1 is re-estimated without flagged stories.

## Mechanism diagnostics (exploratory)

- Stereotype signature: how far each persona moves the label distribution away from the no-persona distribution. Gold labels are balanced within each country, so a pure shift cannot raise accuracy.
- Information signature: the matching gain split by whether the no-persona model was right.

## Known limitations

- Two countries and one model: the result is a proof of concept, not a general claim. More countries or models are not used to rescue a null.
- One mismatched country, so the mismatched control could itself be unusually compatible or incompatible with the stories. The full study would use several mismatched countries.
- The full persona is also longer; the length-matched control addresses this only partly, because the padding sentence is neutral in content, not in style.
- ISEAR labels are what the writer reported, 1990s student samples; accuracy is relative to that label only.
- Chinese ISEAR respondents wrote in English, which may carry cues the dictionary misses.

## Changes after this date

List any deviation here with a date and a reason (for example a prompt change after the pilot).

-
