"""
Step 1 of the MVP: prepare the ISEAR stories.

What this script does, in order:
  1. Loads the ISEAR file (pipe-separated, original column names).
  2. Removes invalid answers ("No response", "Same as in anger", ...),
     following the exclusion logic of Turdubaeva & Lee (2026).
  3. Picks the two source countries with a rule fixed in advance:
     the two countries with the most valid stories.
  4. Builds the persona attributes (sex, age, field of study) for each writer.
  5. Flags stories that contain explicit nationality cues (dictionary).
  6. Exports a blind sample of stories for hand-coding the cue flag
     (Edgar's validation step: does the dictionary make errors that differ by country?).
  7. Computes the smallest difference the MVP can detect (before any model is run).

Usage:
  python prepare_data.py --isear data_isear_mirror.csv --out data/
"""
import argparse, json, math, re
from pathlib import Path
import pandas as pd

# ISEAR codebook (Scherer & Wallbott); check against the codebook shipped with the UNIGE file.
# ISEAR codebook (Scherer & Wallbott): all 37 country codes. Verify against the UNIGE codebook.
COUNTRIES = {
    "1": "Sweden", "2": "Norway", "3": "West Germany", "4": "Finland", "5": "Greece",
    "6": "Hong Kong", "7": "Lebanon", "8": "Austria", "9": "Australia", "10": "Brazil",
    "11": "Botswana", "12": "Bulgaria", "13": "France", "14": "Italy", "15": "Japan",
    "16": "New Zealand", "17": "the Netherlands", "18": "Portugal", "19": "Spain",
    "20": "Zambia", "21": "Zimbabwe", "22": "the United States", "23": "Poland",
    "24": "Nigeria", "25": "Israel", "26": "India", "27": "Malawi", "28": "Switzerland",
    "29": "Chile", "30": "China", "31": "Yugoslavia", "32": "Costa Rica", "33": "Honduras",
    "34": "Mexico", "35": "Guatemala", "36": "Venezuela", "37": "El Salvador",
}
SEX = {"1": "man", "2": "woman"}                      # 0 = missing
FIELD = {"1": "psychology", "2": "the social sciences", "3": "languages",
         "4": "fine arts", "5": "law", "6": "natural sciences",
         "7": "engineering", "8": "medicine", "9": "another subject"}  # 0 = missing
EMOTIONS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

INVALID = re.compile(
    r"^\[?\s*(?:nothing|none|no response|no answer|no description|not applicable|"
    r"not included|blank|n/?a\b|do not (know|remember|recall)|can'?t remember|"
    r"cannot (remember|recall)|can not (remember|recall|think)|never|i (have )?never|"
    r"i can ?not (recall|remember|think)|i do not (recall|remember)|same as|the same|as for|"
    r"doesn'?t apply)", re.I)

# Explicit nationality cues for the countries in the MVP. Extend before running.
CUES = {
    "Sweden": ["swed\\w*", "stockholm", "uppsala", "gothenburg", "g[oö]teborg", "malm[oö]",
               "lund", "ume[aå]", "krona", "kronor", "scandinavia\\w*", "nordic"],
    "Zambia": ["zambia\\w*", "lusaka", "kitwe", "ndola", "kabwe", "livingstone", "copperbelt",
               "kwacha", "ngwee", "bemba", "nyanja", "lozi", "tonga", "chewa", "nshima",
               "chitenge", "africa\\w*"],
    "China": ["china", "chinese", "beijing", "peking", "shanghai", "canton\\w*", "guangzhou",
              "yuan", "renminbi", "rmb", "mao", "communist party", "cultural revolution",
              "spring festival", "lunar new year", "taiwan\\w*", "hong kong", "korea\\w*",
              "japan\\w*", "opium war", "asia\\w*"],
    "_any": ["europe\\w*", "america\\w*", "english", "british"],
}


def load_isear(path):
    """Load ISEAR from parquet or CSV and return every column as a clean string.
    Works for the Hugging Face parquet (savalera/isear-from-original) and pipe/comma CSVs."""
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            sep = "|" if "|" in f.readline() else ","
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip")
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("Int64")
        df[c] = df[c].astype("string").fillna("").str.strip()
    if "Field1" not in df.columns:            # emotion as a word
        df["Field1"] = df["EMOT"]
    if "MYKEY" not in df.columns:
        df["MYKEY"] = ""
    return df


def mde(n_stories, n_writers, discordance, icc, alpha_z=1.645, power_z=0.842):
    """Minimum detectable difference for a paired binary outcome, one-sided test.

    discordance = share of stories whose correctness differs between the two conditions.
    icc         = correlation of outcomes between stories of the same writer.
    """
    m = n_stories / n_writers
    design_effect = 1 + (m - 1) * icc
    n_eff = n_stories / design_effect
    return (alpha_z + power_z) * math.sqrt(discordance / n_eff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isear", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_countries", type=int, default=2)
    ap.add_argument("--cue_sample_per_country", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load_isear(a.isear)
    df = df[df.COUN.isin(COUNTRIES)].copy()
    df["country"] = df.COUN.map(COUNTRIES)
    df["writer_id"] = df.COUN + "_" + df.SUBJ
    df["story_id"] = df.MYKEY.where(df.MYKEY != "", df.ID + "_" + df.EMOT)
    df["emotion"] = df.Field1.str.strip().str.lower()
    df["text"] = (df.SIT.fillna("").str.replace("á", "", regex=False)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    df["invalid"] = df.text.str.len().lt(5) | df.text.str.contains(INVALID)
    df = df[df.emotion.isin(EMOTIONS)]

    print(f"Loaded {len(df)} stories; removed {df.invalid.sum()} invalid answers.")
    valid = df[~df.invalid].copy()

    # Pre-registered country rule: the n countries with the most valid stories.
    counts = valid.groupby("country").size().sort_values(ascending=False)
    chosen = list(counts.index[: a.n_countries])
    print("Valid stories per country:\n", counts.to_string())
    print("Chosen countries (largest valid N):", chosen)

    s = valid[valid.country.isin(chosen)].copy()
    s["sex"] = s.SEX.map(SEX)
    s["age"] = pd.to_numeric(s.AGE, errors="coerce")
    s["field"] = s.FIEL.map(FIELD)
    s["full_persona_complete"] = s[["sex", "age", "field"]].notna().all(axis=1)

    pattern = {c: re.compile(r"\b(?:" + "|".join(CUES[c] + CUES["_any"]) + r")\b", re.I)
               for c in chosen if c in CUES}
    s["cue_flag"] = [bool(pattern[c].search(t)) if c in pattern else False
                     for c, t in zip(s.country, s.text)]

    cols = ["story_id", "writer_id", "country", "emotion", "text", "sex", "age",
            "field", "full_persona_complete", "cue_flag"]
    s[cols].to_csv(out / "stories.csv", index=False)

    # Blind sample for hand-coding the cue flag (coder does not see the dictionary result).
    samp = s.groupby("country").sample(n=a.cue_sample_per_country, random_state=a.seed)
    samp[["story_id", "country", "text"]].assign(hand_cue="").to_csv(
        out / "validation_cue_sample.csv", index=False)

    # Power: computed now, before any model output exists.
    n, w = len(s), s.writer_id.nunique()
    power = {f"d={d}_icc={icc}": round(100 * mde(n, w, d, icc), 2)
             for d in (0.10, 0.20) for icc in (0.05, 0.10)}
    summary = {
        "source_file": a.isear, "countries": chosen,
        "stories_per_country": s.groupby("country").size().to_dict(),
        "writers_per_country": s.groupby("country").writer_id.nunique().to_dict(),
        "stories_total": n, "writers_total": w,
        "labels_per_country": pd.crosstab(s.country, s.emotion).to_dict(orient="index"),
        "full_persona_complete": int(s.full_persona_complete.sum()),
        "cue_flagged": s.groupby("country").cue_flag.sum().to_dict(),
        "mde_percentage_points": power,
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
