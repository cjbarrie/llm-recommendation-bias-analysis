#!/usr/bin/env python3
"""
Prepare YouGov Linked Dataset for LLM Recommendation Bias Analysis

Merges three data sources:
1. Crosswalk: twitter_id <-> nagler_id/respid_N
2. Wave survey data (waves 0-9): ground-truth demographics
3. Tweet parquet files: original tweet text from panel users

User selection criteria (applied before sampling):
- Must have ALL three demographics: gender, race, ideology (no unknowns)
- Must have at least --min-tweets-per-user original tweets

Then --n-users are sampled randomly from qualifying users, and exactly
--tweets-per-user tweets are kept per user.

Default: 1000 users × 50 tweets = 50,000 tweets total.

Outputs:
- datasets/personas.pkl           — experiment-ready (loaded by run_experiment.py)
- datasets/analysis_ready.parquet — all 16 features pre-computed

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --n-users 1000 --tweets-per-user 50 --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.metadata_inference import infer_tweet_metadata


# =============================================================================
# CONFIGURATION
# =============================================================================

YGHPC_DIR_DEFAULT = Path("/Users/christopherbarrie/Dropbox/sandbox/yghpc")

# Toxic keywords for simple toxicity scoring (matches process_survey_twitter_dataset.py)
TOXIC_KEYWORDS = [
    "hate", "kill", "die", "stupid", "idiot", "dumb", "shut up",
    "loser", "moron", "trash", "garbage", "pathetic", "disgusting",
]

# Political ideology mapping from survey values to standard categories
IDEOLOGY_MAP = {
    "very liberal":     "left",
    "liberal":          "center-left",
    "moderate":         "center",
    "conservative":     "center-right",
    "very conservative": "right",
    "not sure":         "unknown",
}

# Wave numbers to search for demographics
WAVE_NUMBERS = list(range(10))  # 0-9


# =============================================================================
# STEP 1: LOAD CROSSWALK
# =============================================================================

def load_crosswalk(yghpc_dir: Path) -> pd.DataFrame:
    """Load twitter_id <-> nagler_id crosswalk."""
    path = yghpc_dir / "aws_db" / "data" / "crosswalk.csv"
    cw = pd.read_csv(path)
    cw["twitter_id"] = cw["twitter_id"].astype(str)
    cw["nagler_id"] = cw["respid_N"].astype(str)
    print(f"Crosswalk: {len(cw):,} users")
    return cw[["twitter_id", "nagler_id"]]


# =============================================================================
# STEP 2: LOAD AND COALESCE WAVE DEMOGRAPHICS
# =============================================================================

def load_wave_demographics(yghpc_dir: Path) -> pd.DataFrame:
    """
    Load demographic columns from all waves and coalesce:
    take first non-null value per respondent across waves 0-9.

    Returns DataFrame with columns: nagler_id, gender, race, ideology, birth_year, educ
    """
    waves_dir = yghpc_dir / "aws_db" / "data"
    target_fields = ["gender", "race", "ideology", "birth_year", "educ"]

    all_rows = {}  # nagler_id -> {field: value}

    for wn in WAVE_NUMBERS:
        path = waves_dir / f"wave_{wn}_data.csv"
        if not path.exists():
            continue

        wave_df = pd.read_csv(path, low_memory=False)
        suffix = f"_w{wn}"

        # Build column mapping: field -> actual column name in this wave
        col_map = {}
        for field in target_fields:
            col = f"{field}{suffix}"
            if col in wave_df.columns:
                col_map[field] = col

        if "nagler_id" not in wave_df.columns:
            continue

        for _, row in wave_df.iterrows():
            nid = str(row["nagler_id"])
            if nid not in all_rows:
                all_rows[nid] = {}
            for field, col in col_map.items():
                if field not in all_rows[nid] or pd.isna(all_rows[nid][field]):
                    val = row.get(col)
                    if not pd.isna(val) if not isinstance(val, float) else not np.isnan(val):
                        all_rows[nid][field] = val

        print(f"  Wave {wn}: {len(wave_df):,} respondents, {len(col_map)} demo fields found")

    # Convert to DataFrame
    records = [{"nagler_id": nid, **fields} for nid, fields in all_rows.items()]
    demo_df = pd.DataFrame(records)

    # Ensure all target fields exist
    for field in target_fields:
        if field not in demo_df.columns:
            demo_df[field] = np.nan

    print(f"\nDemographics coalesced: {len(demo_df):,} unique respondents")
    for field in target_fields:
        n_filled = demo_df[field].notna().sum()
        print(f"  {field}: {n_filled:,} non-null ({n_filled/len(demo_df)*100:.1f}%)")

    return demo_df


# =============================================================================
# STEP 3: MAP TO STANDARD ANALYSIS COLUMNS
# =============================================================================

def map_demographics(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Map raw survey values to standard analysis columns."""
    df = demo_df.copy()

    # author_gender
    gender_map = {"female": "female", "male": "male"}
    df["author_gender"] = (
        df["gender"]
        .fillna("unknown")
        .str.lower()
        .str.strip()
        .map(gender_map)
        .fillna("unknown")
    )

    # author_political_leaning
    df["author_political_leaning"] = (
        df["ideology"]
        .fillna("unknown")
        .str.lower()
        .str.strip()
        .map(IDEOLOGY_MAP)
        .fillna("unknown")
    )

    # author_is_minority
    def is_minority(race_val):
        if pd.isna(race_val):
            return "unknown"
        return "no" if str(race_val).strip().lower() == "white" else "yes"

    df["author_is_minority"] = df["race"].apply(is_minority)

    # Print distributions
    for col in ["author_gender", "author_political_leaning", "author_is_minority"]:
        print(f"\n  {col}:")
        for val, cnt in df[col].value_counts().items():
            print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    return df


# =============================================================================
# STEP 4a: COUNT ORIGINAL TWEETS PER USER (fast pass — no text loaded)
# =============================================================================

def count_original_tweets(yghpc_dir: Path, target_ids: set) -> dict[str, int]:
    """
    Pass 1: count original tweets per user without loading text.
    Only reads user_id and retweeted_status_id columns — much faster.
    """
    tweets_dir = yghpc_dir / "aws_db" / "data" / "tweets"
    parquet_files = sorted(tweets_dir.glob("*.parquet"))

    print(f"  Scanning {len(parquet_files)} parquet files (counts only)...")
    counts: dict[str, int] = {}

    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf, columns=["user_id", "retweeted_status_id"])
        df["user_id"] = df["user_id"].astype(str)
        df = df[df["user_id"].isin(target_ids)]
        orig = df[df["retweeted_status_id"].isna()]
        for uid, cnt in orig.groupby("user_id").size().items():
            counts[uid] = counts.get(uid, 0) + int(cnt)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(parquet_files)}] counted so far: "
                  f"{len(counts):,} users with original tweets")

    return counts


# =============================================================================
# STEP 4b: LOAD TWEET CONTENT FOR SELECTED USERS (full pass — text loaded)
# =============================================================================

def load_tweets_for_users(
    yghpc_dir: Path,
    selected_ids: set,
    tweets_per_user: int,
) -> pd.DataFrame:
    """
    Pass 2: load full tweet content for the selected user subset only.

    Args:
        selected_ids: twitter_ids to load (already sampled 1000 users).
        tweets_per_user: Exactly this many tweets kept per user.

    Returns:
        DataFrame with columns: tweet_id, user_id, text, created_at,
                                 favorite_count, retweet_count
    """
    tweets_dir = yghpc_dir / "aws_db" / "data" / "tweets"
    parquet_files = sorted(tweets_dir.glob("*.parquet"))

    print(f"  Loading tweet content for {len(selected_ids):,} users "
          f"from {len(parquet_files)} parquet files...")

    user_tweets: dict[str, list] = {}

    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf)
        df["user_id"] = df["user_id"].astype(str)
        df = df[df["user_id"].isin(selected_ids)]
        if df.empty:
            continue

        # Original tweets only
        if "retweeted_status_id" in df.columns:
            df = df[df["retweeted_status_id"].isna()]

        # Extract plain text: formatted_text is a list of (key, val) tuples;
        # take the 'text' entry. Fall back to full_text then text.
        if "formatted_text" in df.columns:
            def _extract(ft):
                if isinstance(ft, list):
                    for k, v in ft:
                        if k == "text" and v:
                            return str(v).strip()
                return ""
            df["_text"] = df["formatted_text"].apply(_extract)
        elif "full_text" in df.columns:
            df["_text"] = df["full_text"].fillna("").astype(str).str.strip()
        else:
            df["_text"] = df["text"].fillna("").astype(str).str.strip()
        df = df[df["_text"] != ""]

        for uid, grp in df.groupby("user_id"):
            if uid not in user_tweets:
                user_tweets[uid] = []
            # Stop collecting once we have enough
            needed = tweets_per_user - len(user_tweets[uid])
            if needed > 0:
                user_tweets[uid].extend(grp.head(needed).to_dict("records"))

        if (i + 1) % 20 == 0:
            complete = sum(1 for t in user_tweets.values() if len(t) >= tweets_per_user)
            print(f"    [{i+1}/{len(parquet_files)}] "
                  f"{len(user_tweets):,} users found, "
                  f"{complete:,} with {tweets_per_user}+ tweets")

    # Exactly tweets_per_user per user
    records = []
    for uid, tweets in user_tweets.items():
        records.extend(tweets[:tweets_per_user])

    df_out = pd.DataFrame(records).rename(columns={"_text": "message"})
    keep_cols = ["tweet_id", "user_id", "message", "created_at", "favorite_count", "retweet_count"]
    existing = [c for c in keep_cols if c in df_out.columns]
    df_out = df_out[existing].copy()

    if "tweet_id" not in df_out.columns:
        df_out["tweet_id"] = range(len(df_out))

    print(f"  Loaded: {len(df_out):,} tweets from {df_out['user_id'].nunique():,} users")
    return df_out


# =============================================================================
# STEP 5-7: TOXICITY FEATURES
# =============================================================================

def compute_toxicity(text: str) -> tuple[float, float]:
    """Simple keyword-based toxicity scoring."""
    text_lower = str(text).lower()
    count = sum(1 for kw in TOXIC_KEYWORDS if kw in text_lower)
    toxicity = min(count / 5.0, 1.0)
    return toxicity, toxicity * 0.5


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare YouGov linked dataset for LLM recommendation bias analysis"
    )
    parser.add_argument(
        "--yghpc-dir",
        type=Path,
        default=YGHPC_DIR_DEFAULT,
        help=f"Path to yghpc data directory (default: {YGHPC_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Output directory for datasets/ (default: yougov_linked/datasets/)",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=1000,
        help="Number of users to sample from qualifying pool (default: 1000)",
    )
    parser.add_argument(
        "--tweets-per-user",
        type=int,
        default=50,
        help="Exact number of original tweets per user (default: 50)",
    )
    parser.add_argument(
        "--min-tweets-per-user",
        type=int,
        default=None,
        help="Min tweets to qualify; defaults to --tweets-per-user",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for user sampling (default: 42)",
    )
    args = parser.parse_args()

    # min_tweets defaults to tweets_per_user so every user can fill their quota
    min_tweets = args.min_tweets_per_user or args.tweets_per_user

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOUGOV LINKED DATASET PREPARATION")
    print("=" * 70)
    print(f"YGHPC dir:         {args.yghpc_dir}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Users to sample:   {args.n_users:,}")
    print(f"Tweets per user:   {args.tweets_per_user}")
    print(f"Min tweets to qualify: {min_tweets}")
    print(f"Random seed:       {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Crosswalk
    # ------------------------------------------------------------------
    print("Step 1: Loading crosswalk...")
    crosswalk = load_crosswalk(args.yghpc_dir)
    twitter_to_nagler = dict(zip(crosswalk["twitter_id"], crosswalk["nagler_id"]))
    nagler_to_twitter = dict(zip(crosswalk["nagler_id"], crosswalk["twitter_id"]))
    crosswalk_ids = set(crosswalk["twitter_id"])
    print()

    # ------------------------------------------------------------------
    # Step 2: Wave demographics
    # ------------------------------------------------------------------
    print("Step 2: Loading wave demographics...")
    demo_raw = load_wave_demographics(args.yghpc_dir)
    print()

    # ------------------------------------------------------------------
    # Step 3: Map to standard columns
    # ------------------------------------------------------------------
    print("Step 3: Mapping demographics to standard columns...")
    demo = map_demographics(demo_raw)
    print()

    # ------------------------------------------------------------------
    # Step 3b: Filter to users with COMPLETE demographics (no unknowns)
    # ------------------------------------------------------------------
    print("Step 3b: Filtering to users with complete demographics...")
    complete_mask = (
        (demo["author_gender"] != "unknown") &
        (demo["author_political_leaning"] != "unknown") &
        (demo["author_is_minority"] != "unknown")
    )
    demo_complete = demo[complete_mask].copy()
    # Also keep only users who are in the crosswalk
    demo_complete = demo_complete[demo_complete["nagler_id"].isin(set(nagler_to_twitter.keys()))]
    demo_complete["twitter_id"] = demo_complete["nagler_id"].map(nagler_to_twitter)
    complete_twitter_ids = set(demo_complete["twitter_id"])
    print(f"  Users with complete demographics: {len(demo_complete):,}")
    print()

    # ------------------------------------------------------------------
    # Step 4a: Count original tweets for qualifying users (fast pass)
    # ------------------------------------------------------------------
    print("Step 4a: Counting original tweets per user (fast scan, no text)...")
    tweet_counts = count_original_tweets(args.yghpc_dir, complete_twitter_ids)

    # Filter to users meeting the tweet threshold
    qualifying_ids = {uid for uid, cnt in tweet_counts.items() if cnt >= min_tweets}
    print(f"  Users with {min_tweets}+ original tweets: {len(qualifying_ids):,}")

    if len(qualifying_ids) < args.n_users:
        raise ValueError(
            f"Only {len(qualifying_ids):,} users qualify "
            f"(need {args.n_users:,}). Lower --min-tweets-per-user or --n-users."
        )
    print()

    # ------------------------------------------------------------------
    # Step 4b: Sample n_users randomly
    # ------------------------------------------------------------------
    print(f"Step 4b: Sampling {args.n_users:,} users from {len(qualifying_ids):,} qualifying...")
    rng = random.Random(args.seed)
    selected_ids = set(rng.sample(sorted(qualifying_ids), args.n_users))
    print(f"  Selected {len(selected_ids):,} users (seed={args.seed})")
    print()

    # ------------------------------------------------------------------
    # Step 4c: Load tweet content for selected users (full pass)
    # ------------------------------------------------------------------
    print("Step 4c: Loading tweet content for selected users...")
    tweets = load_tweets_for_users(args.yghpc_dir, selected_ids, args.tweets_per_user)
    print()

    # ------------------------------------------------------------------
    # Step 5: Merge tweets + demographics
    # ------------------------------------------------------------------
    print("Step 5: Merging tweets with demographics...")
    tweets["nagler_id"] = tweets["user_id"].map(twitter_to_nagler)
    # Use only the selected users' demographics
    demo_selected = demo_complete[demo_complete["twitter_id"].isin(selected_ids)]
    merged = tweets.merge(demo_selected, on="nagler_id", how="inner")
    print(f"  After merge: {len(merged):,} tweets from {merged['user_id'].nunique():,} users")
    assert merged["user_id"].nunique() == args.n_users, \
        f"Expected {args.n_users} users, got {merged['user_id'].nunique()}"
    print()

    # ------------------------------------------------------------------
    # Step 6: Extract text features via infer_tweet_metadata
    # ------------------------------------------------------------------
    print("Step 6: Extracting text features (VADER sentiment, keyword topics)...")
    print("  This may take several minutes for large datasets...")
    merged_with_meta = infer_tweet_metadata(
        merged,
        text_column="message",
        sentiment_method="vader",
        topic_method="keyword",
        include_gender=False,
        include_political=False,
    )
    print()

    # ------------------------------------------------------------------
    # Step 7: Compute toxicity
    # ------------------------------------------------------------------
    print("Step 7: Computing toxicity features...")
    tox_results = merged_with_meta["message"].fillna("").apply(compute_toxicity)
    merged_with_meta["toxicity"] = tox_results.apply(lambda x: x[0])
    merged_with_meta["severe_toxicity"] = tox_results.apply(lambda x: x[1])
    print(f"  toxicity mean: {merged_with_meta['toxicity'].mean():.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 8: Assemble final datasets
    # ------------------------------------------------------------------
    print("Step 8: Assembling output datasets...")

    # Column sets
    id_cols = ["user_id", "tweet_id"]
    demo_cols = ["author_gender", "author_political_leaning", "author_is_minority"]
    text_feature_cols = [
        "text_length", "avg_word_length",
        "sentiment_polarity", "sentiment_subjectivity",
        "has_emoji", "has_hashtag", "has_mention", "has_url",
        "polarization_score", "controversy_level", "primary_topic",
        "toxicity", "severe_toxicity",
    ]

    # Verify all expected columns present
    missing = [c for c in demo_cols + text_feature_cols if c not in merged_with_meta.columns]
    if missing:
        print(f"  Warning: Missing columns: {missing}")

    # Standardise binary style features to 0/1 integers (consistent with main pipeline)
    for col in ["has_emoji", "has_hashtag", "has_mention", "has_url"]:
        if col in merged_with_meta.columns:
            merged_with_meta[col] = merged_with_meta[col].astype(int)

    # personas.pkl — for run_experiment.py
    personas_df = pd.DataFrame({
        "user_id": merged_with_meta["user_id"],
        "username": merged_with_meta["user_id"],  # no screen_name in parquet after merge
        "message": merged_with_meta["message"],
        "training": 1,
        "reply_to": None,
    })
    # Attach all 16 pre-computed feature columns
    for col in demo_cols + text_feature_cols:
        if col in merged_with_meta.columns:
            personas_df[col] = merged_with_meta[col].values

    personas_path = args.output_dir / "personas.pkl"
    personas_df.to_pickle(personas_path)
    print(f"  Saved personas.pkl: {len(personas_df):,} rows, {len(personas_df.columns)} columns")
    print(f"  Path: {personas_path}")

    personas_csv_path = args.output_dir / "personas.csv"
    personas_df.to_csv(personas_csv_path, index=False)
    print(f"  Saved personas.csv: {personas_csv_path}")

    # analysis_ready.parquet — full feature set
    analysis_cols = id_cols + demo_cols + text_feature_cols + ["message", "created_at"]
    existing_analysis_cols = [c for c in analysis_cols if c in merged_with_meta.columns]
    analysis_df = merged_with_meta[existing_analysis_cols].copy()
    analysis_path = args.output_dir / "analysis_ready.parquet"
    analysis_df.to_parquet(analysis_path, index=False)
    print(f"  Saved analysis_ready.parquet: {len(analysis_df):,} rows")
    print(f"  Path: {analysis_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Total tweets:    {len(personas_df):,}")
    print(f"  Unique users:    {personas_df['user_id'].nunique():,}")
    print(f"  Feature columns: {len(demo_cols + text_feature_cols)}")
    print()
    print("Next step:")
    print("  python run_experiment.py --model openai/gpt-4o-mini")
    print("  python run_all_experiments.py")


if __name__ == "__main__":
    main()
