"""Language of each TalkingHead-1KH source video, with faster-whisper (Section 4.4.2).

Clips are grouped by source video (from the file name), on the assumption that a
video has a single speaker and language. data/language_ids_final.csv, used by
scripts/checks/check_english_subset.py, comes from four passes, each resumable;
intermediate files go to results/language_ids/:

    python -m scripts.dataset.detect_language
        language detection on one clip per video (the largest file);
    python -m scripts.dataset.detect_language --verify-low-confidence
        a second clip for the videos whose probability is below 0.9;
    python -m scripts.dataset.detect_language --exhaustive --only-disagreed
        every clip of the videos where the two clips disagree, aggregated by
        majority vote;
    python -m scripts.dataset.detect_language --rebuild-final-labels
        one label per video: the majority vote where it exists, the first pass
        otherwise; videos whose winning language has less than 50% of the votes
        are marked is_ambiguous.

Needs the "transcription" dependency group.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel, decode_audio
from tqdm import tqdm

from utils.protocol_utils import LANGUAGE_IDS_PATH, RESULTS_DIR, WAV_DIR

WORK_DIR = RESULTS_DIR / "language_ids"  # intermediate files of the four passes
DEFAULT_WAV_DIR = WAV_DIR
DEFAULT_BY_VIDEO = WORK_DIR / "language_ids_by_video.csv"
DEFAULT_PER_CLIP = WORK_DIR / "language_ids.csv"
DEFAULT_VERIFY_OUT = WORK_DIR / "language_ids_verification.csv"
DEFAULT_FULL_PER_CLIP = WORK_DIR / "language_ids_per_clip_full.csv"
DEFAULT_AGGREGATED = WORK_DIR / "language_ids_aggregated.csv"
DEFAULT_FINAL_BY_VIDEO = LANGUAGE_IDS_PATH
DEFAULT_FINAL_PER_CLIP = WORK_DIR / "language_ids_final_per_clip.csv"
SAMPLE_RATE = 16000
MAX_SECONDS = 30

CLIP_SUFFIX_RE = re.compile(r"_S\d+_E\d+_L\d+_T\d+_R\d+_B\d+$")
CLIP_IDX_RE = re.compile(r"_\d{4}$")


def video_id_from_name(name: str) -> str:
    stem = name[:-4] if name.endswith(".wav") else name
    stem = CLIP_SUFFIX_RE.sub("", stem)
    stem = CLIP_IDX_RE.sub("", stem)
    return stem


def detect_lid(model: WhisperModel, wav_path: Path) -> tuple[str, float, list]:
    """Returns (language, probability, top3) for one wav. Empty/short → ('', 0.0, [])."""
    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    if len(audio) > SAMPLE_RATE * MAX_SECONDS:
        audio = audio[: SAMPLE_RATE * MAX_SECONDS]
    if len(audio) < SAMPLE_RATE:
        return "", 0.0, []
    language, prob, all_probs = model.detect_language(audio=audio.astype(np.float32))
    items = all_probs.items() if isinstance(all_probs, dict) else all_probs
    top3 = sorted(items, key=lambda kv: -kv[1])[:3]
    return language, prob, top3


def load_done_videos(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open() as f:
        return {row["video_id"]: row for row in csv.DictReader(f)}


def load_done_clips(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {row["filename"] for row in csv.DictReader(f)}


def load_disagreed_videos(verify_csv: Path) -> set[str]:
    if not verify_csv.exists():
        return set()
    with verify_csv.open() as f:
        return {row["video_id"] for row in csv.DictReader(f) if row["agree"] == "0"}


def aggregate_per_video(
    per_clip_csv: Path,
    groups: dict[str, list[Path]],
    aggregated_csv: Path,
    only_videos: set[str] | None = None,
) -> None:
    """Read the full per-clip CSV and write one row per video summarizing the votes."""
    per_video: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with per_clip_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                prob = float(row["probability"])
            except (TypeError, ValueError):
                prob = 0.0
            per_video[row["video_id"]].append((row["language"], prob))

    with aggregated_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "n_clips", "winning_language", "winning_count",
            "agreement_rate", "winning_mean_prob",
            "runner_up_language", "runner_up_count", "language_distribution",
        ])
        for vid in sorted(groups):
            if only_videos is not None and vid not in only_videos:
                continue
            votes = per_video.get(vid, [])
            if not votes:
                continue
            counts = Counter(lang for lang, _ in votes if lang)
            if not counts:
                writer.writerow([vid, len(votes), "", 0, 0.0, 0.0, "", 0, "{}"])
                continue
            ranked = counts.most_common()
            winner, win_count = ranked[0]
            runner = ranked[1] if len(ranked) > 1 else ("", 0)
            winner_probs = [p for lang, p in votes if lang == winner]
            mean_prob = sum(winner_probs) / len(winner_probs) if winner_probs else 0.0
            writer.writerow([
                vid, len(votes), winner, win_count,
                f"{win_count/len(votes):.4f}", f"{mean_prob:.4f}",
                runner[0], runner[1], json.dumps(dict(counts)),
            ])


def rebuild_final_labels(args, groups: dict[str, list[Path]]) -> int:
    """Merge rep-only by-video labels with exhaustive aggregation.

    Rules:
      - If a video appears in the aggregated CSV, its label comes from the
        majority vote over all its clips (source='exhaustive').
      - Otherwise the label comes from the rep-only by-video CSV (source='rep').
      - Exhaustive labels with agreement_rate < threshold are flagged
        is_ambiguous=1; consumers should drop or hand-inspect those.
    """
    if not args.by_video_out.exists():
        print(f"Missing {args.by_video_out}; nothing to rebuild from.", file=sys.stderr)
        return 1

    rep = load_done_videos(args.by_video_out)
    aggregated: dict[str, dict] = {}
    if args.aggregated_out.exists():
        with args.aggregated_out.open() as f:
            for row in csv.DictReader(f):
                aggregated[row["video_id"]] = row

    final_rows: list[dict] = []
    for vid in sorted(groups):
        if vid in aggregated and aggregated[vid].get("winning_language"):
            agg = aggregated[vid]
            try:
                agreement = float(agg["agreement_rate"])
            except (TypeError, ValueError):
                agreement = 0.0
            try:
                conf = float(agg["winning_mean_prob"])
            except (TypeError, ValueError):
                conf = 0.0
            final_rows.append({
                "video_id": vid,
                "language": agg["winning_language"],
                "source": "exhaustive",
                "confidence": f"{conf:.4f}",
                "agreement_rate": f"{agreement:.4f}",
                "n_clips": agg["n_clips"],
                "runner_up_language": agg.get("runner_up_language", ""),
                "is_ambiguous": int(agreement < args.ambiguous_agreement_threshold),
            })
        elif vid in rep:
            r = rep[vid]
            final_rows.append({
                "video_id": vid,
                "language": r["language"],
                "source": "rep",
                "confidence": r["probability"],
                "agreement_rate": "",
                "n_clips": len(groups[vid]),
                "runner_up_language": "",
                "is_ambiguous": 0,
            })

    fieldnames = [
        "video_id", "language", "source", "confidence",
        "agreement_rate", "n_clips", "runner_up_language", "is_ambiguous",
    ]
    with args.final_by_video_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_rows)

    by_vid_final = {r["video_id"]: r for r in final_rows}
    with args.final_per_clip_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "video_id", "language", "source", "confidence", "is_ambiguous"])
        for vid in sorted(groups):
            row = by_vid_final.get(vid)
            if not row:
                continue
            for p in sorted(groups[vid]):
                w.writerow([
                    p.name, vid, row["language"], row["source"],
                    row["confidence"], row["is_ambiguous"],
                ])

    n_exh = sum(1 for r in final_rows if r["source"] == "exhaustive")
    n_rep = sum(1 for r in final_rows if r["source"] == "rep")
    n_amb = sum(1 for r in final_rows if r["is_ambiguous"])
    print(
        f"Final labels: {len(final_rows)} videos "
        f"({n_exh} from exhaustive, {n_rep} from rep). "
        f"{n_amb} flagged is_ambiguous."
    )
    print(f"Wrote {args.final_by_video_out} and {args.final_per_clip_out}")
    return 0


def run_exhaustive(args, wavs: list[Path], groups: dict[str, list[Path]]) -> int:
    """Run LID on every wav (resumable), then aggregate per source video."""
    only_videos: set[str] | None = None
    if args.only_disagreed:
        only_videos = load_disagreed_videos(args.verify_out)
        if not only_videos:
            print(
                f"--only-disagreed set, but no disagreements found in {args.verify_out}",
                file=sys.stderr,
            )
            return 1
        wavs = [w for w in wavs if video_id_from_name(w.name) in only_videos]
        print(f"Filtered to {len(only_videos)} disagreed videos → {len(wavs)} clips")

    done = load_done_clips(args.full_per_clip_out)
    todo = [w for w in wavs if w.name not in done]
    print(
        f"Clips: {len(wavs)} | already done: {len(done)} | to process: {len(todo)}"
    )

    if todo:
        print(f"Loading model '{args.model}' on {args.device} ({args.compute_type})...")
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

        write_header = not args.full_per_clip_out.exists()
        with args.full_per_clip_out.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["filename", "video_id", "language", "probability", "top3"])
                f.flush()

            for wav_path in tqdm(todo, desc="LID-full"):
                vid = video_id_from_name(wav_path.name)
                try:
                    lang, prob, top3 = detect_lid(model, wav_path)
                    writer.writerow(
                        [wav_path.name, vid, lang, f"{prob:.4f}", json.dumps(top3)]
                    )
                    f.flush()
                except Exception as e:
                    print(f"\n[warn] {wav_path.name}: {e}", file=sys.stderr)
                    writer.writerow([wav_path.name, vid, "ERROR", 0.0, json.dumps(str(e))])
                    f.flush()

    aggregate_per_video(
        args.full_per_clip_out, groups, args.aggregated_out, only_videos=only_videos
    )
    print(f"Wrote per-video aggregation to {args.aggregated_out}")
    return 0


def run_verification(
    args, groups: dict[str, list[Path]], reps: dict[str, Path], done: dict[str, dict]
) -> int:
    """Re-run LID on a second clip for low-confidence videos."""
    candidates: list[tuple[str, Path, dict]] = []
    skipped_singletons = 0
    for vid, row in done.items():
        try:
            rep_prob = float(row["probability"])
        except (TypeError, ValueError):
            rep_prob = 0.0
        if rep_prob >= args.verify_threshold:
            continue
        clips = groups.get(vid, [])
        rep_name = reps[vid].name if vid in reps else row["representative_wav"]
        others = [p for p in clips if p.name != rep_name]
        if not others:
            skipped_singletons += 1
            continue
        # Second-largest file: another long sample, but distinct from the rep.
        second = max(others, key=lambda p: p.stat().st_size)
        candidates.append((vid, second, row))

    print(
        f"Low-confidence videos (prob < {args.verify_threshold}): "
        f"{sum(1 for r in done.values() if float(r['probability'] or 0) < args.verify_threshold)} | "
        f"verifiable (have ≥2 clips): {len(candidates)} | "
        f"skipped singletons: {skipped_singletons}"
    )
    if not candidates:
        return 0

    print(f"Loading model '{args.model}' on {args.device} ({args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    write_header = not args.verify_out.exists()
    with args.verify_out.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "video_id", "rep_wav", "rep_language", "rep_probability",
                "verify_wav", "verify_language", "verify_probability",
                "agree", "verify_top3",
            ])
            f.flush()

        agree_n = 0
        for vid, vpath, row in tqdm(candidates, desc="verify"):
            try:
                lang, prob, top3 = detect_lid(model, vpath)
                agree = int(lang == row["language"] and lang != "")
                agree_n += agree
                writer.writerow([
                    vid, row["representative_wav"], row["language"], row["probability"],
                    vpath.name, lang, f"{prob:.4f}", agree, json.dumps(top3),
                ])
                f.flush()
            except Exception as e:
                print(f"\n[warn] {vid} ({vpath.name}): {e}", file=sys.stderr)
                writer.writerow([
                    vid, row["representative_wav"], row["language"], row["probability"],
                    vpath.name, "ERROR", 0.0, 0, json.dumps(str(e)),
                ])
                f.flush()

    print(
        f"Agreement: {agree_n}/{len(candidates)} "
        f"({100*agree_n/len(candidates):.1f}%); disagreements written to {args.verify_out}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    ap.add_argument("--by-video-out", type=Path, default=DEFAULT_BY_VIDEO)
    ap.add_argument("--per-clip-out", type=Path, default=DEFAULT_PER_CLIP)
    ap.add_argument("--model", default="small")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--verify-low-confidence",
        action="store_true",
        help="Re-run LID on a second clip for videos whose representative "
             "probability is below --verify-threshold. Skips the main pass.",
    )
    ap.add_argument("--verify-threshold", type=float, default=0.9)
    ap.add_argument("--verify-out", type=Path, default=DEFAULT_VERIFY_OUT)
    ap.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run LID on every clip (not just one representative per video) "
             "and aggregate per video. Resumable per clip.",
    )
    ap.add_argument("--full-per-clip-out", type=Path, default=DEFAULT_FULL_PER_CLIP)
    ap.add_argument("--aggregated-out", type=Path, default=DEFAULT_AGGREGATED)
    ap.add_argument(
        "--only-disagreed",
        action="store_true",
        help="With --exhaustive, restrict to videos whose verification row has "
             "agree=0 in --verify-out (read-only).",
    )
    ap.add_argument(
        "--rebuild-final-labels",
        action="store_true",
        help="Merge by-video labels with exhaustive aggregation to produce a "
             "canonical language label per video and per clip.",
    )
    ap.add_argument("--final-by-video-out", type=Path, default=DEFAULT_FINAL_BY_VIDEO)
    ap.add_argument("--final-per-clip-out", type=Path, default=DEFAULT_FINAL_PER_CLIP)
    ap.add_argument(
        "--ambiguous-agreement-threshold",
        type=float,
        default=0.5,
        help="Exhaustive videos with winning agreement_rate below this are flagged is_ambiguous=1.",
    )
    args = ap.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.wav_dir.glob("*.wav"))
    if not wavs:
        print(f"No wavs found in {args.wav_dir}", file=sys.stderr)
        return 1

    groups: dict[str, list[Path]] = defaultdict(list)
    for w in wavs:
        groups[video_id_from_name(w.name)].append(w)

    # Representative = largest file in the group (proxy for longest speech).
    reps: dict[str, Path] = {
        vid: max(paths, key=lambda p: p.stat().st_size)
        for vid, paths in groups.items()
    }

    done = load_done_videos(args.by_video_out)

    if args.rebuild_final_labels:
        return rebuild_final_labels(args, groups)

    if args.exhaustive:
        return run_exhaustive(args, wavs, groups)

    if args.verify_low_confidence:
        if not done:
            print(
                f"No by-video CSV at {args.by_video_out}; run the main pass first.",
                file=sys.stderr,
            )
            return 1
        return run_verification(args, groups, reps, done)

    todo = [(vid, reps[vid]) for vid in sorted(groups) if vid not in done]
    print(
        f"Clips: {len(wavs)} | videos: {len(groups)} | "
        f"already done: {len(done)} | to process: {len(todo)}"
    )

    if todo:
        print(f"Loading model '{args.model}' on {args.device} ({args.compute_type})...")
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

        write_header = not args.by_video_out.exists()
        with args.by_video_out.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["video_id", "representative_wav", "language", "probability", "top3"]
                )
                f.flush()

            for vid, rep_path in tqdm(todo, desc="LID"):
                try:
                    language, prob, top3 = detect_lid(model, rep_path)
                    writer.writerow(
                        [vid, rep_path.name, language, f"{prob:.4f}", json.dumps(top3)]
                    )
                    f.flush()
                except Exception as e:
                    print(f"\n[warn] {vid} ({rep_path.name}): {e}", file=sys.stderr)
                    writer.writerow([vid, rep_path.name, "ERROR", 0.0, json.dumps(str(e))])
                    f.flush()

    # Propagate from by-video CSV to per-clip CSV (always rebuild; cheap).
    by_video = load_done_videos(args.by_video_out)
    with args.per_clip_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "video_id", "language", "probability", "top3", "is_representative"]
        )
        for vid in sorted(groups):
            row = by_video.get(vid)
            if not row:
                continue
            rep_name = row["representative_wav"]
            for p in sorted(groups[vid]):
                writer.writerow([
                    p.name, vid, row["language"], row["probability"],
                    row["top3"], int(p.name == rep_name),
                ])
    print(f"Wrote per-clip labels to {args.per_clip_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
