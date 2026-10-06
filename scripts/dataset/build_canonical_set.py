"""Build data/manifest_canonical.csv: one clip per source video.

For each source YouTube video, keep the alphabetically first clip among those
with a DiffPoseTalk inference, and resolve its audio, SMIRK reconstruction,
ARTalk inference and DiffPoseTalk style file. The clips with a DiffPoseTalk
inference are the ones that passed the audio and tracking filters described in
Section 4.4.2 of the thesis; that filtering is not part of this repository, and
the published manifest fixes the resulting selection.

With --move-extra-samples, extra DiffPoseTalk samples of the same clip
(files ending in _NNN.npz) are moved to a subdirectory first, so that they are
not taken as separate clips.
"""

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

from utils.protocol_utils import ARTALK_DIR, DATA_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, WAV_DIR

STYLE_DIR = DATA_DIR / "TH1KH" / "style"
EXTRA_SAMPLES_DIR = DPT_DIR / "_extra_samples"
DEFAULT_MANIFEST = MANIFEST_PATH
ARTALK_TRAILING_RE = re.compile(r"_natural_\d+_mesh_motions$")

CLIP_SUFFIX_RE = re.compile(r"_S\d+_E\d+_L\d+_T\d+_R\d+_B\d+")
CLIP_IDX_RE = re.compile(r"_\d{4}$")
DPT_SAMPLE_SUFFIX_RE = re.compile(r"_\d{3}$")        # _000, _001, ...
STYLE_WINDOW_SUFFIX_RE = re.compile(r"_s\d+$")        # _s000000, _s000100, ...


def video_id_from_clip_stem(stem: str) -> str:
    s = CLIP_SUFFIX_RE.sub("", stem)
    s = CLIP_IDX_RE.sub("", s)
    return s


def move_extra_samples() -> int:
    """Move extra DiffPoseTalk samples (*_NNN.npz) to DPT_DIR/_extra_samples."""
    extras = [
        p for p in DPT_DIR.glob("*.npz")
        if DPT_SAMPLE_SUFFIX_RE.search(p.stem)
    ]
    if not extras:
        print(f"No extra-sample files found in {DPT_DIR}.")
        return 0
    EXTRA_SAMPLES_DIR.mkdir(exist_ok=True)
    for p in extras:
        shutil.move(str(p), str(EXTRA_SAMPLES_DIR / p.name))
    print(f"Moved {len(extras)} extra-sample files to {EXTRA_SAMPLES_DIR}")
    return len(extras)


def index_styles() -> dict[str, str]:
    """Return clip_stem -> chosen style filename (prefers `_s000000`)."""
    by_clip: dict[str, list[Path]] = defaultdict(list)
    for p in STYLE_DIR.glob("*.npy"):
        m = STYLE_WINDOW_SUFFIX_RE.search(p.stem)
        clip_stem = p.stem[: m.start()] if m else p.stem
        by_clip[clip_stem].append(p)
    chosen: dict[str, str] = {}
    for clip_stem, paths in by_clip.items():
        # prefer _s000000, else the earliest window numerically.
        def window_key(p: Path) -> int:
            m = STYLE_WINDOW_SUFFIX_RE.search(p.stem)
            return int(m.group()[2:]) if m else 0
        chosen[clip_stem] = min(paths, key=window_key).name
    return chosen


def index_wavs() -> dict[str, str]:
    """Return clip_stem -> wav filename."""
    return {p.stem: p.name for p in WAV_DIR.glob("*.wav")}


def index_gt_npz() -> dict[str, str]:
    """Return clip_stem -> SMIRK ground-truth FLAME npz filename."""
    return {p.stem: p.name for p in GT_NPZ_DIR.glob("*.npz")}


def index_artalk() -> dict[str, str]:
    """Return clip_stem -> ARTalk inference filename.

    ARTalk filenames are `{clip_stem}_natural_{N}_mesh_motions.pt`.
    """
    out: dict[str, str] = {}
    for p in ARTALK_DIR.glob("*.pt"):
        stem = ARTALK_TRAILING_RE.sub("", p.stem)
        out[stem] = p.name
    return out


def build_manifest(out_path: Path) -> None:
    style_index = index_styles()
    wav_index = index_wavs()
    gt_index = index_gt_npz()
    artalk_index = index_artalk()

    by_video: dict[str, list[Path]] = defaultdict(list)
    for p in DPT_DIR.glob("*.npz"):
        by_video[video_id_from_clip_stem(p.stem)].append(p)

    rows = []
    missing_wav = []
    missing_gt = []
    missing_artalk = []
    for vid in sorted(by_video):
        chosen = sorted(by_video[vid], key=lambda p: p.name)[0]
        clip_stem = chosen.stem
        wav = wav_index.get(clip_stem, "")
        gt = gt_index.get(clip_stem, "")
        style = style_index.get(clip_stem, "")
        artalk = artalk_index.get(clip_stem, "")
        rows.append({
            "video_id": vid,
            "clip_stem": clip_stem,
            "wav": wav,
            "gt_npz": gt,
            "dpt_inference": chosen.name,
            "artalk_inference": artalk,
            "style": style,
        })
        if not wav:
            missing_wav.append(clip_stem)
        if not gt:
            missing_gt.append(clip_stem)
        if not artalk:
            missing_artalk.append(clip_stem)

    fields = [
        "video_id", "clip_stem", "wav", "gt_npz",
        "dpt_inference", "artalk_inference", "style",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_with_style = sum(1 for r in rows if r["style"])
    n_with_wav = sum(1 for r in rows if r["wav"])
    n_with_gt = sum(1 for r in rows if r["gt_npz"])
    n_with_artalk = sum(1 for r in rows if r["artalk_inference"])
    print(f"Wrote {out_path}")
    print(f"  Total videos (rows):           {len(rows)}")
    print(f"  Rows with wav resolved:        {n_with_wav}")
    print(f"  Rows with GT npz resolved:     {n_with_gt}")
    print(f"  Rows with ARTalk resolved:     {n_with_artalk}")
    print(f"  Rows with style resolved:      {n_with_style}")
    print(f"  Rows with NO style (default):  {len(rows) - n_with_style}")
    if missing_wav:
        print(f"  WARNING: {len(missing_wav)} canonical clips have no matching wav.")
        print(f"  First 3: {missing_wav[:3]}")
    if missing_gt:
        print(f"  WARNING: {len(missing_gt)} canonical clips have no matching GT npz.")
        print(f"  First 3: {missing_gt[:3]}")
    if missing_artalk:
        print(f"  WARNING: {len(missing_artalk)} canonical clips have no matching ARTalk inference.")
        print(f"  First 3: {missing_artalk[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--move-extra-samples", action="store_true",
        help="Move the *_NNN.npz extra DiffPoseTalk samples out of the inference directory first.",
    )
    args = ap.parse_args()

    if args.move_extra_samples:
        move_extra_samples()
    build_manifest(args.manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
