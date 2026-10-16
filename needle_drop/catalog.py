"""Build and load the track catalog from the MTG-Jamendo dataset metadata.

Sources (all fetched from github.com/MTG/mtg-jamendo-dataset by scripts/fetch_data.sh):
  raw.meta.tsv           track/artist/album names and Jamendo URLs
  raw_30s_cleantags.tsv  genre / instrument / mood tags per track
  audio_licenses.txt     per-track Creative Commons license, as published with the dataset

Only tracks that have at least one mood/theme tag are kept: they are the ones we can
search "by vibe". Every track keeps the license line it shipped with, so licensing is
never guessed (see licensing.py).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config

LICENSE_SOURCE = "https://github.com/MTG/mtg-jamendo-dataset/blob/master/audio_licenses.txt"

# Rough energy prior per mood tag, used for tempo/energy matching (0 calm .. 1 intense).
MOOD_ENERGY = {
    "energetic": .9, "action": .9, "fast": .95, "party": .85, "sport": .85, "powerful": .85,
    "heavy": .9, "upbeat": .8, "fun": .75, "funny": .7, "groovy": .7, "epic": .8, "trailer": .8,
    "adventure": .75, "motivational": .7, "uplifting": .65, "happy": .7, "positive": .65,
    "summer": .65, "commercial": .6, "advertising": .6, "corporate": .55, "inspiring": .6,
    "cool": .55, "retro": .55, "sexy": .5, "game": .65, "dramatic": .65, "drama": .55,
    "horror": .5, "dark": .5, "holiday": .55, "christmas": .55, "children": .55, "travel": .55,
    "film": .5, "movie": .5, "documentary": .4, "background": .35, "melodic": .45, "love": .4,
    "romantic": .35, "emotional": .4, "hopeful": .45, "deep": .4, "space": .35, "nature": .3,
    "ballad": .3, "melancholic": .3, "sad": .25, "dream": .3, "soundscape": .3, "ambiental": .2,
    "mellow": .3, "soft": .2, "slow": .15, "calm": .15, "relaxing": .15, "meditative": .1,
}


@dataclass
class Track:
    id: str                 # e.g. "jamendo:214"
    jamendo_id: int
    title: str
    artist: str
    album: str
    release_date: str
    page_url: str
    audio_url: str
    duration: float
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    energy: float = 0.5
    license_name: str = ""
    license_url: str = ""
    license_source: str = ""

    def describe(self) -> str:
        """Natural-language description used for embedding (the 'vibe document')."""
        parts = []
        if self.moods:
            parts.append(", ".join(self.moods) + " mood")
        if self.genres:
            parts.append(" / ".join(self.genres) + " music")
        if self.instruments:
            parts.append("featuring " + ", ".join(self.instruments))
        e = self.energy
        parts.append("high energy, fast" if e >= .7 else "low energy, slow" if e <= .3 else "mid tempo")
        return f"{self.title} by {self.artist}. " + "; ".join(parts) + "."

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_licenses(path: Path) -> dict[int, tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[int, tuple[str, str]] = {}
    for block in text.split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines()]
        if len(lines) < 3:
            continue
        m = re.search(r"jamendo\.com/track/(\d+)", lines[1])
        lic = re.match(r"Available under a (.+?) license: (\S+)", lines[2])
        if m and lic:
            out[int(m.group(1))] = (lic.group(1), lic.group(2))
    return out


def build_catalog(raw_dir: Path = config.DATA / "raw", out: Path = config.CATALOG_PATH) -> int:
    meta = {}
    with open(raw_dir / "raw.meta.tsv", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            if len(row) >= 8:
                meta[row[0]] = row
    licenses = _parse_licenses(raw_dir / "audio_licenses.txt")
    n = 0
    with open(raw_dir / "raw_30s_cleantags.tsv", encoding="utf-8") as f, open(out, "w") as w:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            tid, duration, tags = row[0], float(row[4]), row[5:]
            g = [t.split("---")[1] for t in tags if t.startswith("genre---")]
            m = [t.split("---")[1] for t in tags if t.startswith("mood/theme---")]
            i = [t.split("---")[1] for t in tags if t.startswith("instrument---")]
            if not m or tid not in meta or duration < 30:
                continue
            jid = int(tid.split("_")[1])
            lic = licenses.get(jid, ("", ""))
            md = meta[tid]
            energies = [MOOD_ENERGY[x] for x in m if x in MOOD_ENERGY]
            t = Track(
                id=f"jamendo:{jid}", jamendo_id=jid, title=md[3], artist=md[4], album=md[5],
                release_date=md[6], page_url=f"https://www.jamendo.com/track/{jid}",
                audio_url=f"https://prod-1.storage.jamendo.com/?trackid={jid}&format=mp31",
                duration=duration, genres=g, moods=m, instruments=i,
                energy=round(sum(energies) / len(energies), 3) if energies else 0.5,
                license_name=lic[0], license_url=lic[1],
                license_source=LICENSE_SOURCE if lic[0] else "",
            )
            w.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


@lru_cache(maxsize=1)
def load_catalog(path: str = str(config.CATALOG_PATH)) -> list[Track]:
    with open(path, encoding="utf-8") as f:
        return [Track(**json.loads(l)) for l in f if l.strip()]


@lru_cache(maxsize=1)
def tag_vocab() -> dict[str, list[str]]:
    g, m, i = set(), set(), set()
    for t in load_catalog():
        g.update(t.genres); m.update(t.moods); i.update(t.instruments)
    return {"genres": sorted(g), "moods": sorted(m), "instruments": sorted(i)}


if __name__ == "__main__":
    print(build_catalog(), "tracks written to", config.CATALOG_PATH)
