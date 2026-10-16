"""Guardrail and licensing rules (Phase 8). These run without the LLM or the embedding model."""
from needle_drop import guardrails
from needle_drop.catalog import Track
from needle_drop.licensing import RULES, check_license, parse_license_url


def track(url="http://creativecommons.org/licenses/by/3.0/", source="https://example/audio_licenses.txt"):
    return Track(id="jamendo:1", jamendo_id=1, title="T", artist="A", album="", release_date="", page_url="p",
                 audio_url="a", duration=60, genres=["rock"], moods=["happy"], license_name="CC",
                 license_url=url, license_source=source)


def test_parse_license_urls():
    assert parse_license_url("http://creativecommons.org/licenses/by-nc-sa/2.0/fr") == ("cc-by-nc-sa", "2.0")
    assert parse_license_url("https://creativecommons.org/licenses/by-nd/4.0") == ("cc-by-nd", "4.0")
    assert parse_license_url("http://example.com/license") == (None, None)
    assert parse_license_url("") == (None, None)


def test_nd_never_cleared_for_video(monkeypatch):
    monkeypatch.setattr("needle_drop.licensing.cite", lambda *a: [])
    for use in ("personal", "monetized", "client"):
        v = check_license(track("http://creativecommons.org/licenses/by-nd/3.0/"), use, live=False)
        assert v.status == "not_cleared" and not v.cleared


def test_nc_not_cleared_for_monetized(monkeypatch):
    monkeypatch.setattr("needle_drop.licensing.cite", lambda *a: [])
    assert check_license(track("http://creativecommons.org/licenses/by-nc/3.0/"), "monetized", live=False).status == "not_cleared"
    assert check_license(track("http://creativecommons.org/licenses/by-nc/3.0/"), "personal", live=False).status == "cleared"


def test_missing_source_is_unverified(monkeypatch):
    monkeypatch.setattr("needle_drop.licensing.cite", lambda *a: [])
    v = check_license(track(source=""), "personal", live=False)
    assert v.status == "unverified" and not v.cleared and v.cost["amount_usd"] is None
    v = check_license(track(url="all rights reserved"), "personal", live=False)
    assert v.status == "unverified" and not v.cleared


def test_cleared_carries_snapshot_flag(monkeypatch):
    monkeypatch.setattr("needle_drop.licensing.cite", lambda *a: [])
    v = check_license(track(), "monetized", live=False)
    assert v.cleared and v.cost["amount_usd"] == 0.0
    assert any("snapshot" in f for f in v.flags)


def test_every_rule_covers_every_use():
    for lid, r in RULES.items():
        assert set(r) == {"personal", "monetized", "client"}, lid


def _item(status, verification="dataset_snapshot", source="s", blurb="", score=0.8):
    return {"score": score, "blurb": blurb,
            "license": {"status": status, "verification": verification, "source": source, "flags": [], "reason": "r"}}


def test_llm_cannot_claim_clearance():
    it = _item("not_cleared", blurb="Great driving energy. This track is royalty-free and cleared for YouTube.")
    guardrails.apply([it])
    assert not it["cleared"]
    assert "royalty" not in it["blurb"].lower() and "cleared" not in it["blurb"].lower()
    assert any("Removed an unsupported licensing claim" in f for f in it["flags"])


def test_cleared_requires_source():
    it = _item("cleared", source="")
    guardrails.apply([it])
    assert not it["cleared"]
    it = _item("cleared", verification="none")
    guardrails.apply([it])
    assert not it["cleared"]


def test_unverified_and_weak_matches_flagged():
    a, b = _item("unverified"), _item("cleared", score=0.2)
    warnings = guardrails.apply([a, b])
    assert a["needs_review"] and any("No verified license" in f for f in a["flags"])
    assert any("Weak match" in f for f in b["flags"])
    assert warnings == []
    assert guardrails.apply([_item("not_cleared")]) == ["No track in this shortlist is cleared for the selected use."]
