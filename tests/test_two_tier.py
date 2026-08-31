"""Two-tier selection menu (v2 design): treatments vs evidence-ranked opportunities."""
from agent.brain import parse_method_card_metadata, two_tier_methods_text, method_cards_for_dataset

SAMPLE = """# preamble digest line

### fix-overfit: A treatment
- mechanism: regularize things.
- treats: overfit
- reference_primary: none
- expected_gain / cost: +0.001 / low.
- status_pure: untried
- status_1k: untried

### big-upgrade: An opportunity
- mechanism: better architecture.
- treats: flat-signal
- kind: opportunity
- reference_primary: 0.6055
- expected_gain / cost: +0.003 / medium.
- status_pure: measured-win (somewhere)
- status_1k: untried

### small-upgrade: A weaker opportunity
- mechanism: minor tweak.
- treats: variance
- kind: opportunity
- reference_primary: 0.6040
- expected_gain / cost: +0.001 / low.
- status_pure: untried
- status_1k: untried
"""


def test_kind_parses_and_defaults_to_treatment():
    cards = SAMPLE.split("### ")
    assert parse_method_card_metadata("### " + cards[1])["kind"] == "treatment"
    assert parse_method_card_metadata("### " + cards[2])["kind"] == "opportunity"


def test_two_tier_sections_and_ranking():
    text = two_tier_methods_text(method_cards_for_dataset(SAMPLE, "pure"))
    assert "TREATMENT CARDS" in text and "OPPORTUNITY CARDS" in text
    # treatment stays in its section, before the opportunity header
    assert text.index("fix-overfit") < text.index("OPPORTUNITY CARDS")
    # opportunities after the header, measured-win ranked first
    opp = text[text.index("OPPORTUNITY CARDS"):]
    assert opp.index("big-upgrade") < opp.index("small-upgrade")
    # preamble preserved
    assert text.startswith("# preamble digest line")


def test_opportunities_visible_regardless_of_diagnosis_gating():
    # structural guarantee: the opportunity section exists independent of any
    # diagnosis; a selector reading this menu always sees measured-win upgrades.
    text = two_tier_methods_text(method_cards_for_dataset(SAMPLE, "1k"))
    assert "big-upgrade" in text.split("OPPORTUNITY CARDS")[1]
