"""
NP-MRD id key handling: MARINA1 metadata stores it under 'npmrd_id', MARINA-DB
under 'npid'. Both must resolve to a name + link, or NP-MRD-only entries (e.g.
Kavaratamide A) show up unnamed with no database link.
"""
from app.result_builder import _primary, _database_links


def test_npmrd_id_key_marina1():
    entry = {"npmrd": {"npmrd_id": "NP0332591", "name": "Kavaratamide A"}}
    name, link = _primary(entry)
    assert name == "Kavaratamide A"
    assert link == "https://np-mrd.org/natural_products/NP0332591"
    assert _database_links(entry)["npmrd"].endswith("NP0332591")


def test_npid_key_marina_db():
    entry = {"npmrd": {"npid": "NP0332591", "name": "Kavaratamide A"}, "coconut": None, "lotus": None}
    name, link = _primary(entry)
    assert name == "Kavaratamide A"
    assert link == "https://np-mrd.org/natural_products/NP0332591"
    assert _database_links(entry)["npmrd"].endswith("NP0332591")


def test_coconut_still_takes_precedence_but_npmrd_link_present():
    entry = {
        "coconut": {"coconut_id": "CNP0561556.2", "name": "Some Name"},
        "npmrd": {"npid": "NP0332333", "name": "alt"},
    }
    name, link = _primary(entry)
    assert name == "Some Name"                       # coconut is primary
    assert "coconut" in link
    assert _database_links(entry)["npmrd"].endswith("NP0332333")   # npmrd link still surfaced
