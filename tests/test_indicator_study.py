"""Every stored indicator has a study lesson the page can render."""

from engine.catalog import load_catalog
from engine.indicator_study import categories, group_intro, lessons


def test_every_catalog_indicator_has_a_study_lesson():
    items = load_catalog()
    records = lessons()
    assert [record.id for record in records] == [item["id"] for item in items]
    assert len(records) == len({record.id for record in records})
    listed = set(categories())
    for item, record in zip(items, records):
        assert record.name == item["name"]
        assert record.category == item["category"]
        assert record.category in listed
        assert group_intro(record.category)
        assert record.formula == item["formula"]
        assert record.what.strip()
        assert len(record.how) > 80
        assert len(record.why) > 80
        assert record.inputs == tuple(item["inputs"])
        assert record.outputs == tuple(item["outputs"])
        assert record.input_text
        assert record.param_text

    by_id = {record.id: record for record in records}
    assert by_id["atr"].calculated and by_id["atr"].code_id == "atr"
    assert by_id["bbands"].calculated and by_id["bbands"].code_id == "bollinger"
    assert by_id["stoch"].code_id == "stochastic"
    assert not by_id["greeks"].calculated
    assert by_id["greeks"].limit
    assert not by_id["gann_fan"].calculated
    assert "Renko" not in by_id["gann_fan"].limit
