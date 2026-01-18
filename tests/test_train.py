import pytest

import diabetic_classification.train as train_module


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("age", "age"),
        ("age,bmi", ["age", "bmi"]),
        (" age , bmi ", ["age", "bmi"]),
    ],
)
def test_parse_attributes(value: str | None, expected: list[str] | str | None) -> None:
    assert train_module._parse_attributes(value) == expected, "Parsed attributes should match expected output."


def test_parse_attributes_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="Attribute list cannot be empty."):
        train_module._parse_attributes(" , ")
