from __future__ import annotations


def test_temperature_parser_handles_degree_word_units_without_bare_degree_guessing():
    from strap.user_input_parsing import extract_temperatures_c

    cases = {
        "below 212 degrees Fahrenheit": 100.0,
        "below 212 degree Fahrenheit": 100.0,
        "below 212 degrees F": 100.0,
        "under 373.15 degrees Kelvin": 100.0,
        "under 373.15 degree Kelvin": 100.0,
        "below 100 degrees Celsius": 100.0,
    }
    for text, expected_c in cases.items():
        values = extract_temperatures_c(text)
        assert len(values) == 1, text
        assert abs(values[0] - expected_c) < 1e-6

    assert extract_temperatures_c("below 100 degrees") == []
    assert extract_temperatures_c("use 100 deg") == []


def test_output_destination_does_not_absorb_following_prose():
    from strap.user_input_parsing import extract_output_destination

    destination = extract_output_destination(
        "save to /home/aaltamimi2/plots\nOnly include separation visuals."
    )

    assert destination is not None
    assert destination.output_dir == "/home/aaltamimi2/plots"


def test_output_destination_repairs_clear_wrapped_path_fragments():
    from strap.user_input_parsing import extract_output_destination

    destination = extract_output_destination(
        "save to /home/aaltamimi2/case-studies/case-1/01-ldpe-evoh-p\n"
        "  et\n"
        "    -solubility/json."
    )

    assert destination is not None
    assert destination.output_dir == (
        "/home/aaltamimi2/case-studies/case-1/01-ldpe-evoh-pet-solubility/json"
    )
