from failure_selection import select_unique_patient_cases


def test_select_unique_patient_cases_keeps_highest_ranked_scan_per_patient():
    ranking = [
        {"patient_id": "B037", "case_id": "B037_Week0", "error_score": 3088.0},
        {"patient_id": "L011", "case_id": "L011_Week2", "error_score": 930.0},
        {"patient_id": "L011", "case_id": "L011_Week7", "error_score": 878.5},
        {"patient_id": "K042", "case_id": "K042_Week0", "error_score": 876.0},
        {"patient_id": "M013", "case_id": "M013_Week0", "error_score": 622.0},
        {"patient_id": "H007", "case_id": "H007_Week0", "error_score": 476.5},
    ]

    selected = select_unique_patient_cases(ranking, max_patients=5)

    assert [(case["patient_id"], case["case_id"]) for case in selected] == [
        ("B037", "B037_Week0"),
        ("L011", "L011_Week2"),
        ("K042", "K042_Week0"),
        ("M013", "M013_Week0"),
        ("H007", "H007_Week0"),
    ]


def test_select_unique_patient_cases_rejects_non_positive_limit():
    try:
        select_unique_patient_cases([], max_patients=0)
    except ValueError as error:
        assert str(error) == "max_patients must be positive"
    else:
        raise AssertionError("zero patient limit must be rejected")
