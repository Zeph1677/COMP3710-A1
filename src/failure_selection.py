"""Deterministic selection helpers for failure-case reporting."""


def select_unique_patient_cases(
    ranking: list[dict[str, object]], max_patients: int
) -> list[dict[str, object]]:
    """Select ranked failure cases with at most one scan per patient."""

    if max_patients <= 0:
        raise ValueError("max_patients must be positive")

    selected: list[dict[str, object]] = []
    seen_patients: set[str] = set()
    for case in ranking:
        patient_id = str(case["patient_id"])
        if patient_id in seen_patients:
            continue
        selected.append(case)
        seen_patients.add(patient_id)
        if len(selected) == max_patients:
            break
    return selected
