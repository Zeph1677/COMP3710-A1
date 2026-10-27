from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "Submission-final" / "README.md"


def test_submission_readme_reports_formal_per_class_metrics_and_parameters() -> None:
    text = README.read_text(encoding="utf-8")

    table_rows = {
        tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        for line in text.splitlines()
        if line.lstrip().startswith("|")
    }
    expected_rows = (
        ("Background", "0.982804", "0.966255", "0.994107", "0.988292"),
        ("Body", "0.958148", "0.919909", "0.977807", "0.956600"),
        ("Bones", "0.859094", "0.753894", "0.904555", "0.825973"),
        ("Bladder", "0.957028", "0.917811", "0.960610", "0.924386"),
        ("Rectum", "0.852767", "0.747437", "0.861629", "0.758714"),
        ("Prostate", "0.822599", "0.702348", "0.831385", "0.715546"),
        (
            "Baseline",
            "1,400,646",
            "977.6 MiB",
            "1,047.4 MiB",
            "1,369.1 s",
            "0.248920 s",
        ),
        (
            "Improved",
            "1,423,106",
            "1,193.4 MiB",
            "1,047.5 MiB",
            "1,388.2 s",
            "0.249449 s",
        ),
    )
    for row in expected_rows:
        assert row in table_rows


def test_submission_readme_embeds_all_five_paired_failure_cases() -> None:
    text = README.read_text(encoding="utf-8")

    for patient in ("b037", "h007", "k042", "l011", "m013"):
        assert text.count(f"assets/failure_case_{patient}_week") == 1
