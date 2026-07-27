from scripts.report_kedf_feasibility_progress import electronic_iterations


def test_electronic_iterations_counts_each_md_step() -> None:
    text = """
STEP OF MOLECULAR DYNAMICS: 1
 TN0 a
 TN1 b
STEP OF MOLECULAR DYNAMICS: 2
 TN0 c
 TN1 d
 TN2 e
"""
    assert electronic_iterations(text) == [2, 3]
