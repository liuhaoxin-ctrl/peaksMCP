"""C3 gold-selection grading: follow the selection the code performs.

The grader used to skip C3 whenever a ``fit_gold`` receiver could not be traced
to a literal index, which is exactly what the natural pattern looks like::

    for gi in summary.gold:            # or: next(i for i in summary.gold ...)
        gold = scans[f"BP_{gi:04d}"]
        fit = gold.fit_gold(plot=False)

A skipped check can never pass, so a correct agent that delegated the choice to
the system's own classification scored *lower* than one that hard-coded the
index.  These tests pin the data-flow resolution that replaced it, including the
cases it must still refuse (fitting a cut, or an untraceable object).
"""

from __future__ import annotations

from benchmark.run_case import _fit_gold_receiver_indices

GOLD = {20}


def _resolve(code: str, gold: set[int] | None = None) -> tuple[set[int], bool]:
    return _fit_gold_receiver_indices([code], gold if gold is not None else GOLD)


# --------------------------------------------------------------------------- #
# the patterns a correct agent actually writes                                #
# --------------------------------------------------------------------------- #

def test_loop_over_the_classified_gold_resolves_to_the_gold_index():
    code = (
        "summary = inspect_experiment(scans)\n"
        "for gi in summary.gold:\n"
        "    gold = scans[f'BP_{gi:04d}']\n"
        "    fit = gold.fit_gold(plot=False)\n"
    )
    found, unresolved = _resolve(code)
    assert found == GOLD
    assert unresolved is False


def test_next_over_the_classified_gold_resolves_to_the_gold_index():
    code = (
        "summary = inspect_experiment(scans)\n"
        "gold_index = next(i for i in summary.gold if i in present)\n"
        "gold_stem = f'BP_{gold_index:04d}'\n"
        "gold = scans[gold_stem]\n"
        "fit = gold.fit_gold(plot=False)\n"
    )
    found, unresolved = _resolve(code)
    assert found == GOLD
    assert unresolved is False


def test_experiment_gold_ignores_unrelated_digits_in_input_paths():
    code = (
        "first_conversion = peaks.pxt2nc('/tmp/pytest-499/input')\n"
        "experiment = peaks.load_experiment(first_conversion.destination)\n"
        "gold_index = next(i for i in experiment.gold if i in present)\n"
        "gold_stem = f'BP_{gold_index:04d}'\n"
        "gold = experiment[gold_stem]\n"
        "fit = gold.fit_gold(plot=False)\n"
    )
    found, unresolved = _resolve(code)
    assert found == GOLD
    assert unresolved is False


def test_summary_gold_subscript_and_bare_call_resolve():
    code = (
        "gold = scans[summary.gold[0]]\n"
        "fit = fit_gold(gold, plot=False)\n"
    )
    found, unresolved = _resolve(code)
    assert found == GOLD
    assert unresolved is False


def test_experiment_gold_position_is_not_mistaken_for_scan_zero():
    code = (
        "gold_index = experiment.gold[0]\n"
        "gold_scan = experiment[gold_index]\n"
        "gold_fit = gold_scan.fit_gold()\n"
    )
    found, unresolved = _resolve(code)
    assert found == GOLD
    assert unresolved is False


def test_literal_bindings_still_resolve():
    for code in (
        "gold = scans['BP_0020']\nfit = gold.fit_gold()\n",
        "gold = scans[20]\nfit = gold.fit_gold()\n",
        "fit = scans['BP_0020'].fit_gold()\n",
        "gold_index = 20\ngold = scans[f'BP_{gold_index:04d}']\nfit = gold.fit_gold()\n",
    ):
        found, unresolved = _resolve(code)
        assert found == GOLD, code
        assert unresolved is False, code


# --------------------------------------------------------------------------- #
# what must still be refused                                                  #
# --------------------------------------------------------------------------- #

def test_fitting_a_cut_resolves_to_the_cut_and_fails_the_equality_check():
    code = (
        "cut = scans['BP_0005']\n"
        "fit = cut.fit_gold(plot=False)\n"
    )
    found, _ = _resolve(code)
    assert found == {5}
    assert found != GOLD


def test_an_untraceable_receiver_stays_unresolved():
    code = "fit = load_something().fit_gold(plot=False)\n"
    found, unresolved = _resolve(code)
    assert found == set()
    assert unresolved is True


def test_cut_list_literals_in_the_same_cell_are_not_mistaken_for_gold():
    code = (
        "cuts = ['BP_0005', 'BP_0006', 'BP_0009']\n"
        "gold = scans['BP_0020']\n"
        "fit = gold.fit_gold(plot=False)\n"
    )
    found, _ = _resolve(code)
    assert found == GOLD


def test_no_gold_set_available_keeps_the_classified_branch_unresolved():
    """Without a task gold set the ``summary.gold`` branch stays unresolved
    instead of inventing an index."""
    code = (
        "gold = scans[f'BP_{gi:04d}']\n"
        "fit = gold.fit_gold(plot=False)\n"
    )
    found, unresolved = _fit_gold_receiver_indices([code])
    assert found == set()
    assert unresolved is True
