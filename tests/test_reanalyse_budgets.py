"""A converged run is finished, not truncated.

`reanalyse_depth` decides which arms may *bound* a target they never reached.
Getting that wrong changes signs: an arm dropped for being short is excused, and
an arm scored at a truncated length is condemned for a power cut.

The mistake this file exists to prevent is the second kind's mirror image, found
on 2026-08-30. OpenFOAM stops early when `residualControl` is satisfied and
prints "SIMPLE solution converged in N iterations". Judging completeness by
length alone calls that a truncation -- so it silently discards the arms that
converge *fastest*, which on the 13-case corpus meant the oracle control on one
case and a control reading +49.7% where it should read +93%. A rule that
penalises success is worse than no rule.
"""

from __future__ import annotations

from scripts.reanalyse_depth import scoreable_budgets


def test_a_run_at_the_full_budget_may_bound():
    lengths = {("c", "cold"): 6000, ("c", "warm"): 6000}
    budgets = scoreable_budgets(lengths, {})
    assert budgets[("c", "warm")] == 6000


def test_a_truncated_run_is_left_unscored():
    # Half the budget and no convergence message: a power cut, not a result.
    lengths = {("c", "cold"): 6000, ("c", "warm"): 3000}
    budgets = scoreable_budgets(lengths, {("c", "warm"): False})
    assert budgets[("c", "warm")] == 0


def test_a_converged_run_is_scored_at_its_own_length():
    # The bug: 1289 of 6000, but OpenFOAM exited because residualControl was
    # met. There are no further iterations to be had, so 1289 *is* the budget.
    lengths = {("c", "cold"): 6000, ("c", "oracle"): 1289}
    budgets = scoreable_budgets(lengths, {("c", "oracle"): True})
    assert budgets[("c", "oracle")] == 1289


def test_convergence_rescues_only_the_run_that_converged():
    lengths = {("c", "cold"): 6000, ("c", "oracle"): 1289, ("c", "dead"): 1200}
    budgets = scoreable_budgets(lengths,
                                {("c", "oracle"): True, ("c", "dead"): False})
    assert budgets[("c", "oracle")] == 1289
    assert budgets[("c", "dead")] == 0


def test_the_budget_is_the_longest_run_in_the_tree_not_a_constant():
    # Trees are scored at whatever budget they were run at; nothing may assume
    # 6000. A tree whose longest run is 2000 must accept 1900 as complete.
    lengths = {("c", "cold"): 2000, ("c", "warm"): 1900}
    assert scoreable_budgets(lengths, {})[("c", "warm")] == 1900


def test_an_empty_tree_does_not_divide_by_a_missing_maximum():
    assert scoreable_budgets({}, {}) == {}
