import unittest

from src.analyze.gaps import final_gap_records, pairwise_comparisons


def _row(
    *,
    run_id: str,
    problem: str,
    decoder: str,
    objective: float,
    target: float,
    logged_gap: float,
    logged_gap_pct: float,
) -> dict[str, object]:
    return {
        "_run_id": run_id,
        "_problem": problem,
        "_encoder": "attention",
        "_decoder": decoder,
        "_step": 100,
        "_timestamp": 1.0,
        "epoch": 10,
        "test/count": 100,
        "test/objective": objective,
        "test/target_objective": target,
        "test/optimal_gap": logged_gap,
        "test/optimal_gap_pct": logged_gap_pct,
        "test/feasibility_rate": 1.0,
        "test/inference_time_sec": 0.5,
    }


class GapAnalysisTests(unittest.TestCase):
    def test_normalizes_min_and_max_objectives_to_lower_is_better(self) -> None:
        rows = [
            _row(
                run_id="tsp-a",
                problem="tsp",
                decoder="attention_pointer",
                objective=12.0,
                target=10.0,
                logged_gap=2.0,
                logged_gap_pct=20.0,
            ),
            _row(
                run_id="mis-a",
                problem="mis",
                decoder="attention_pointer",
                objective=8.0,
                target=10.0,
                logged_gap=2.0,
                logged_gap_pct=20.0,
            ),
        ]

        records = final_gap_records(rows, fallback_split=None)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.aggregate_gap_pct == 20.0 for record in records))

    def test_flags_unstable_logged_percentage_but_keeps_stable_aggregate(self) -> None:
        rows = [
            _row(
                run_id="op-sigmoid",
                problem="orienteering",
                decoder="sigmoid_subset",
                objective=50.0,
                target=100.0,
                logged_gap=50.0,
                logged_gap_pct=-1.0e15,
            )
        ]

        record = final_gap_records(rows, fallback_split=None)[0]

        self.assertEqual(record.aggregate_gap_pct, 50.0)
        self.assertIn("unstable_logged_gap_pct", record.quality_flags)

    def test_pairwise_delta_is_negative_when_decoder_a_is_better(self) -> None:
        rows = [
            _row(
                run_id="tsp-a",
                problem="tsp",
                decoder="attention_pointer",
                objective=11.0,
                target=10.0,
                logged_gap=1.0,
                logged_gap_pct=10.0,
            ),
            _row(
                run_id="tsp-b",
                problem="tsp",
                decoder="lstm_pointer",
                objective=12.0,
                target=10.0,
                logged_gap=2.0,
                logged_gap_pct=20.0,
            ),
        ]

        comparison = pairwise_comparisons(
            final_gap_records(rows, fallback_split=None)
        )[0]

        self.assertEqual(comparison.winner, "attention_pointer")
        self.assertEqual(comparison.gap_delta_a_minus_b_pp, -10.0)
        self.assertEqual(comparison.winner_advantage_pp, 10.0)


if __name__ == "__main__":
    unittest.main()
