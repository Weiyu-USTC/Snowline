"""Guard paired comparisons against silently shifted masks or random streams."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate_alta_tuning import assert_identical_mask, aligned_prediction_batches


class PairedEvaluationChecks(unittest.TestCase):
    def test_rejects_different_geometry_even_when_valid_counts_match(self):
        before = np.array([[True,False],[True,False]])
        after = np.array([[False,True],[False,True]])
        self.assertEqual(before.sum(),after.sum())
        with self.assertRaises(AssertionError):
            assert_identical_mask(after,before)
        assert_identical_mask(before.copy(),before)

    def test_reused_prediction_boundary_keeps_original_case_and_seed_positions(self):
        # Three development cases share original batch164 with the first test.
        reference = np.arange(347)
        target = reference[167:]
        schedule = list(aligned_prediction_batches(reference,target))
        start,batch,destinations = schedule[0]
        self.assertEqual(start,164)
        np.testing.assert_array_equal(batch,[164,165,166,167])
        self.assertEqual(destinations,[(3,0)])
        pairs = [(int(batch[offset]),destination) for _,batch,destinations in schedule for offset,destination in destinations]
        self.assertEqual(pairs,[(int(value),i) for i,value in enumerate(target)])
        self.assertEqual(schedule[-1][0],344)

    def test_missing_target_is_not_silently_omitted(self):
        with self.assertRaises(ValueError):
            list(aligned_prediction_batches([1,2,3],[4]))


if __name__=='__main__': unittest.main()
