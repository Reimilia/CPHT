"""Video frames follow physical time, independent of control decisions."""
import unittest
from fjbench.recording import FrameClock


class FrameClockTests(unittest.TestCase):
    def test_frame_rate_follows_physical_time(self):
        clock = FrameClock(25, .2)
        targets = []
        for t in (0., .002, .004, .034, .064, .094, .124, .154, .184, .2):
            targets += clock.due(t)
        self.assertEqual(len(targets), 5)
        for a, b in zip(targets, (0., .04, .08, .12, .16)):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
