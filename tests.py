#!/usr/bin/env python3.7
import unittest
from detect_utils import box_contains_exluded_points, get_center_point

class TestStringMethods(unittest.TestCase):

    def test_box_contains_exluded_points(self):
        self.assertTrue(box_contains_exluded_points(1280, 960, [0.15284267, 0.5236471, 0.22450888, 0.59080607], [[710,180]]))
        self.assertFalse(box_contains_exluded_points(1280, 960, [0.15284267, 0.5236471, 0.22450888, 0.59080607], [[400,600]]))

    def test_get_center_point(self):
        self.assertEquals(get_center_point(1280, 960, [0.06151412, 0.5364242, 0.1112366, 0.5791678]), [714, 83])


if __name__ == '__main__':
    unittest.main()
