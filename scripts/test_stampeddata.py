# coding=utf-8

import random
import unittest

from collections import deque

import numpy as np

from utils import StampedData



class TestStampedData(unittest.TestCase):
    """
    """
    def test_int(self):
        """Input is int"""
        d = StampedData()

        l = [3, 4, 1, 6, 7, 9, 3, 2, 1, 5]

        for i, x in enumerate(l):
            d.add(i, x)

        m = d.reduce(reduction="mean")
        self.assertEqual(m, np.array(l).mean())

        beg = 3
        end = 6
        m = d.reduce(beg, end, reduction="mean")
        self.assertEqual(m, np.array(l[beg:end+1]).mean())


    def test_float(self):
        """Input is float"""
        d = StampedData()

        l = np.random.rand(10)

        for i, x in enumerate(l):
            d.add(i, x)

        m = d.reduce(reduction="mean")
        self.assertTrue(abs(m - l.mean()) < 1e-9)

        beg = 3
        end = 8
        m = d.reduce(beg, end, reduction="mean")
        self.assertTrue(abs(m - l[beg:end+1].mean()) < 1e-9)


    def test_list(self):
        """Input is list"""
        d = StampedData()

        data = np.random.rand(10, 5)

        for i in range(data.shape[0]):
            l = list(data[i])
            d.add(i, l)

        m = d.reduce(reduction="mean")
        self.assertTrue(np.linalg.norm(m - data.mean(axis=0)) < 1e-9)

        beg = 3
        end = 8
        m = d.reduce(beg, end, reduction="mean")
        self.assertTrue(np.linalg.norm(m - data[beg:end+1].mean(axis=0)) < 1e-9)

    def test_array(self):
        """Input is array"""
        d = StampedData()

        data = np.random.rand(10, 5)

        for i in range(data.shape[0]):
            l = list(data[i])
            d.add(i, np.array(l))

        m = d.reduce(reduction="mean")
        self.assertTrue(np.linalg.norm(m - data.mean(axis=0)) < 1e-9)

        beg = 3
        end = 8
        m = d.reduce(beg, end, reduction="mean")
        self.assertTrue(np.linalg.norm(m - data[beg:end+1].mean(axis=0)) < 1e-9)

    def test_dict(self):
        """Input is dict"""
        d = StampedData()

        data = [
            {"robot": 0.1, "screen": 0.3},
            {"robot": 0.2, "screen": 0.1},
            {"robot": 0.1, "screen": 0.6},
            {"robot": 0.5, "screen": 0.4},
            {"robot": 0.5, },
            {"robot": 0.1, "screen": 0.4},
            {"robot": 0.2, "screen": 0.4, "person": 0.1},
            {"robot": 0.1, "screen": 0.4},
            {"robot": 0.1, },
            {"robot": 0.1, "screen": 0.6, "person": 0.3},
        ]

        for i in range(len(data)):
            d.add(i, data[i])

        m = d.reduce(reduction="mean")

        expected = {"robot": 0.2, "screen": 0.4, "person": 0.2}

        for k in m:
            self.assertTrue(k in expected)
        for k in expected:
            self.assertTrue(k in m)

        for k in m:
            self.assertTrue(abs(m[k] - expected[k]) < 1e-9)

    def test_update_key(self):
        """Input is dict"""
        d = StampedData()

        data = [
            {"robot": 0.1, "person1": 0.3},
            {"robot": 0.2, "person1": 0.1},
            {"robot": 0.1, "person1": 0.6},
            {"robot": 0.5, "person1": 0.2},
            {"robot": 0.5, },
            {"robot": 0.1, "person4": 0.4},
            {"robot": 0.2, "person4": 0.4, "person7": 0.1},
            {"robot": 0.1, "person4": 0.4},
            {"robot": 0.1, },
            {"robot": 0.1, "person4": 0.8, "person7": 0.3},
        ]

        for i in range(len(data)):
            d.add(i, data[i])

        m = d.reduce(reduction="mean")
        expected = {"robot": 0.2, "person4": 0.5, "person1": 0.3, "person7": 0.2}
        for k in m:
            self.assertTrue(abs(m[k] - expected[k]) < 1e-9)

        d.update_key("person4", "person1")

        m = d.reduce(reduction="mean")
        expected = {"robot": 0.2, "person1": 0.4, "person7": 0.2}
        for k in m:
            self.assertTrue(abs(m[k] - expected[k]) < 1e-9)

    def test_merge(self):
        t1 = []
        t2 = []
        l1 = [3, 4, 1, 6, 7, 9, 3, 2, 1, 5]
        l2 = [5, 1, 2, 3, 7, 5, 1]
        s1 = StampedData()
        s2 = StampedData()

        t = 1
        for e in l1:
            s1.add(t, e)
            t1.append(t)
            t += 1

        t += 10

        for e in l2:
            s2.add(t, e)
            t2.append(t)
            t += 1

        s1.merge(s2)

        s3 = StampedData()
        for t, l in zip(t1 + t2, l1 + l2):
            s3.add(t, l)


        self.assertEqual(s1.data, s3.data)

    def test_merge_sort(self):
        t1 = []
        t2 = []
        l1 = [3, 4, 1, 6, 7, 9, 3, 2, 1, 5]
        l2 = [5, 1, 2, 3, 7, 5, 1]
        s1 = StampedData()
        s2 = StampedData()

        t = 1
        for e in l1:
            s1.add(t, e)
            t1.append(t)
            t += 1

        t += 10

        for e in l2:
            s2.add(t, e)
            t2.append(t)
            t += 1

        s2.merge(s1)

        s3 = StampedData()
        for t, l in zip(t1 + t2, l1 + l2):
            s3.add(t, l)

        self.assertEqual(s2.data, s3.data)

if __name__ == "__main__":
    unittest.main(verbosity=10)
