# timeseries-modeling

## Running tests

```console
$ python -m unittest discover tests/
F
======================================================================
FAIL: test_add (test_main.MainTest.test_add)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/lx/Development/timeseries-modeling/tests/test_main.py", line 16, in test_add
    self.assertEqual(result, expected_sum)
AssertionError: 0 != 3

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (failures=1)
```
