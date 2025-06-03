import unittest

import triage


# NOTE: This structure is taken from the python documentation
# Remove once we have actual unit tests
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    # NOTE: Make one test fail to check
    # def test_fail(self):
    #     self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
