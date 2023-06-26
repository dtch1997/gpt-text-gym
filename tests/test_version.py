import unittest

import gpt_text_gym


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check gpt_text_gym exposes a version attribute """
        self.assertTrue(hasattr(gpt_text_gym, "__version__"))
        self.assertIsInstance(gpt_text_gym.__version__, str)


if __name__ == "__main__":
    unittest.main()
