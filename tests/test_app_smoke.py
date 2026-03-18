import unittest
import importlib.util


class AppSmokeTests(unittest.TestCase):
    def test_import_insikt_app(self):
        if importlib.util.find_spec("langchain_community") is None:
            self.skipTest("langchain_community is not installed in this environment")
        import insikt_app  # noqa: F401

        self.assertTrue(hasattr(insikt_app, "main"))


if __name__ == "__main__":
    unittest.main()
