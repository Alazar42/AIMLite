import unittest
import sys


def main():
    print("Testing AIMLite...\n")
    loader = unittest.TestLoader()
    suite = loader.discover("test", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
