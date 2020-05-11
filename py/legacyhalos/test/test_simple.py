import unittest

class TestSimple(unittest.TestCase):
    
    def test_simple(self):
        import legacyhalos
        self.assertEqual(1, 1)

def main():
    unittest.main()

if __name__ == "__main__":
    unittest.main()
