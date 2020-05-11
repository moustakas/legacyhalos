def test_suite():
    """Returns unittest.TestSuite of legacyhalos tests for setup.py test
    """
    import unittest
    from os.path import dirname
    py_dir = dirname(dirname(__file__))
    return unittest.defaultTestLoader.discover(py_dir,
                                               top_level_dir=dirname(py_dir))
