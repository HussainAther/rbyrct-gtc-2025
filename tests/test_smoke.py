def test_imports():
    import rbyrct_core
    from rbyrct_core.core import forward_project, mart_reconstruct
    assert callable(forward_project) and callable(mart_reconstruct)

