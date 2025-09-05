def test_package_imports():
    import fastf1_portfolio

    assert hasattr(fastf1_portfolio, "load_session")
    assert hasattr(fastf1_portfolio, "apply_style")
