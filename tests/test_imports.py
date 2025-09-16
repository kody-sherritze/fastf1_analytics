def test_package_imports() -> None:
    """Test that the package can be imported and has expected attributes."""
    import fastf1_portfolio

    assert hasattr(fastf1_portfolio, "load_session")
    assert hasattr(fastf1_portfolio, "apply_style")
    assert hasattr(fastf1_portfolio, "__version__")
