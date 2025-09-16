def test_package_imports() -> None:
    """Test that the package can be imported and has expected attributes."""
    import fastf1_analytics

    assert hasattr(fastf1_analytics, "load_session")
    assert hasattr(fastf1_analytics, "apply_style")
    assert hasattr(fastf1_analytics, "__version__")
