import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-integration", action="store_true", help="skip integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests which connect to the Pennsieve platform"
    )


def pytest_runtest_setup(item):
    if "integration" in item.keywords and item.config.getoption("--skip-integration"):
        pytest.skip("Skipping integration tests")
