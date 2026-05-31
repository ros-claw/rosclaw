"""Pytest configuration for ROSClaw test suite."""

import logging


def pytest_runtest_setup(item):
    """Ensure log propagation is enabled so caplog works.

    Some pytest environments (e.g. with ROS launch-testing plugins)
    create loggers with propagate=False, which breaks the standard
    ``caplog`` fixture. We force propagation back on for all loggers
    before each test runs.
    """
    # Fix existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    # Also fix root logger children default
    logging.getLogger().propagate = True


def pytest_runtest_call(item):
    """Re-apply propagate fix right before the test body executes."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    logging.getLogger().propagate = True
