import json
import logging

from pitchsense.logging import configure_logging


def test_json_logging_to_stderr(capsys):
    logger = configure_logging()
    logger.info("ready")
    record = json.loads(capsys.readouterr().err)
    assert record["level"] == "INFO"
    assert record["message"] == "ready"
    assert record["logger"] == "pitchsense"

    configure_logging()
    logger.log(logging.WARNING, "once")
    assert len(capsys.readouterr().err.splitlines()) == 1
