"""Domain errors for future pipeline boundaries."""


class PitchSenseError(Exception):
    """Base error for PitchSense operations."""


class ConfigurationError(PitchSenseError):
    """Invalid or unusable configuration."""


class InputVideoError(PitchSenseError):
    """Input video cannot be read."""


class EmptyVideoError(InputVideoError):
    """Input video contains no frames."""


class BackendUnavailableError(PitchSenseError):
    """Requested inference backend is unavailable."""


class ModelLoadError(PitchSenseError):
    """Model could not be loaded."""


class IncompatibleModelClassesError(ModelLoadError):
    """Model classes do not satisfy the pipeline contract."""


class OutputWriteError(PitchSenseError):
    """Output could not be written."""
