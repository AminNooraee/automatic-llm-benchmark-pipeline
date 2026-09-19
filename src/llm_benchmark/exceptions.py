"""Application-specific exceptions."""


class BenchmarkError(Exception):
    """Base class for expected benchmark pipeline errors."""


class ConfigurationError(BenchmarkError):
    """Raised when configuration loading or validation fails."""


class RunInitializationError(BenchmarkError):
    """Raised when a run directory cannot be initialized safely."""


class RunLifecycleError(BenchmarkError):
    """Raised when a pipeline stage violates the run state machine."""


class DatasetError(BenchmarkError):
    """Base class for dataset source, detection, and normalization errors."""


class DatasetSourceError(DatasetError):
    """Raised when a dataset source cannot be read."""


class DatasetDetectionError(DatasetError):
    """Raised when a dataset schema cannot be detected unambiguously."""


class DatasetValidationError(DatasetError):
    """Raised when an individual dataset record is invalid."""
