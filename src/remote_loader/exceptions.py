"""Public exception hierarchy with actionable, non-sensitive messages."""


class RemoteLoaderError(RuntimeError):
    """Base class for all client errors."""


class ConfigurationError(RemoteLoaderError, ValueError):
    """The local batch or runtime configuration is invalid."""


class AuthenticationError(RemoteLoaderError):
    """Authentication failed or the identity token expired."""


class PermissionDeniedError(RemoteLoaderError):
    """The caller is not allowed to access the requested resource."""


class SessionNotFoundError(RemoteLoaderError):
    """The remote loader session does not exist."""


class SessionExpiredError(RemoteLoaderError):
    """The remote loader session or stream token expired."""


class SessionStateError(RemoteLoaderError):
    """An operation is invalid for the current loader/session state."""


class RateLimitError(RemoteLoaderError):
    """The account or tenant exceeded a service limit."""


class ServiceUnavailableError(RemoteLoaderError):
    """The control or data service is temporarily unavailable."""


class ProtocolError(RemoteLoaderError):
    """The peer sent an invalid or unsupported protocol message."""


class SchemaMismatchError(ProtocolError):
    """The received batch does not match the negotiated schema."""


class ChecksumError(ProtocolError):
    """The received batch payload failed integrity verification."""


class BatchTimeoutError(RemoteLoaderError, TimeoutError):
    """No complete batch arrived before the configured timeout."""


class EndOfStream(RemoteLoaderError, StopIteration):
    """The server intentionally ended a finite data stream."""


class StatisticsFailedError(RemoteLoaderError):
    """A statistics job failed; no partial result is available."""


class StatisticsCancelledError(RemoteLoaderError):
    """A statistics job was cancelled; no partial result is available."""


class StatisticsTimeoutError(RemoteLoaderError, TimeoutError):
    """The caller's wait deadline elapsed; the remote job is not cancelled."""
