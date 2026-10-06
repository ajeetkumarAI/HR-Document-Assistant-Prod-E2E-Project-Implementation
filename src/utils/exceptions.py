"""Domain exceptions mapped to HTTP responses in the API layer."""


class RAGError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(RAGError):
    code = "configuration_error"


class IngestionError(RAGError):
    status_code = 422
    code = "ingestion_error"


class UnsupportedFileTypeError(IngestionError):
    status_code = 415
    code = "unsupported_file_type"


class GuardrailViolation(RAGError):
    status_code = 400
    code = "guardrail_violation"


class AuthError(RAGError):
    status_code = 401
    code = "unauthorized"


class RateLimitExceeded(RAGError):
    status_code = 429
    code = "rate_limited"


class LLMError(RAGError):
    status_code = 502
    code = "llm_error"


class RetrievalError(RAGError):
    status_code = 503
    code = "retrieval_error"
