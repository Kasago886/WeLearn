"""错误分类测试（FR-709、FR-712）。"""

import pytest

from app.infra.deepseek.errors import LlmError, LlmErrorCategory


@pytest.mark.parametrize(
    ("status", "category", "retryable"),
    [
        (400, LlmErrorCategory.BAD_REQUEST, False),
        (401, LlmErrorCategory.AUTH, False),
        (402, LlmErrorCategory.BALANCE, False),
        (422, LlmErrorCategory.INVALID_PARAMS, False),
        (429, LlmErrorCategory.RATE_LIMIT, True),
        (500, LlmErrorCategory.SERVER_ERROR, True),
        (503, LlmErrorCategory.UNAVAILABLE, True),
    ],
)
def test_official_status_codes_are_mapped(status: int, category: LlmErrorCategory, retryable: bool) -> None:
    error = LlmError.from_status(status, detail="body")
    assert error.category is category
    assert error.retryable is retryable
    assert error.status_code == status
    assert error.user_message


def test_unknown_status_codes_fall_back_by_range() -> None:
    assert LlmError.from_status(418).retryable is False
    assert LlmError.from_status(504).retryable is True


def test_error_factories() -> None:
    assert LlmError.config("x").retryable is False
    assert LlmError.timeout("t").retryable is True
    assert LlmError.network("n").retryable is True
    protocol = LlmError.protocol("p")
    assert protocol.category is LlmErrorCategory.PROTOCOL
    assert protocol.retryable is False


def test_payload_shape_is_safe_for_frontend() -> None:
    payload = LlmError.from_status(401, detail="secret").to_payload()
    assert payload == {
        "category": "auth",
        "message": LlmError.from_status(401).user_message,
        "retryable": False,
    }
    # detail 不进入前端载荷，避免泄露响应体
    assert "secret" not in str(payload)
