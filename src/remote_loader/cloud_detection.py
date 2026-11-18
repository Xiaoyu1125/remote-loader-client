from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from remote_loader.exceptions import ConfigurationError

TENCENT_INSTANCE_ID_URL = "http://metadata.tencentyun.com/latest/meta-data/instance-id"
VOLCENGINE_TOKEN_URL = "http://100.96.0.96/latest/api/token"
VOLCENGINE_INSTANCE_ID_URL = "http://100.96.0.96/latest/instance_id"
VOLCENGINE_TOKEN_TTL_SECONDS = "60"
_DMI_VENDOR = Path("/sys/class/dmi/id/sys_vendor")

CloudProvider = Literal["tencent", "volcengine", "baidu"]


class CloudNotDetectedError(ConfigurationError):
    """No supported cloud identifier is available in the current environment."""


@dataclass(frozen=True)
class CloudEnvironment:
    provider: CloudProvider
    scheduler_endpoint: str

    @property
    def label(self) -> str:
        return {"tencent": "腾讯云", "volcengine": "火山引擎", "baidu": "百度智能云"}[self.provider]


_ENVIRONMENTS: dict[str, CloudEnvironment] = {
    "tencent": CloudEnvironment("tencent", "http://10.110.0.2:8080"),
    "volcengine": CloudEnvironment("volcengine", "http://10.100.0.2:8080"),
    "baidu": CloudEnvironment("baidu", "http://10.100.0.2:8080"),
}


def environment_for_provider(provider: str) -> CloudEnvironment:
    if not isinstance(provider, str) or provider not in _ENVIRONMENTS:
        raise ConfigurationError("last_detected_cloud must be a supported cloud provider")
    return _ENVIRONMENTS[provider]


def _response_text(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> str | None:
    try:
        response = client.request(method, url, headers=headers)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    value = response.text.strip()
    return value if value and len(value) <= 256 else None


def _detect_tencent(client: httpx.Client) -> bool:
    instance_id = _response_text(client, "GET", TENCENT_INSTANCE_ID_URL)
    return instance_id is not None and instance_id.startswith("ins-")


def _detect_volcengine(client: httpx.Client) -> bool:
    token = _response_text(
        client,
        "PUT",
        VOLCENGINE_TOKEN_URL,
        headers={"X-volc-ecs-metadata-token-ttl-seconds": VOLCENGINE_TOKEN_TTL_SECONDS},
    )
    headers = None if token is None else {"X-volc-ecs-metadata-token": token}
    instance_id = _response_text(client, "GET", VOLCENGINE_INSTANCE_ID_URL, headers=headers)
    return instance_id is not None and instance_id.startswith("i-")


def _detect_baidu() -> bool:
    try:
        return _DMI_VENDOR.read_text(encoding="ascii").strip() == "Baidu Cloud"
    except (OSError, UnicodeError):
        return False


def detect_cloud_environment(
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 0.5,
) -> CloudEnvironment:
    """Identify the VM provider from instance metadata or the Baidu Cloud DMI vendor."""
    with httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout),
        trust_env=False,
        follow_redirects=False,
    ) as client:
        providers = [
            provider
            for provider, detected in (
                ("tencent", _detect_tencent(client)),
                ("volcengine", _detect_volcengine(client)),
                ("baidu", _detect_baidu()),
            )
            if detected
        ]
    if not providers:
        raise CloudNotDetectedError(
            "无法识别云厂商: 请检查腾讯云或火山引擎的实例元数据、百度云的 DMI 厂商标识, "
            "或使用 CLI 显式配置连接和 mTLS 凭据."
        )
    if len(providers) > 1:
        raise ConfigurationError("检测到多个云厂商标识, 无法选择 Scheduler 和服务端 CA.")
    return _ENVIRONMENTS[providers[0]]
