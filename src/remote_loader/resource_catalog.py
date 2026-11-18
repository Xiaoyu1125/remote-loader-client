from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneOption:
    id: str
    label: str


@dataclass(frozen=True)
class RegionOption:
    id: str
    label: str
    zones: tuple[ZoneOption, ...]


@dataclass(frozen=True)
class ProviderOption:
    id: str
    label: str
    regions: tuple[RegionOption, ...]


ALIYUN = ProviderOption(
    id="aliyun",
    label="Alibaba Cloud / 阿里云",
    regions=(
        RegionOption(
            id="cn-beijing",
            label="华北 2 (北京)",
            zones=(
                ZoneOption("cn-beijing-f", "可用区 F"),
                ZoneOption("cn-beijing-l", "可用区 L"),
            ),
        ),
        RegionOption(
            id="cn-wulanchabu",
            label="华北 6 (乌兰察布)",
            zones=(
                ZoneOption("cn-wulanchabu-a", "可用区 A"),
                ZoneOption("cn-wulanchabu-b", "可用区 B"),
            ),
        ),
        RegionOption(
            id="cn-hangzhou",
            label="华东 1 (杭州)",
            zones=(
                ZoneOption("cn-hangzhou-b", "可用区 B"),
                ZoneOption("cn-hangzhou-j", "可用区 J"),
            ),
        ),
        RegionOption(
            id="cn-shanghai",
            label="华东 2 (上海)",
            zones=(
                ZoneOption("cn-shanghai-b", "可用区 B"),
                ZoneOption("cn-shanghai-e", "可用区 E"),
                ZoneOption("cn-shanghai-l", "可用区 L"),
            ),
        ),
        RegionOption(
            id="eu-central-1",
            label="德国 (法兰克福)",
            zones=(ZoneOption("eu-central-1a", "可用区 A"),),
        ),
    ),
)

PROVIDERS: tuple[ProviderOption, ...] = (ALIYUN,)


def find_provider(provider_id: str) -> ProviderOption | None:
    return next((provider for provider in PROVIDERS if provider.id == provider_id), None)


def find_region(provider: ProviderOption, region_id: str) -> RegionOption | None:
    return next((region for region in provider.regions if region.id == region_id), None)
