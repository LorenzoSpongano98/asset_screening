import enum
import re
from dataclasses import dataclass
from datetime import date

ISO_4217_PATTERN = re.compile(r"[A-Z]{3}")
ISO_10383_PATTERN = re.compile(r"[A-Z9-0]{4}")


class AdjustmentMethod(enum.Flag):
    NONE = 0
    SPLIT = enum.auto()
    DIVIDEND = enum.auto()


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Dataclass to represent provider capabilities"""

    adjustment_method: AdjustmentMethod
    currencies: frozenset[str]
    markets: frozenset[str]
    max_symbols_per_request: int
    max_days_per_request: int | None
    history_start: date | None
    min_request_interval_s: float

    def __post_init__(self):

        if self.max_symbols_per_request < 1:
            raise ValueError(
                f"Symbols per request should be at least 1. Got {self.max_symbols_per_request}."
            )

        if not isinstance(self.currencies, frozenset):
            raise TypeError(
                f"Currencies must be a frozenset. Got {type(self.currencies).__name__}."
            )

        if not isinstance(self.markets, frozenset):
            raise TypeError(
                f"Markets must be a frozenset. Got {type(self.markets).__name__}."
            )

        for c in self.currencies:
            if not ISO_4217_PATTERN.fullmatch(c):
                raise ValueError(
                    f"Currencies should follow the ISO 4217 standard nomenclature. Got {c!r}."
                )

        for m in self.markets:
            if not ISO_10383_PATTERN.fullmatch(m):
                raise ValueError(
                    f"Markets should follow the ISO 10383 standard nomenclature. Got {m!r}. "
                )

        if self.max_days_per_request is not None and self.max_days_per_request < 1:
            raise ValueError(
                f"Days per request should be at least 1. Got {self.max_days_per_request}."
            )

        if self.min_request_interval_s < 0:
            raise ValueError(
                f"Min request interval should be non negative. Got {self.min_request_interval_s}."
            )
