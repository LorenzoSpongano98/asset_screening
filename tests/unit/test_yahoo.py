import io
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yfinance as yf
from asset_screening.data.capabilities import AdjustmentMethod
from asset_screening.data.contracts import (
    FetchResult,
    SymbolErrorType,
    UnavailableProviderError,
)
from asset_screening.data.providers import yahoo
from asset_screening.data.providers.yahoo import YahooFinanceProvider
from hypothesis import given, settings
from hypothesis import strategies as st
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

TEMPORARY = SymbolErrorType.TEMPORARY
PERMANENT = SymbolErrorType.PERMANENT

START = date(2024, 1, 2)
END = date(2024, 1, 4)


def make_frame(base: float = 100.0, rows: int = 3) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-02", periods=rows, freq="B", tz="America/New_York", name="Date"
    )
    steps = [float(i) for i in range(rows)]
    return pd.DataFrame(
        {
            "Open": [base + s for s in steps],
            "High": [base + s + 1.5 for s in steps],
            "Low": [base + s - 1.5 for s in steps],
            "Close": [base + s + 0.5 for s in steps],
            "Adj Close": [base + s for s in steps],
            "Volume": [1_000 + int(s) for s in steps],
            "Dividends": [0.0] * rows,
            "Stock Splits": [0.0] * rows,
        },
        index=index,
    )


class FakeYahoo:
    def __init__(self) -> None:
        self.script: dict[str, str] = {}
        self.events: list[tuple[Any, ...]] = []

    def use(self, script: dict[str, str]) -> list[str]:
        self.script = dict(script)
        return list(script)

    @property
    def history_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [(e[1], e[2]) for e in self.events if e[0] == "history"]

    @property
    def history_symbols(self) -> list[str]:
        return [symbol for symbol, _ in self.history_calls]

    @property
    def sleeps(self) -> list[float]:
        return [e[1] for e in self.events if e[0] == "sleep"]

    @staticmethod
    def frame_for(symbol: str) -> pd.DataFrame:
        return make_frame(base=100.0 + sum(map(ord, symbol)) % 50)

    @staticmethod
    def failure(symbol: str, kind: str) -> Exception:
        match kind:
            case "rate_limit":
                return YFRateLimitError()
            case "tz_missing":
                return YFTzMissingError(symbol)
            case "prices_missing":
                return YFPricesMissingError(symbol, debug_info="")
            case _:
                raise ValueError(f"Unknown failure kind: {kind!r}")

    def ticker(self, symbol: str) -> "_FakeTicker":
        return _FakeTicker(self, symbol)


class _FakeTicker:
    def __init__(self, world: FakeYahoo, symbol: str) -> None:
        self.world = world
        self.symbol = symbol

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.world.events.append(("history", self.symbol, kwargs))
        outcome = self.world.script.get(self.symbol, "ok")
        match outcome:
            case "ok":
                return self.world.frame_for(self.symbol)
            case "empty":
                return make_frame().iloc[0:0]
            case _:
                raise self.world.failure(self.symbol, outcome)


@pytest.fixture
def fake_yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    world = FakeYahoo()
    monkeypatch.setattr(yahoo.yf, "Ticker", world.ticker)

    def fake_sleep(seconds: float) -> None:
        world.events.append(("sleep", seconds))

    monkeypatch.setattr(yahoo, "time", SimpleNamespace(sleep=fake_sleep))
    return world


@pytest.fixture
def provider(
    fake_yahoo: FakeYahoo, monkeypatch: pytest.MonkeyPatch
) -> YahooFinanceProvider:
    monkeypatch.setattr(yf.config.debug, "hide_exceptions", True)
    return YahooFinanceProvider()


def decode(payload: Any) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(payload.content))


def served(result: FetchResult) -> list[str]:
    return [p.request_parameters["symbol"] for p in result.payloads]


def failed(result: FetchResult) -> list[tuple[str, SymbolErrorType]]:
    return [(e.symbol, e.error_type) for e in result.symbol_errors]


def test_constructor_makes_yfinance_raise_instead_of_hiding_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(yf.config.debug, "hide_exceptions", True)
    YahooFinanceProvider()
    assert yf.config.debug.hide_exceptions is False


class TestCapabilities:
    def test_provider_code(self, provider: YahooFinanceProvider) -> None:
        assert provider.provider_code == "yahoo"

    def test_declared_values(self, provider: YahooFinanceProvider) -> None:
        caps = provider.capabilities()
        assert caps.adjustment_method == AdjustmentMethod.SPLIT
        assert caps.currencies == {"USD", "EUR", "GBP"}
        assert caps.markets == {"XNYS", "XNAS", "XMIL", "XPAR", "XETR", "XLON"}
        assert caps.max_symbols_per_request == 50
        assert caps.max_days_per_request is None
        assert caps.history_start is None
        assert caps.min_request_interval_s == 1.0

    def test_adjustment_claim_matches_what_fetch_requests(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok"})
        provider.fetch(START, END, symbols)
        ((_, kwargs),) = fake_yahoo.history_calls
        claims_dividends = (
            AdjustmentMethod.DIVIDEND in provider.capabilities().adjustment_method
        )
        assert kwargs["auto_adjust"] is claims_dividends


@st.composite
def ohlcv_frames(draw: st.DrawFn) -> pd.DataFrame:
    n = draw(st.integers(min_value=1, max_value=15))
    tz = draw(st.sampled_from(["UTC", "America/New_York", "Europe/Rome"]))
    values = st.lists(st.floats(), min_size=n, max_size=n)
    data = {name: draw(values) for name in ("Open", "High", "Low", "Close", "Volume")}
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz=tz, name="Date")
    return pd.DataFrame(data, index=index)


class TestToParquet:
    def test_roundtrip_keeps_values_and_turns_the_index_into_a_column(self) -> None:
        df = make_frame()
        back = pd.read_parquet(io.BytesIO(YahooFinanceProvider._to_parquet(df)))
        assert list(back.columns) == ["Date", *df.columns]
        pd.testing.assert_frame_equal(back, df.reset_index())

    def test_output_is_a_parquet_file(self) -> None:
        content = YahooFinanceProvider._to_parquet(make_frame())
        assert isinstance(content, bytes)
        assert content[:4] == b"PAR1"
        assert content[-4:] == b"PAR1"

    def test_input_is_not_mutated(self) -> None:
        df = make_frame()
        snapshot = df.copy(deep=True)
        YahooFinanceProvider._to_parquet(df)
        pd.testing.assert_frame_equal(df, snapshot)
        assert df.index.name == "Date"

    @settings(max_examples=40, deadline=None)
    @given(df=ohlcv_frames())
    def test_roundtrip_property(self, df: pd.DataFrame) -> None:
        back = pd.read_parquet(io.BytesIO(YahooFinanceProvider._to_parquet(df)))
        pd.testing.assert_frame_equal(back, df.reset_index())


class TestFetchInputValidation:
    def test_rejects_more_symbols_than_the_declared_limit(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        limit = provider.capabilities().max_symbols_per_request
        symbols = [f"S{i}" for i in range(limit + 1)]
        with pytest.raises(ValueError, match=rf"{limit}.*{limit + 1}"):
            provider.fetch(START, END, symbols)
        assert fake_yahoo.events == []

    def test_accepts_exactly_the_limit(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        limit = provider.capabilities().max_symbols_per_request
        symbols = [f"S{i}" for i in range(limit)]
        result = provider.fetch(START, END, symbols)
        assert len(result.payloads) == limit
        assert result.symbol_errors == ()
        assert len(fake_yahoo.history_calls) == limit

    def test_no_symbols_gives_an_empty_result_and_no_exception(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        result = provider.fetch(START, END, [])
        assert result == FetchResult(payloads=(), symbol_errors=())
        assert fake_yahoo.events == []


class TestFetchLoop:
    def test_one_payload_per_symbol_in_input_order_with_right_content(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"CCC": "ok", "AAA": "ok", "BBB": "ok"})
        result = provider.fetch(START, END, symbols)

        assert isinstance(result.payloads, tuple)
        assert isinstance(result.symbol_errors, tuple)
        assert result.symbol_errors == ()
        assert served(result) == ["CCC", "AAA", "BBB"]
        for payload, symbol in zip(result.payloads, symbols, strict=True):
            assert payload.content_type == "application/vnd.apache.parquet"
            pd.testing.assert_frame_equal(
                decode(payload), fake_yahoo.frame_for(symbol).reset_index()
            )

    def test_request_parameters_record_the_closed_interval_of_the_caller(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok"})
        result = provider.fetch(START, END, symbols)
        assert result.payloads[0].request_parameters == {
            "symbol": "AAA",
            "start": "2024-01-02",
            "end": "2024-01-04",
        }

    @pytest.mark.parametrize(
        ("start", "end", "yahoo_end"),
        [
            pytest.param(date(2024, 1, 2), date(2024, 1, 4), "2024-01-05", id="plain"),
            pytest.param(
                date(2023, 12, 1), date(2023, 12, 31), "2024-01-01", id="year end"
            ),
            pytest.param(
                date(2024, 2, 1), date(2024, 2, 28), "2024-02-29", id="leap year"
            ),
        ],
    )
    def test_asks_yahoo_for_unadjusted_daily_bars_with_actions_and_an_inclusive_end(
        self,
        provider: YahooFinanceProvider,
        fake_yahoo: FakeYahoo,
        start: date,
        end: date,
        yahoo_end: str,
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok"})
        provider.fetch(start, end, symbols)
        ((symbol, kwargs),) = fake_yahoo.history_calls
        assert symbol == "AAA"
        assert kwargs == {
            "start": start.isoformat(),
            "end": yahoo_end,
            "interval": "1d",
            "auto_adjust": False,
            "actions": True,
        }

    def test_received_at_is_utc_and_taken_while_fetching(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok", "BBB": "ok"})
        before = datetime.now(UTC)
        result = provider.fetch(START, END, symbols)
        after = datetime.now(UTC)
        assert len(result.payloads) == 2
        for payload in result.payloads:
            assert payload.received_at.utcoffset() == timedelta(0)
            assert before <= payload.received_at <= after

    def test_waits_the_declared_interval_before_every_request_but_the_first(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        interval = provider.capabilities().min_request_interval_s
        symbols = fake_yahoo.use({"AAA": "ok", "BBB": "ok", "CCC": "ok"})
        provider.fetch(START, END, symbols)
        assert [event[:2] for event in fake_yahoo.events] == [
            ("history", "AAA"),
            ("sleep", interval),
            ("history", "BBB"),
            ("sleep", interval),
            ("history", "CCC"),
        ]

    def test_a_single_symbol_never_sleeps(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok"})
        provider.fetch(START, END, symbols)
        assert fake_yahoo.sleeps == []


class TestPermanentFailures:
    @pytest.mark.parametrize("kind", ["tz_missing", "prices_missing"])
    def test_yfinance_symbol_errors_are_permanent_and_do_not_stop_the_loop(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo, kind: str
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok", "BBB": kind, "CCC": "ok"})
        result = provider.fetch(START, END, symbols)
        assert served(result) == ["AAA", "CCC"]
        assert failed(result) == [("BBB", PERMANENT)]
        assert result.symbol_errors[0].message == str(fake_yahoo.failure("BBB", kind))
        assert fake_yahoo.history_symbols == ["AAA", "BBB", "CCC"]

    def test_empty_response_is_permanent_and_does_not_stop_the_loop(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "empty", "BBB": "ok"})
        result = provider.fetch(START, END, symbols)

        assert served(result) == ["BBB"]
        assert failed(result) == [("AAA", PERMANENT)]
        assert "AAA" in result.symbol_errors[0].message


class TestFetchRateLimit:
    def test_marks_failing_and_remaining_symbols_temporary_then_stops(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        interval = provider.capabilities().min_request_interval_s
        symbols = fake_yahoo.use(
            {"AAA": "ok", "BBB": "rate_limit", "CCC": "ok", "DDD": "ok"}
        )
        result = provider.fetch(START, END, symbols)

        assert served(result) == ["AAA"]
        assert failed(result) == [
            ("BBB", TEMPORARY),
            ("CCC", TEMPORARY),
            ("DDD", TEMPORARY),
        ]
        first, *skipped = result.symbol_errors
        assert first.message == str(fake_yahoo.failure("BBB", "rate_limit"))
        assert all("not attempted" in e.message.lower() for e in skipped)
        assert fake_yahoo.history_symbols == ["AAA", "BBB"]
        assert fake_yahoo.sleeps == [interval]

    def test_on_the_last_symbol_there_is_nothing_left_to_skip(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use({"AAA": "ok", "BBB": "rate_limit"})
        result = provider.fetch(START, END, symbols)
        assert served(result) == ["AAA"]
        assert failed(result) == [("BBB", TEMPORARY)]


class TestProviderUnavailable:
    @pytest.mark.parametrize(
        ("script", "failures"),
        [
            pytest.param(
                {"AAA": "rate_limit", "BBB": "ok", "CCC": "ok"},
                3,
                id="limit on the first",
            ),
            pytest.param({"AAA": "rate_limit"}, 1, id="single symbol"),
        ],
    )
    def test_raises_when_nothing_served_and_every_failure_temporary(
        self,
        provider: YahooFinanceProvider,
        fake_yahoo: FakeYahoo,
        script: dict[str, str],
        failures: int,
    ) -> None:
        symbols = fake_yahoo.use(script)
        with pytest.raises(UnavailableProviderError, match=rf"yahoo.*all {failures} "):
            provider.fetch(START, END, symbols)

    def test_partial_success_with_a_permanent_error_returns_both(
        self, provider: YahooFinanceProvider, fake_yahoo: FakeYahoo
    ) -> None:
        symbols = fake_yahoo.use(
            {"AAA": "ok", "BBB": "tz_missing", "CCC": "rate_limit", "DDD": "ok"}
        )
        result = provider.fetch(START, END, symbols)
        assert served(result) == ["AAA"]
        assert failed(result) == [
            ("BBB", PERMANENT),
            ("CCC", TEMPORARY),
            ("DDD", TEMPORARY),
        ]

    @pytest.mark.parametrize(
        ("script", "n_errors"),
        [
            pytest.param(
                {"AAA": "tz_missing", "BBB": "rate_limit", "CCC": "ok"},
                3,
                id="permanent + temporary",
            ),
            pytest.param({"AAA": "tz_missing", "BBB": "empty"}, 2, id="all permanent"),
        ],
    )
    def test_nothing_served_but_a_permanent_error_returns_instead_of_raising(
        self,
        provider: YahooFinanceProvider,
        fake_yahoo: FakeYahoo,
        script: dict[str, str],
        n_errors: int,
    ) -> None:
        symbols = fake_yahoo.use(script)
        result = provider.fetch(START, END, symbols)

        assert result.payloads == ()
        assert len(result.symbol_errors) == n_errors
        assert any(e.error_type is PERMANENT for e in result.symbol_errors)
