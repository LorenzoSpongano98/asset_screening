import enum

"""Table schemas, ISO code patterns and schema validation."""


class RawTableNames(enum.StrEnum):
    PRICES = "prices"
    CORPORATE_ACTIONS = "corporate_actions"
    FX = "fx"
    REFERENCE_SNAPSHOTS = "reference_snapshots"
