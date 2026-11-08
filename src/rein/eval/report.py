from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rein.eval import constants as c
from rein.eval.metrics import METRICS, Delta, PredictionRow, paired_bootstrap, summarize

SliceFn = Callable[[PredictionRow], bool]

DEFAULT_SLICES: dict[str, SliceFn] = {
    c.SLICE_ALL: lambda row: True,
    c.SLICE_UNSEEN_RULES: lambda row: bool(row.tags.get(c.TAG_UNSEEN_RULE)),
}
DEFAULT_TABLE_METRICS = ("n", "accuracy", "ece", "brier")


@dataclass(frozen=True)
class Report:
    models: tuple[str, ...]
    slices: tuple[str, ...]
    cells: Mapping[tuple[str, str], Mapping[str, float]]
    rows: Mapping[str, tuple[PredictionRow, ...]]
    slice_fns: Mapping[str, SliceFn]

    def cell(self, slice_name: str, model: str) -> Mapping[str, float]:
        return self.cells[(slice_name, model)]

    def slice_rows(self, slice_name: str, model: str) -> list[PredictionRow]:
        keep = self.slice_fns[slice_name]
        return [row for row in self.rows[model] if keep(row)]

    def paired_delta(self, slice_name: str, model_a: str, model_b: str, metric: str = "accuracy") -> Delta:
        return paired_bootstrap(
            self.slice_rows(slice_name, model_a), self.slice_rows(slice_name, model_b), METRICS[metric]
        )

    def to_markdown(self, metrics: Sequence[str] = DEFAULT_TABLE_METRICS) -> str:
        header = "| slice | model | " + " | ".join(metrics) + " |"
        divider = "|" + "---|" * (len(metrics) + 2)
        lines = [header, divider]
        for slice_name in self.slices:
            for model in self.models:
                values = self.cell(slice_name, model)
                lines.append(
                    f"| {slice_name} | {model} | " + " | ".join(f"{values[name]:.4g}" for name in metrics) + " |"
                )
        return "\n".join(lines)


def build_report(
    predictions: Mapping[str, Sequence[PredictionRow]], slices: Mapping[str, SliceFn] | None = None
) -> Report:
    if not predictions:
        raise ValueError("no model predictions to report")
    slice_fns = dict(slices or DEFAULT_SLICES)
    models = tuple(predictions)
    reference = {row.item for row in predictions[models[0]]}
    for model in models:
        items = [row.item for row in predictions[model]]
        if len(items) != len(set(items)):
            raise ValueError(f"model {model!r} has duplicate items")
        if set(items) != reference:
            raise ValueError("every model must be scored on the same items")
    rows = {model: tuple(predictions[model]) for model in models}
    cells = {
        (slice_name, model): summarize([row for row in rows[model] if keep(row)])
        for slice_name, keep in slice_fns.items()
        for model in models
    }
    return Report(models=models, slices=tuple(slice_fns), cells=cells, rows=rows, slice_fns=slice_fns)
