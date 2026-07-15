"""CPU analysis workers for potentially expensive global processing."""

from __future__ import annotations

from threading import Event

from PySide6.QtCore import QThread, Signal

from camlabel3d.core.processing import OutlierScope, ProcessingContext, ProcessingEngine


class OutlierAnalysisWorker(QThread):
    analysisCompleted = Signal(object)
    analysisFailed = Signal(str)

    def __init__(
        self,
        engine: ProcessingEngine,
        context: ProcessingContext,
        enabled_rule_ids: list[str],
        params_by_rule: dict[str, dict[str, float | int]],
        generation: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.context = context
        self.enabled_rule_ids = list(enabled_rule_ids)
        self.params_by_rule = {
            rule_id: dict(params) for rule_id, params in params_by_rule.items()
        }
        self.generation = int(generation)
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            hits = self.engine.analyze_outliers(
                records=self.context.records,
                scope=OutlierScope.GLOBAL,
                enabled_rule_ids=self.enabled_rule_ids,
                params_by_rule=self.params_by_rule,
                context=self.context,
                should_cancel=self._cancel_event.is_set,
            )
            self.analysisCompleted.emit(
                {
                    "generation": self.generation,
                    "hits": hits,
                    "canceled": self._cancel_event.is_set(),
                }
            )
        except Exception as exc:
            self.analysisFailed.emit(str(exc))
