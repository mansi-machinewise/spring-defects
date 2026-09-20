from __future__ import annotations

from src.models.schemas import Failure, ModuleResult, ModuleState


class DecisionEngine:
    def __init__(self, config: dict) -> None: self.config = config

    def decide(self, results: list[ModuleResult]) -> tuple[ModuleState, list[Failure]]:
        failures: list[Failure] = []
        for result in results:
            if result.state == ModuleState.SKIPPED:
                if self.config["inspection"]["require_all_modules"]:
                    failures.append(Failure(module=result.module, reason="Required inspection module was skipped"))
                continue
            if result.state == ModuleState.ERROR:
                failures.append(Failure(module=result.module, reason=result.message or "Module error"))
            elif result.state == ModuleState.FAIL:
                self._add_module_failures(result, failures)
        return (ModuleState.FAIL, failures) if failures else (ModuleState.PASS, failures)

    @staticmethod
    def _add_module_failures(result: ModuleResult, failures: list[Failure]) -> None:
        if result.module == 0:
            failures.append(Failure(module=0, reason=result.message or "Spring was not confirmed in both camera feeds."))
        elif result.module == 1:
            failures.append(Failure(module=1, reason="Visual anomaly detected", value=result.metrics.get("anomaly_score")))
        elif result.module == 2:
            for name in result.details.get("tolerance_failures", []):
                failures.append(Failure(module=2, reason=f"{name} out of tolerance", value=result.metrics.get(name)))
        elif result.module == 3:
            if result.metrics.get("hook_alignment_error", 0) > result.details.get("threshold", float("inf")):
                failures.append(Failure(module=3, reason="Hook angle mismatch", value=result.metrics.get("hook_alignment_error")))
