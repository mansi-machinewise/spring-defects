from __future__ import annotations

from src.models.schemas import Failure, ModuleResult, ModuleState


class DecisionEngine:
    def __init__(self, config: dict) -> None:
        self.config = config

    def decide(
        self,
        results: list[ModuleResult],
    ) -> tuple[ModuleState, list[Failure]]:
        failures: list[Failure] = []

        for result in results:
            # ---------------------------------------------------------
            # SKIPPED MODULE
            # ---------------------------------------------------------
            if result.state == ModuleState.SKIPPED:
                if self.config["inspection"]["require_all_modules"]:
                    failures.append(
                        Failure(
                            module=result.module,
                            reason="Required inspection module was skipped",
                        )
                    )
                continue

            # ---------------------------------------------------------
            # ERROR
            # ---------------------------------------------------------
            # A real module execution error should still fail the
            # inspection, including PatchCore execution errors.
            # ---------------------------------------------------------
            if result.state == ModuleState.ERROR:
                failures.append(
                    Failure(
                        module=result.module,
                        reason=result.message or "Module execution error",
                    )
                )
                continue

            # ---------------------------------------------------------
            # PATCHCORE — LOCALIZATION ONLY
            # ---------------------------------------------------------
            # Module 1 may report FAIL because an anomaly was found,
            # but that anomaly must NOT determine the final decision.
            #
            # PatchCore is used only for:
            #   - anomaly scores
            #   - regional localization
            #   - anomaly heatmaps
            # ---------------------------------------------------------
            if result.module == 1:
                continue

            # ---------------------------------------------------------
            # OTHER MODULE FAILURES
            # ---------------------------------------------------------
            if result.state == ModuleState.FAIL:
                self._add_module_failures(result, failures)

        return (
            (ModuleState.FAIL, failures)
            if failures
            else (ModuleState.PASS, failures)
        )

    @staticmethod
    def _add_module_failures(
        result: ModuleResult,
        failures: list[Failure],
    ) -> None:

        # -------------------------------------------------------------
        # Module 0 — YOLO Detection
        # -------------------------------------------------------------
        if result.module == 0:
            failures.append(
                Failure(
                    module=0,
                    reason=(
                        result.message
                        or "Spring was not confirmed in both camera feeds."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Module 1 — PatchCore
        #
        # IMPORTANT:
        # PatchCore FAIL is intentionally NOT handled here.
        # It is localization only.
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # Module 2 — Front Pose
        # -------------------------------------------------------------
        elif result.module == 2:
            for name in result.details.get("tolerance_failures", []):
                failures.append(
                    Failure(
                        module=2,
                        reason=f"{name} out of tolerance",
                        value=result.metrics.get(name),
                    )
                )

        # -------------------------------------------------------------
        # Module 3 — Top Pose
        # -------------------------------------------------------------
        elif result.module == 3:
            if result.metrics.get("hook_alignment_error", 0) > result.details.get(
                "threshold",
                float("inf"),
            ):
                failures.append(
                    Failure(
                        module=3,
                        reason="Hook angle mismatch",
                        value=result.metrics.get("hook_alignment_error"),
                    )
                )

        # -------------------------------------------------------------
        # Module 4 — Classifier
        # -------------------------------------------------------------
        elif result.module == 4:
            prediction = result.details.get(
                "prediction",
                "UNKNOWN",
            )

            decision = result.details.get(
                "decision",
                "UNKNOWN",
            )

            confidence = result.metrics.get(
                "classifier_confidence",
            )

            if decision == "REJECT":
                failures.append(
                    Failure(
                        module=4,
                        reason=f"Classifier detected defective spring: {prediction}",
                        value=confidence,
                    )
                )

            elif decision == "UNCERTAIN":
                failures.append(
                    Failure(
                        module=4,
                        reason="Classifier confidence below required threshold",
                        value=confidence,
                    )
                )

            else:
                failures.append(
                    Failure(
                        module=4,
                        reason=result.message or "Classifier rejected spring",
                        value=confidence,
                    )
                )