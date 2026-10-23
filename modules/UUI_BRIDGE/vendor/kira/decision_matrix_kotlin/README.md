# KIRA Decision & Confirmation Matrix 0.1

Independent Kotlin/JVM evaluation component. Intended for `kira-core`, compatible with Android Kotlin (no Android dependencies). It **does not** alter any existing Kira application: merge source and call from the existing event pipeline after obtaining real predictions and outcomes.

## Integration
1. Copy `src/DecisionMatrix.kt` into the `kira-core` Kotlin source tree (package `kira.core.evaluation`).
2. Map your prediction and observation event fields into `EvaluationInput`. Never manufacture confidence, surprise, valence or information gain from reaction labels.
3. Publish `Evaluation` as an evaluation event (with original evidence reference, source, correlation ID and event ID), and keep storage/writes in the existing evidence subsystem. Don't allow a UI utterance to trigger trading, robotics, or other external actions.
4. Route `primary.utterance` to HoKo or UI if enabled; preserve `signals` and numeric evidence for debugging. Avoid vocalizing every event.
5. Use `evidenceReviewRequired` to open a review task, not to erase prior knowledge. `investigatePriority`/`repeatPriority` are heuristic proposal scores, not autonomous action authorization.

## Testing
`kotlinc src/DecisionMatrix.kt tests/MatrixSmoke.kt -include-runtime -d matrix-test.jar && java -jar matrix-test.jar`

## Status / limitations
The 13 user-selected names are included. Scoring and precedence are preliminary heuristics; no calibrated affect model or independently learned preferences. Curiosity/inquiry is separate from surprise (as specified in Kira Masterplan). `GUILTY` is a recurrence-of-contradiction label, **not** an accusation about a person. `CANTOR` is an epistemic anomaly label, **not** proof of impossibility. `FUCK` doesn't suppress information gain. This package is an integration-ready module, not a verified integration into the complete app.
