# Equinox Forge v0.48 — Specific-to-General Numeric Runtime

This benchmark freezes the first DB-independent runtime over the canonical numeric ID space.
Resolution starts at the queried concept ID and climbs configured parent relations only when
no sufficiently strong local fact exists. Specific valid facts therefore outrank more general
facts categorically; unresolved same-depth conflicts abstain.
