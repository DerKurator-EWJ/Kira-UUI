# Equinox Forge v0.49 — Macro-Aware Numeric Routing

This benchmark freezes the first runtime use of recursive macro IDs as validated routing shortcuts.
The v0.48 specific-to-general resolver remains the semantic reference. A macro shortcut is exported
only when it reproduces the v0.48 answer for the same concept/predicate at compile time. Runtime
therefore saves hierarchy traversal without granting the macro layer new semantic authority.
