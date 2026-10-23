# v0.32 Support-Gated Prediction

This benchmark reuses the frozen v0.31 gauntlet cases without modification.

The v0.32 intervention changes one semantic rule: associative recall may expose predictive contexts, but a recalled predictive context may execute only when it has sufficient live-context support. Unknown live tokens cannot be generalized over by lexical recall alone; they require an explicit compiled context/frame/prototype/hierarchy admission path.

The key adversarial pair is:

- `Wolf ist ein`
- `Stein ist ein`

In the generic v0.31 ladder pack both contain one unknown entity and the same known `ist ein` suffix. Without independent semantic evidence, the runtime cannot justify treating one as a valid taxonomy transfer and the other as unsupported. v0.32 therefore abstains on both in the lexical-only ladder, while the separately typed-frame track continues to admit `Wolf`/`Fuchs` and reject `Stein` using independent semantic evidence.
