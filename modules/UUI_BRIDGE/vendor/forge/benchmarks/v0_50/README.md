# v0.50 Teacher → Numeric Compiler Trial

This benchmark freezes the first end-to-end execution of the v0.44–v0.49 numeric architecture from a computed neural teacher into a DB-free runtime pack.

The benchmark deliberately preserves the negative macro-routing result: generic `PROPERTY` and `ACTION` are multi-valued in the teacher-derived hierarchy, while the current numeric runtime API is scalar. Therefore the v0.49 safety validator emits no ambiguous macro shortcut. All extracted relations remain present in the numeric pack.

No external pretrained Hugging Face runtime result is claimed by this reference.
