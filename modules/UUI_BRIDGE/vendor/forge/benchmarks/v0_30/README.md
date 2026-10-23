# v0.30 frozen benchmark contract

This directory is the stable evaluation contract introduced at the v0.30 engineering freeze.

The contract freezes **metric meanings**, not the current scores. Later milestones may append datasets, cases and metrics, but should not redefine an existing metric or silently remove a failing case.

A benchmark result should record enough provenance to reproduce the comparison: Equinox version, runtime policy, dataset hash, `.eqbase` hash, source/teacher identity and relevant compile policy.

The first post-freeze target is a bounded real-teacher domain. The same domain must be evaluated as: teacher reference, direct-replay control, full Equinox runtime and leave-one-mechanism-out ablations.
