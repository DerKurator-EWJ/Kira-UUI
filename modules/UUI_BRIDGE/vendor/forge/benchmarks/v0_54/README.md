# v0.54 Unified Numeric Relation Routing

Frozen contract for replacing runtime-side relation-family special cases with one compiled numeric `query_relation_id -> execution_relation_id` routing table. Family IDs, scalar predicates, relation facets, and recursive relation-macro IDs are all queried through one `resolve(subject_id, relation_id)` API.
