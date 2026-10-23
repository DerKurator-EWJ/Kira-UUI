# v0.41 — Active Probe Selection / Information-Gain Proxy

v0.41 is a pool-based acquisition benchmark on the fixed v0.39 neural animal-domain teacher.
It does **not** change RuntimeCognition (runtime remains v0.40) and it does not generate arbitrary
natural-language questions. A known pool of entity/relation prompts is available, but their answers
are hidden from the selector.

The selector:

1. asks one rotating relation per entity to create a broad 32-query seed;
2. compiles the seed BASE;
3. scores every remaining prompt using only current Equinox runtime uncertainty/abstention;
4. spends the remaining 32-query budget on the highest-uncertainty frontier with coverage tie-breaks;
5. recompiles using a teacher-query cache so seed re-use does not count as new teacher queries.

Frozen full-domain release setting: 32 entities, 128 core facts, 64 unique teacher queries, 5 seeded-random control trials.
