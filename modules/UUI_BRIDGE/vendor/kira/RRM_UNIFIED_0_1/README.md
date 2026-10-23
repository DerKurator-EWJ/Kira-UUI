# RRM UNIFIED 0.1 — integrierte relationale Runtime

## Inhalt

* `rrm_id.py`: originale hybride 8-Bit-Kopf-/variable-Nutzdaten-Kodierung (RRM-ID 0.1).
* `runtime.py`: originale persistente relationale Runtime (RRM-RUNTIME 0.2), inklusive bisherigem Evidenzpfad.
* `binary_ops.py`: originale referenzbasierte Operationen `RELATE`, `DIFF`, `CONTINUE`.
* `rrm05.py`: originaler Evidenz-/Kontextalgorithmus.
* `unified.py`: **neue Integration**: `UnifiedRuntime` verbindet die bestehenden Operationen mit einer begrenzten LRU-Cache-Ebene über derselben SQLite-Record-Arena. Die stabilen SQLite-Handles bleiben unabhängig vom Aufenthaltsort.
* `test_unified.py`: Integrations-/Persistenztests.
* `benchmark.py`: reproduzierbarer kleiner Vergleich.

## Start (Python 3.10+, nur Standardbibliothek)

```bash
python -m unittest discover -v
python benchmark.py
```

## Beispiel

```python
from unified import UnifiedRuntime

rt = UnifiedRuntime('wissen.sqlite', cache_budget=64 * 1024)
a = {'nodes': {'a':'point', 'b':'point'}, 'edges': [['a','near','b']]}
b = {'nodes': {'a':'point', 'b':'point'}, 'edges': [['a','far','b']]}
old, new = rt.store_state(a), rt.store_state(b)
delta = rt.ops.diff(old, new)
assert rt.load_state(rt.ops.continue_with(old, delta[0])) == b
print(rt.cache_stats())
rt.close()
```

## Tatsächliche Integration und Grenzen

* **Ein SQLite-Speicher** für binäre Records, Beobachtungen und Evidenz; **ein logischer Handle-Raum** für direkte Binäroperationen und persistentes Lernen.
* `_get` liest zuerst aus einem **byte-budgetierten LRU-Wire-Cache**, sonst über SQLite. `_put` verwendet denselben Interning-Mechanismus und erwärmt den Cache. Kein zweiter SQLite-Datenbestand.
* Die bestehenden Lern-/Vorhersagefunktionen benutzen weiterhin JSON-Signaturen und rekonstruieren bei `observe` und `predict` strukturierte Zustände. **Nicht** vollständig binärer Lernpfad.
* `DIFF` benötigt unveränderte Knotenzuordnung und identische Kantenreihenfolge; keine allgemeine Graphisomorphie.
* Cachebudget zählt nur Nutzbytes der Wire-Records, nicht Python-Objekt-/Index-Overhead, SQLite-Pages oder Betriebssystemcache.
* `:memory:` bezeichnet eine SQLite-Datenbank im RAM; Dateipfade bieten Persistenz. Kein explizites GPU/VRAM- oder SSD-Placement.
* Cache enthält auch frisch eingefügte Records; bei externem Datenbank-Rollback oder Manipulation durch andere Verbindungen muss die Runtime geschlossen/neu geöffnet werden. Für Mehrprozessbetrieb fehlen Kohärenz- und Transaktionsinvalidierung.
* Benchmarks sind kleine synthetische CPU/SQLite-Messungen und **kein** Nachweis einer überlegenen universellen Architektur.

## Nächster Prüfpunkt

Binäre Evidenzschlüssel statt JSON, verlässliche Transaktions-/Cachekohärenz, echte Workloads mit größeren eindeutigen Recordmengen und Cold/Warm-Messungen.
