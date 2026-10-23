# KIRA NUMERIC CORE 0.5 — adaptiver Versuch

Die Version ergänzt den bestehenden v0.4-Kern um `numeric_adaptive.py`. Pro numerischer Folge wird persistent zwischen `deferred` (v0.3: günstige Schreiboperationen, Konsolidierung bei Bedarf) und `incremental` (v0.4: laufende Musterpflege) unterschieden. Ab `promote_reads` Vorhersageabrufen wird eine Folge einmalig durch Replay der vorhandenen Beobachtungen in den inkrementellen Modus überführt. Der Modus bleibt nach Neustart erhalten. Alle Werte sind exakte rationale Zahlen; keine fundamentalen Graphen oder räumlichen Koordinaten.

## Test

`PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q`

24 Tests bestanden (einschließlich 2 neuer adaptiver Tests). Bereits bestehende ResourceWarnings können bei anderer Testausführung auftreten.

## Benchmark

`python benchmark_adaptive.py`

Bei 1.000 numerischen Änderungen und 200 Vorhersagen (ein lokaler Durchlauf):

| Modus | Lernen | Vorhersagen gesamt |
|---|---:|---:|
| v0.3 deferred | 53,6 ms | 191,4 ms |
| v0.4 incremental | 88,0 ms | 2,4 ms |
| v0.5 adaptive | 58,4 ms | 34,9 ms |

**Wichtig:** Der adaptive Modus ist im Test schlechter als der inkrementelle Modus bei vielen Lesezugriffen. Die Umstellung erfordert einen einmaligen Replay, und die ersten Abrufe erfolgen noch über den langsamen Pfad. Der Grenzwert `promote_reads=8` ist eine Heuristik, keine gemessene Kostenoptimierung. Diese Version ist ein überprüfbarer Zwischenschritt, kein allgemeiner Performance-Sieg. Benchmarks sind nicht statistisch abgesichert und enthalten SQLite-/Python-Overhead.

## Grenzen

- Die Moduswahl nutzt bislang eine feste Abrufschwelle, keine adaptive Kostenmessung und keine Rückstufung.
- Eine gemeinsame SQLite-Datei enthält die Rohbeobachtungen; Makros ersetzen diese nicht.
- Die v0.3-Vorhersage lädt die Folge weiterhin vollständig; nach Promotion wird dieser Aufwand vermieden.
- Bei Änderungen der Konfiguration `max_period` zwischen Neustarts ist keine automatische Migration implementiert.
- Es handelt sich um periodische Differenzmuster, nicht um allgemeine Weltmodellbildung oder universelles One-Shot-Lernen.
- Die vorhandenen älteren RRM-/Graphpfade sind als Kompatibilitätsbestand im Paket, nicht numerisch migriert.

Nächster sinnvoller Schritt: Umschaltpunkt aus gemessenen marginalen Schreib-/Lese-/Replaykosten bestimmen, danach gegen beide statischen Pfade mit unterschiedlichen Lese-/Schreibverhältnissen testen.
