# KIRA NUMERIC CORE 0.6 — Kostenorientierter Speicherpfad

Neuer `numeric_cost_aware.py` als experimenteller, alternativer Dispatcher.

- Deferred: Einmalige Konsolidierung nach neuen Beobachtungen, danach O(log n)-ähnlicher indizierter Zugriff auf den letzten Wert statt O(n)-History-Scan pro Vorhersage. SQLite-B-Tree, reale I/O-Kosten variieren.
- Incremental: Bestehender v0.4-Pfad wird nur bei geschätztem Vorteil aktiviert. Umschaltkosten und zusätzliche Schreibarbeit sind Teil einer expliziten Heuristik. Keine Behauptung optimaler Entscheidungen.
- Periodenprüfung bleibt auf max_period begrenzt, rein rationale Zahlen, keine geometrischen oder graphischen Grundelemente.
- Persistente Modi und Abfragezähler. Kein vollständiger History-Scan bei unveränderten wiederholten Vorhersagen.

**Einschränkung:** Das Kostenmodell verwendet einstellbare relative Einheiten und eine grobe zukünftige Zugriffsschätzung; es ist noch nicht aus Hardwaremessungen kalibriert. Der inkrementelle Altpfad führt bei Initialisierung eine Migration vorhandener Serien aus, weshalb er nur verzögert initialisiert wird. Keine Garantie auf global subquadratische Kosten über alle möglichen Aufgaben.

Ausführen: `python -m unittest discover -p 'test_*.py'`.

## Gemessener lokaler Vergleich

Siehe `benchmark_v06_results.json`: jeweils drei Wiederholungen, Median, gleiche Eingabefolge und SQLite-Dateien. Bei 1000 Beobachtungen/200 Abfragen: v0.5 Lernen 56,55 ms, Abfragen 41,21 ms; v0.6 Lernen 110,15 ms, Abfragen 26,66 ms. Zusammen ist v0.6 **langsamer** (136,81 vs. 97,76 ms). Bei 100 Beobachtungen/100 Abfragen ist v0.6 ebenfalls langsamer. Die neue Heuristik ist somit **nicht als Standardpfad freigegeben**. Mehrere SQLite-Verbindungen/Transaktionen erzeugen Zusatzkosten; der Kostenschätzer ist nicht kalibriert. Die Verbesserung bei vielen Abfragen stammt vom gezielten letzten-Wert-Zugriff, nicht vom automatischen Umschalten.

Testkommando mit Abhängigkeit: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py'`.
