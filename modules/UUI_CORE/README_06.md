# UUI Core 0.6 – Persistenz und Provenienz

Basiert auf UUI Core 0.5. Python-Interpreter; noch keine native UUI-VM.

## Änderungen

- `run_uui.py` verwendet standardmäßig die Datei `kira_uui.sqlite` statt `:memory:`. Mit `--database :memory:` bleibt der bisherige flüchtige Modus verfügbar.
- `lern_draus(subject,predicate,value,source_ref,event_id[,source_type[,confidence]])` akzeptiert nun explizite Quellenarten `observation`, `action_result`, `external_claim`, `inference`, `premise` und optionale rationale Konfidenz von 0 bis 1.
- `such_widerspruch()` zeigt offene Konfliktprüfungen (JSON-Text).
- `pruef_dich(subject,predicate)` liefert eine konservative Zusammenfassung und **keinen** Wahrheitsbeweis (JSON-Text).
- Tests prüfen Persistenz über einen vollständigen Neustart, Provenienz, Konflikte und ungültige Eingaben.

## Windows / Python 3.14

```powershell
py -3.14 -m unittest discover -v
py -3.14 run_uui.py persistent_demo.uui --database kira_uui.sqlite
```

**Achtung:** `persistent_demo.uui` verwendet eine feste Ereignis-ID und schlägt beim zweiten Start absichtlich mit `duplicate evidence event_id` fehl. Das ist ein Schutz gegen doppeltes Eintragen. Für reine Abfragen ohne neue Ereignisse eine zweite `.uui`-Datei mit `was_weisst_du(...)` verwenden.

## Grenzen

Quellenarten und Konfidenzen sind Angaben des Aufrufers, nicht unabhängig verifiziert. Widersprüche werden nur bei gleichem Subjekt, Prädikat und Kontext und unterschiedlichen Werten als Prüffall angelegt. Kein automatisches Wahrheitsurteil, keine selbständige Modellrevision, keine echte Bytecode-VM. Die SQLite-Datenbank enthält potenziell sensible Angaben und wird **nicht verschlüsselt**. Für echte Sensordaten müssen vertrauenswürdige Adapter die Herkunft belegen.
