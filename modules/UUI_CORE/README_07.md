# UUI Core 0.7 – kontrollierte Evidenzrevision

Aufbauend auf 0.6: drei Bibliotheksfunktionen, **keine neuen Syntax-Schlüsselwörter**:

- `pruef_revision(review_id)`: liefert eine konservative Prüfvorlage, Quellenarten, Konfidenzen und mögliche Entscheidungen. Schlägt grundsätzlich `retain_both` vor; keine automatisierte Wahrheitsfeststellung.
- `entscheide_revision(review_id, decision, reason)`: explizite, begründete Entscheidung; Optionen `retain_both`, `supersede_old`, `reject_new`. Keine Evidenz wird gelöscht.
- `revision_verlauf(review_id)`: dauerhaft gespeicherte Entscheidung und Begründung.

Review-ID abrufen: `such_widerspruch()`. Da UUI noch keine strukturierten Werte besitzt, werden die Ergebnisse als JSON-Text zurückgegeben.

**Grenzen:** Keine automatische Quellenvalidierung, keine Kalibrierung von Konfidenzen, keine generalisierte Modellrevision und kein eigenes maschinelles Lernen. Die vorhandene Python-Runtime und das JSON-basierte `.uuix`-Format bleiben bestehen.

Tests: `py -3.14 -m unittest discover -v`
