# UUI Core 0.8 – semantikneutrale relationale Revision

Neue Operationen: `knoten(id)`, `benenne(id,label)`, `relation(source,type,target,relation_id)`, `haengt_ab_von(prerequisite_relation_id,dependent_relation_id)`, `folgen_von(relation_id)`, `revidiere_relation(relation_id,reason)`, `pruef_relation(relation_id)`.

* Kennungen und Relationstypen sind **opaque**. Numerische Nähe, Wortähnlichkeit und optionale Labels begründen keine Semantik.
* Knotenbeziehungen und epistemische Abhängigkeiten sind **getrennt**. Nur explizit gespeicherte Abhängigkeiten propagieren Revisionsbedarf.
* Eine Revision setzt die gewählte Relation auf `superseded`, abhängige aktive Relationen auf `needs_review`. Sie erklärt diese **nicht automatisch für falsch**.
* Relationen, Abhängigkeiten und Revisionsprotokolle liegen in derselben SQLite-Datei wie die bestehende Kira-Evidenz. Kein automatisches Löschen.
* UUI gibt strukturierte Daten derzeit als JSON-Text zurück. Keine autonome Semantikbildung, keine selbstständige Wahrheitsermittlung.
* `benenne` ändert lediglich ein optionales Anzeigeetikett; dies ist kein Lernbeweis.

Test: `python -m unittest discover -v`
Demo: `python run_uui.py neutral_demo.uui --database :memory:`

Hinweis: `neutral_demo.uui` mit einer bereits gefüllten persistenten Datenbank erneut auszuführen verursacht bewusst einen Fehler bei doppelten Relationskennungen. Für wiederholte Demos `--database :memory:` verwenden.

## Willi-Namen (0.8.1)

Die bevorzugten UUI-Aufrufe heißen nun `neues_ding`, `gib_name`, `kleber`, `das_und_das`, `passiert_wenn`, `weg_radieren` und `guck_nach`. Die bisherigen Namen funktionieren als Kompatibilitäts-Aliase weiterhin. `weg_radieren` ist **keine physische Löschung**: Die Relation wird als überholt markiert und ihre Historie bleibt bestehen.
