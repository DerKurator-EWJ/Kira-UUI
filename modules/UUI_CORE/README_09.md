# UUI Core 0.9 – Semantikneutrale Muster und Makros

Neue Operationen: `find_muster(2)`, `bau_makro(2)`, `zeig_makro("pattern:...")`.

Erkannt werden **ausschließlich exakt typgleiche, gerichtete Zwei-Relations-Ketten** `A -T1-> B -T2-> C` mit verschiedenen Endknoten. Mehrere Instanzen gleicher Typfolge ergeben einen Musterkandidaten. Identifikatoren sind opak; Labels werden nicht ausgewertet. Das ist **keine allgemeine Graph-Isomorphie, keine Bedeutungsinduktion und keine automatische Generalisierung**. Auch die gleiche Typfolge kann zufällig vorkommen. Ein Makro ist eine persistierte Zusammenfassung mit nachvollziehbaren Ursprungsrelationen, **kein Ersatz für diese Relationen**.

`bau_makro` aktualisiert den gespeicherten Snapshot der aktuellen aktiven Instanzen. `zeig_makro` zeigt gespeicherte Instanzen und deren aktuellen Aktivitätsstatus; Revisionen löschen die Herkunft nicht. Keine automatische semantische Interpretation, keine Video-/Audioanbindung, keine eigenständige Lern- oder Wahrheitsentscheidung. `.uuix` bleibt die bisherige JSON-AST-Zwischendarstellung, nicht nativer Bytecode.

Test: `py -3.14 -m unittest discover -v`; Demo: `py -3.14 run_uui.py pattern_demo.uui --database :memory:`.

Für UUI 1.0 (Audio/Video) benötigen wir die **letzte nachweislich lauffähige Kira-App-Version mit Audio-/Videoquellen** sowie gegebenenfalls ihre separaten nativen Module. Forge ist dafür aktuell nicht erforderlich; die vorhandene Bridge bleibt enthalten.
