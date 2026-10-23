# UUI Core 0.1 — erster ausführbarer Sprachprototyp

Dies ist eine **eigene minimale Syntax mit Lexer, Parser und Interpreter**, implementiert in Python. **Keine** plattformneutrale VM, kein Bytecode und keine eigenständige native Programmiersprache im Sinne einer von Python unabhängigen Runtime.

## Ausführen

Python 3.10+ (Standardbibliothek; zusätzliche vendorte Abhängigkeiten können für Forge-Pakete erforderlich sein):

```bash
cd UUI_CORE_0_1
python run_uui.py hello.uui
python -m unittest -v test_uui.py test_bridge.py
```

Optional: `--database kira.sqlite` für persistente numerische Beobachtungen; `--forge-pack PATH` für ein kompatibles Forge-Kernel-Paket.

## UUI 0.1

- `let name = expression` — Variable
- `print expression` — Ausgabe
- `observe expression as "source"` — rationalen Wert im Kira Numeric Core erfassen
- `resolve expression, expression` — Forge-Abfrage, nur mit geladenem Pack
- Exakte rationale Arithmetik `+ - * /`, Klammern, Dezimalzahlen, Strings, Kommentare `#`.
- Eine Anweisung pro Zeile oder mit `;` getrennt.
- Keine Schleifen, Bedingungen, Funktionen, Datei-/Netzwerkzugriffe oder dynamische Host-Ausführung.

## Grenzen

Interpreter, Bridge und Vendor-Code benötigen weiterhin Python. Keine UUIX-Datei, keine JVM-/Android-Portierung. Der Forge-Pfad ist nur ein Adapter und nicht durch einen echten End-to-End-Pack-Test validiert. Die Sprachumgebung ist **keine Sicherheits-Sandbox für feindlichen Code** (große Zahlen können Ressourcen verbrauchen); keine Ausführung unzuverlässiger Programme anbieten. Die Begrenzung von Token/Anweisungen ist nur eine erste Schutzmaßnahme.

Die bisherigen Bridge- und Vendor-Dateien bleiben erhalten, damit sich die Weiterentwicklung gegen die ursprünglichen Kira-/Forge-Module prüfen lässt.
