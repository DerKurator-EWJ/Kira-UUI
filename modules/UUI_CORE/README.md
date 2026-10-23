# UUI Core 0.4 – Kira-spezifische Befehlsfamilien

Basiert auf UUI Core 0.3. Enthält **13 Familien mit 39 registrierten Namen** in `uui_cognition.py`.

**14 Operationen** haben eine begrenzte, explizit dokumentierte Implementierung. Die übrigen **25** melden beim Aufruf einen klaren Fehler statt eine Fähigkeit vorzutäuschen. Die Fragenfunktionen geben nur Fragevorlagen zurück, keine KI-generierten Erklärungen. `ist_das_noch` prüft ausschließlich Wertgleichheit, keine visuelle Identität. `wer_darf` verweigert standardmäßig. `merk_dir` speichert nur rationale Zahlen über den vorhandenen Kira-Numeric-Core. `erinner_dich` liest dessen numerische IDs. `bist_du_sicher` bestätigt keine ungeprüfte Evidenz. Es gibt noch keine Kamera-, Audio-, Robotik-, Simulations- oder autonome Lernimplementierung.

Der UUI-Parser, die Willi-Syntax, der JSON-basierte `.uuix`-Container und die Python-basierte Runtime sind unverändert. **Keine native VM oder binärer Bytecode.**

## Start (Python 3.14)

```powershell
py -3.14 run_uui.py cognition_demo.uui
py -3.14 -m unittest discover -v
```

`uui_cognition.py` dokumentiert die 39 Befehle. Nicht implementierte Operationen sind reservierte Namen, keine behaupteten Fähigkeiten. Benutzerfunktionen können sie nicht überschreiben. Alle bestehenden Tests sollten weiterlaufen.
