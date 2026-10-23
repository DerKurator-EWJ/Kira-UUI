# UUI Core 1.0 — multimodales Importprotokoll (erste Integrationsstufe)

**Status:** Die vorhandenen Android-Quantizer sind in `Kira_v0_9_8_BasePackage_HotMount(5).zip` als Kotlin-Quellcode vorhanden. Diese Version integriert **nicht** die Android-Kamera oder das Mikrofon live und kompiliert keine Kotlin-Dateien. Sie implementiert den **empfangenden UUI-Adapter** für exportierte, bereits quantisierte Ereignisse.

## Herkunft und Schnittstelle

- `AdaptiveVisualQuantizer.kt`: erwartet **7** numerische visuelle Features und liefert u. a. eine Prototyp-ID.
- `AdaptiveAudioQuantizer.kt`: nutzt **6** Features und liefert u. a. eine Prototyp-ID.
- Die UUI-Importdaten müssen vom Android-Sender als JSONL exportiert werden. Die Datei `sensor_demo.jsonl` enthält **synthetische Testwerte**, keine realen Kamera-/Mikrofonmessungen.
- `event_id`, `source`, `sequence` sind explizite Herkunfts-/Ordnungsdaten. Prototypkennungen werden je `modality` **und** `source` getrennt; eine Zahl hat keine inhärente Bedeutung.
- `features` sind endliche Zahlen mit Länge 7 (visual) oder 6 (audio). Es findet keine unzulässige Rückübersetzung der Prototyp-ID in ein Objekt oder einen Begriff statt.

## Benutzung

```powershell
py -3.14 import_sensor.py sensor_demo.jsonl --database kira_uui.sqlite
py -3.14 -m unittest discover -v
```

**Neuer UUI-Befehlssatz** (jeweils JSON-Text, da UUI noch keine nativen Record-Typen hat):
- `sensor_rein(json_ereignis)`
- `sensor_paket(json_liste)`
- `sensor_nachsehen(ereignis_id)`

Der Import erzeugt persistente Ereignis- und Prototypknoten sowie ausdrücklich typisierte Relationen `sensor.produced`, `sensor.quantized_as` und (bei früheren Sequenzen derselben Quelle und Modalität) `sensor.precedes`. Bestehende UUI-0.9-Mustererkennung kann darauf arbeiten; Muster sind **keine** automatisch erkannte Objektidentität oder semantische Erkenntnis.

**Einschränkungen:** Noch kein Android-Sendeadapter, keine laufende Stream-Verbindung, kein Rohdatenarchiv, keine Live-Audio-/Video-Pipeline, keine übergreifende Objektidentität. Die Sequenzrelation belegt Reihenfolge der eingetroffenen Daten, keine Kausalität. Import ist pro Ereignis atomar; ein Batch ist nicht insgesamt atomar. Bei Fehlern bleiben bereits erfolgreich importierte Ereignisse bestehen. Die `metadata`-Felder sind vom Sender behauptete Angaben, nicht unabhängig verifiziert.
