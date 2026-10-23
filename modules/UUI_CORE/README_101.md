# UUI Core 1.0.1 – Willi-Sensornamen

Neue Aliasnamen (alte Namen bleiben gültig):

- `auge_auf` → `sensor_rein` (ein JSON-Ereignis)
- `nase_schnueffeln` → `sensor_paket` (JSON-Liste von Ereignissen)
- `was_war` → `sensor_nachsehen` (Ereignis-ID)

Die UUI-Quellsyntax verwendet ASCII: `nase_schnueffeln` statt `nase_schnüffeln`, entsprechend `pruef` statt `prüf`. Die Bezeichnung `nase_schnueffeln` verarbeitet **nicht ausschließlich Geruchsdaten**, sondern wie `sensor_paket` jede unterstützte Sensordatenliste. Es ist eine Umbenennung, keine neue Sensorfähigkeit.
