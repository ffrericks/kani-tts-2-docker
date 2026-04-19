# n8n Integratie & Accenten Gids

Ik heb zojuist de **Accenten functionaliteit** succesvol toegevoegd aan de Kani-TTS server! Je hebt nu 15 officiële (standaard) stemmen waaronder Engels, Spaans, Duits, Arabisch, Chinees en Koreaans.

Omdat n8n een geweldig no-code / low-code platform is, heb ik de API zo gebouwd dat hij ontzettend makkelijk te integreren is in jouw n8n workflows.

## 1. Beschikbare Accenten
Je kunt 1 van de volgende stemmen meegeven aan de API in het `voice` veld:
- `david`, `puck`, `kore`, `andrew`, `jenny`, `simon`, `katie`, `bert` (Engels)
- `thorsten` (Duits)
- `maria` (Spaans)
- `seulgi` (Koreaans)
- `mei`, `ming` (Chinees)
- `karim`, `nur` (Arabisch)

---

## 2. De n8n Workflow Opzetten

Voeg in n8n een **HTTP Request** node toe, en stel deze exact zo in:

### Basispaneel:
- **Method:** `POST`
- **URL:** `http://<JOUW_CASAOS_IP>:8000/tts`  *(Let op: vervang het IP door je eigen server IP, bijv. 192.168.1.100)*
- **Authentication:** `None`

### Send Body Panel:
Zet **Send Body** AAN.
- **Body Type:** `JSON`
- Zorg ervoor dat **Specify Body** is geselecteerd.
- **Body:** Typ het volgende er in (of gebruik Expressions om data dynamisch in te vullen):
  ```json
  {
    "text": "Dit is de tekst die ik wil omzetten naar audio.",
    "voice": "maria",
    "temperature": 0.6
  }
  ```
  *(Opmerking: Haal `voice` weg of houd de string leeg als je wil dat hij zelf detecteert/de basisstem gebruikt.)*

### Response Settings:
Dit is een **belangrijke** stap om de audio intern in n8n op te slaan:
- Klap **Options** open onderaan het HTTP Request knooppunt.
- Voeg een nieuwe optie toe: **Response** -> **Response Format**
- Verander de Response Format van `Autodetect` (of JSON) naar `File`.
- (Optioneel) Voer bij `Property Name` iets in als `data`.

### Resultaat
Als je de node nu uitvoert in n8n, krijg je **niet** een blok tekst terug, maar geeft de HTTP Request node een **Binary file** (*data*). Dit is een .wav audiobestand!

Dit binaire bestand kun je nu direct koppelen aan een volgende node in n8n, zoals:
- **Telegram:** Send a Voice Message
- **Email:** Stuur als bijlage
- **AWS S3 / Dropbox:** Upload een bestand
