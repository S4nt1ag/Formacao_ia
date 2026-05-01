# Aula 05 - POI Finder + n8n Weather Checklist

Este pacote deixa o fluxo pronto com:

- MCP server `poi-finder` (`poi.find`) usando OpenTripMap.
- Workflow n8n via webhook para previsao do tempo + checklist de bagagem.

## 1) Pre-requisitos

- Python com `.venv` ativo no projeto.
- Dependencia `mcp` instalada no ambiente Python usado pelo MCP.
- Chave da OpenTripMap em `OPENTRIPMAP_API_KEY`.
- n8n rodando localmente ou em cloud.

## 2) MCP no Cursor

Ja existe configuracao no arquivo de raiz `mcp.json`:

- `json-validator`
- `poi-finder`

Antes de usar, troque no `mcp.json`:

- `OPENTRIPMAP_API_KEY`: valor real da sua chave.

## 3) Workflow n8n

Importe o arquivo:

- `modulo03/aula05/n8n_weather_packing_webhook.json`

Fluxo:

1. `Webhook` (POST `/weather-packing-checklist`)
2. `Parse Input`
3. `Split In Batches` (1 por item)
4. `Open-Meteo Forecast`
5. `Build Checklist`
6. `Aggregate Results`
7. `Respond to Webhook`

Payload de teste:

- `modulo03/aula05/webhook_payload_exemplo.json`

Exemplo curl (troque URL pelo seu webhook de teste/producao):

```bash
curl -X POST "http://localhost:5678/webhook-test/weather-packing-checklist" \
  -H "Content-Type: application/json" \
  --data @modulo03/aula05/webhook_payload_exemplo.json
```

## 4) Formato de retorno do webhook

```json
{
  "itinerary_weather": [
    {
      "city": "Lisbon",
      "date": "2025-06-15",
      "forecast": {
        "temp_max_c": 27.1,
        "temp_min_c": 16.4,
        "rain_probability_max": 10,
        "uv_index_max": 7.2
      }
    }
  ],
  "checklists": [
    {
      "city": "Lisbon",
      "date": "2025-06-15",
      "items": ["Levar protetor solar"],
      "metadata": {
        "confidence": "medium",
        "source": "open-meteo.com",
        "generated_at": "2026-05-01T18:00:00.000Z"
      }
    }
  ]
}
```
