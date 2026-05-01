import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Dependencia ausente: mcp. Instale com: pip install mcp") from exc


BASE_URL = "https://api.opentripmap.com/0.1/en/places"
CACHE_TTL_SECONDS = int(os.getenv("POI_CACHE_TTL_SECONDS", "900"))
MAX_DETAIL_REQUESTS = int(os.getenv("POI_MAX_DETAIL_REQUESTS", "20"))
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY", "").strip()

if not OPENTRIPMAP_API_KEY:
    raise SystemExit("Defina OPENTRIPMAP_API_KEY no ambiente para usar o POI Finder.")

mcp = FastMCP("poi-finder")
_cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    row = _cache.get(key)
    if not row:
        return None
    created_at, value = row
    if time.time() - created_at > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


def _http_get_json(url: str, timeout: int = 25) -> Dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _build_url(path: str, params: Dict[str, Any]) -> str:
    query = urllib.parse.urlencode(params)
    return f"{BASE_URL}{path}?{query}"


def _truncate(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _poi_preview(detail: Dict[str, Any]) -> str:
    wikipedia_extracts = detail.get("wikipedia_extracts") or {}
    if wikipedia_extracts.get("text"):
        return _truncate(wikipedia_extracts["text"])

    info = detail.get("info") or {}
    if info.get("descr"):
        return _truncate(info["descr"])

    address = detail.get("address") or {}
    if address.get("road"):
        return _truncate(f"Local em {address['road']}")

    return "Sem descricao curta disponivel."


def _poi_kind(detail: Dict[str, Any], fallback: str = "") -> str:
    kinds = (detail.get("kinds") or fallback or "").split(",")
    clean = [k.strip() for k in kinds if k.strip()]
    return clean[0] if clean else "unknown"


@mcp.tool(name="poi.find")
def poi_find(
    lat: float,
    lon: float,
    radius_m: int = 1500,
    kinds: Optional[str] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """
    Busca pontos de interesse ao redor de coordenadas usando OpenTripMap.

    Args:
      lat: Latitude em decimal.
      lon: Longitude em decimal.
      radius_m: Raio em metros para busca.
      kinds: Categorias OpenTripMap, ex.: "museums,cultural,parks".
      limit: Numero maximo de POIs retornados (1-20).

    Returns:
      Lista estruturada de POIs no formato:
      {
        "id": "xid",
        "name": "Nome do local",
        "kind": "museums",
        "lat": 38.71,
        "lon": -9.13,
        "dist_m": 456,
        "preview": "descricao curta",
        "wikidata": "Q1234"
      }
    """
    radius_m = max(100, min(radius_m, 50_000))
    limit = max(1, min(limit, 20))
    kinds = (kinds or "").strip() or None

    cache_key = f"{lat:.5f}|{lon:.5f}|{radius_m}|{kinds}|{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params: Dict[str, Any] = {
        "apikey": OPENTRIPMAP_API_KEY,
        "radius": radius_m,
        "lon": lon,
        "lat": lat,
        "limit": limit,
        "rate": 2,
        "format": "json",
    }
    if kinds:
        params["kinds"] = kinds

    try:
        radius_url = _build_url("/radius", params)
        radius_data = _http_get_json(radius_url)
    except Exception as exc:
        raise RuntimeError(f"Erro ao consultar OpenTripMap (radius): {exc}") from exc

    if not isinstance(radius_data, list):
        raise RuntimeError("Resposta inesperada da API OpenTripMap no endpoint radius.")

    pois: List[Dict[str, Any]] = []
    for item in radius_data[:MAX_DETAIL_REQUESTS]:
        xid = item.get("xid")
        if not xid:
            continue

        detail_key = f"detail|{xid}"
        detail = _cache_get(detail_key)
        if detail is None:
            try:
                detail_url = _build_url(f"/xid/{urllib.parse.quote(str(xid))}", {"apikey": OPENTRIPMAP_API_KEY})
                detail = _http_get_json(detail_url)
                _cache_set(detail_key, detail)
            except Exception:
                detail = {}

        point = detail.get("point") or {}
        poi = {
            "id": xid,
            "name": detail.get("name") or item.get("name") or "Sem nome",
            "kind": _poi_kind(detail, fallback=item.get("kinds", "")),
            "lat": point.get("lat", item.get("point", {}).get("lat", lat)),
            "lon": point.get("lon", item.get("point", {}).get("lon", lon)),
            "dist_m": int(item.get("dist", 0)),
            "preview": _poi_preview(detail),
            "wikidata": detail.get("wikidata") or None,
        }
        pois.append(poi)
        if len(pois) >= limit:
            break

    _cache_set(cache_key, pois)
    return pois


if __name__ == "__main__":
    mcp.run()
