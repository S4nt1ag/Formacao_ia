import json
from typing import Any, Dict, List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "Dependencia ausente: mcp. Instale com: pip install mcp"
    ) from exc

try:
    from jsonpath_ng import parse as parse_jsonpath
except ImportError as exc:
    raise SystemExit(
        "Dependencia ausente: jsonpath-ng. Instale com: pip install jsonpath-ng"
    ) from exc


mcp = FastMCP("JSONValidator")


def _parse_json_payload(json_text: str) -> Any:
    return json.loads(json_text)


@mcp.tool()
def validate_json(json_text: str) -> Dict[str, Any]:
    """Valida se a string informada e um JSON bem formado."""
    try:
        _parse_json_payload(json_text)
        return {"valid": True, "error": None}
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": str(exc)}


@mcp.tool()
def pretty_print_json(json_text: str) -> Dict[str, Any]:
    """Formata um JSON para uma versao legivel e indentada."""
    try:
        payload = _parse_json_payload(json_text)
        pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return {"valid": True, "pretty": pretty, "error": None}
    except json.JSONDecodeError as exc:
        return {"valid": False, "pretty": None, "error": str(exc)}


@mcp.tool()
def extract_jsonpath(json_text: str, jsonpath_expr: str) -> Dict[str, Any]:
    """Extrai valores de um JSON com base em uma expressao JSONPath."""
    try:
        payload = _parse_json_payload(json_text)
    except json.JSONDecodeError as exc:
        return {"valid": False, "results": [], "error": f"JSON invalido: {exc}"}

    try:
        expression = parse_jsonpath(jsonpath_expr)
        matches = expression.find(payload)
        results: List[Any] = [match.value for match in matches]
        return {"valid": True, "results": results, "error": None}
    except Exception as exc:
        return {
            "valid": False,
            "results": [],
            "error": f"Erro ao processar JSONPath: {exc}",
        }


if __name__ == "__main__":
    mcp.run()
