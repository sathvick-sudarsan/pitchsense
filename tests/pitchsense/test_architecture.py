import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "pitchsense"
INHERITED = {
    "trackers",
    "team_assigner",
    "player_ball_assigner",
    "camera_movement_estimator",
    "view_transformer",
    "speed_and_distance_estimator",
    "utils",
    "main",
}


def forbidden_imports(source):
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.append(node.module or "")
    return [name for name in imports if name.split(".")[0] in INHERITED]


def test_guard_detects_direct_and_from_imports():
    assert forbidden_imports(
        "import trackers.core\nfrom utils.geometry import distance"
    ) == ["trackers.core", "utils.geometry"]


def test_new_package_does_not_import_inherited_modules():
    files = list(ROOT.rglob("*.py"))
    assert files
    offenders = {
        str(path.relative_to(ROOT)): forbidden_imports(path.read_text(encoding="utf-8"))
        for path in files
    }
    assert not {path: names for path, names in offenders.items() if names}
