import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "ar_path_viewer"


def test_no_ros_publishers_or_action_clients_are_created():
    forbidden_calls = {"create_publisher", "create_client", "ActionClient"}
    found = []
    for source_path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else (
                    function.id if isinstance(function, ast.Name) else ""
                )
                if name in forbidden_calls:
                    found.append((source_path.name, node.lineno, name))
    assert found == []


def test_no_control_topic_or_nav_goal_names_in_runtime_sources():
    forbidden = ("/cmd_vel", '"/stop"', "NavigateToPose", "NavigateThroughPoses")
    contents = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py")
    )
    for token in forbidden:
        assert token not in contents


def test_runtime_sources_do_not_hardcode_sample_map_or_waypoint_name():
    contents = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py")
    )
    assert "1014-15_True" not in contents
    assert "1014_15_True" not in contents
