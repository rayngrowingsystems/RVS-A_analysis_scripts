import json
import os
import re

import pytest

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXCLUDED_DIRS = {".git", ".idea", "tests", "__pycache__", ".pytest_cache", "presets"}

# Discover all analysis script folders
script_dirs = [d for d in os.listdir(REPO_DIR) if os.path.isdir(os.path.join(REPO_DIR, d)) and d not in EXCLUDED_DIRS]

# Build paths for each folder
script_config_pairs = [
    (os.path.join(REPO_DIR, d, f"{d}.py"), os.path.join(REPO_DIR, d, f"{d}.config"))
    for d in script_dirs
    if os.path.exists(os.path.join(REPO_DIR, d, f"{d}.py")) and os.path.exists(os.path.join(REPO_DIR, d, f"{d}.config"))
]


def extract_keys_from_config(config_block):
    """Extract setting 'name' keys from a config section."""
    keys = set()
    if "options" in config_block and "sections" in config_block["options"]:
        for section in config_block["options"]["sections"]:
            for setting in section.get("settings", []):
                if "name" in setting:
                    keys.add(setting["name"])
    return keys


def extract_keys_from_config_presets(config_data):
    """Extract setting keys from any preset file listed in the config (simple or nested)."""

    preset_keys = set()

    if "script" in config_data and "options" in config_data["script"]:
        for section in config_data["script"]["options"].get("sections", []):
            for preset_file in section.get("presets", []):
                preset_path = os.path.join(REPO_DIR, "presets", preset_file)
                if not os.path.isfile(preset_path):
                    continue
                with open(preset_path, "r") as f:
                    try:
                        preset_data = json.load(f)

                        # CASE 1: Flat structure like {"settings": [...]}
                        if "settings" in preset_data:
                            for setting in preset_data["settings"]:
                                if "name" in setting:
                                    preset_keys.add(setting["name"])

                        # CASE 2: Named presets like { "Base": [ { "settings": [...] } ] }
                        else:
                            for preset_name, preset_sections in preset_data.items():
                                for section in preset_sections:
                                    for setting in section.get("settings", []):
                                        if "name" in setting:
                                            preset_keys.add(setting["name"])

                    except Exception as e:
                        print(f"Failed to load or parse preset file '{preset_file}': {e}")

    return preset_keys


def extract_used_script_keys(script_path):
    """Find all keys accessed via script_options[...] or full settings path."""

    with open(script_path, "r") as f:
        content = f.read()

    keys = set()

    # Detect alias variable assigned to settings["..."]["..."]["scriptOptions"]["general"]
    alias_pattern = re.compile(
        r"(\w+)\s*=\s*settings\[\s*['\"]experimentSettings['\"]\s*]\[\s*['\"]analysis['\"]\s*]\[\s*['\"]scriptOptions['\"]\s*]\[\s*['\"]general['\"]\s*]"
    )
    aliases = alias_pattern.findall(content)
    aliases.append('settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]')  # direct path

    # Search for key access from any alias
    for alias in aliases:
        escaped_alias = re.escape(alias)
        key_pattern = re.compile(rf"{escaped_alias}\[\s*['\"](.*?)['\"]\s*]")
        matches = key_pattern.findall(content)
        keys.update(matches)

    return keys


@pytest.mark.parametrize("script_path, config_path", script_config_pairs)
def test_analysis_config_structure(script_path, config_path):
    """Check if config file has both 'mask' and 'script' sections with required structure."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    for top_key in ("mask", "script"):
        assert top_key in config_data, f"Missing top-level key: '{top_key}' in {config_path}"
        assert "info" in config_data[top_key], f"Missing '{top_key}.info' in {config_path}"
        assert "options" in config_data[top_key], f"Missing '{top_key}.options' in {config_path}"
        assert "sections" in config_data[top_key]["options"], f"Missing '{top_key}.options.sections' in {config_path}"

        for section in config_data[top_key]["options"]["sections"]:
            assert "settings" in section or "presets" in section, (
                f"Missing 'settings' in section of '{top_key}' in {config_path}"
            )


@pytest.mark.parametrize("script_path, config_path", script_config_pairs)
def test_script_settings_vs_config_and_presets(script_path, config_path):
    """Check that script only uses keys defined in script config or presets."""

    used_keys = extract_used_script_keys(script_path)

    with open(config_path, "r") as f:
        config_data = json.load(f)

    config_keys = extract_keys_from_config(config_data["script"])
    preset_keys = extract_keys_from_config_presets(config_data)

    all_available_keys = config_keys.union(preset_keys)
    missing = used_keys - all_available_keys

    # DEBUG PRINT
    print(f"\nSCRIPT: {os.path.basename(script_path)}")
    print("Used keys:", used_keys)
    print("Config keys:", config_keys)
    print("Preset keys:", preset_keys)
    print("Missing keys:", missing)

    assert not missing, f"{os.path.basename(script_path)} uses undefined settings: {missing}"


# Check dropdown_values()/range_values() conditions match config
@pytest.mark.parametrize("script_path, config_path", script_config_pairs)
def test_getValuesFor_conditions(script_path, config_path):
    """Check if dropdown/range values in config are handled in function bodies."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    get_keys = set()

    # extract getValuesFor and getRangesFor from script section
    for section in config_data["script"]["options"]["sections"]:
        for setting in section.get("settings", []):
            if "getValuesFor" in setting:
                get_keys.add(("dropdown_values", setting["getValuesFor"]))
            if "getRangesFor" in setting:
                get_keys.add(("range_values", setting["getRangesFor"]))

    with open(script_path, "r") as f:
        content = f.read()

    for func_name, key in get_keys:
        assert f"def {func_name}" in content, f"Missing function: {func_name}() in {script_path}"
        # Handle different arg names (e.g., `name` or `setting`)
        pattern = re.compile(rf"def {func_name}\((.*?)\):([\s\S]*?)(def |\Z)", re.MULTILINE)
        match = pattern.search(content)
        assert match, f"Function body for {func_name} not found in {script_path}"

        func_body = match.group(2)
        assert f'"{key}"' in func_body or f"'{key}'" in func_body, (
            f"Missing if condition for key '{key}' in function {func_name}() in {script_path}"
        )
