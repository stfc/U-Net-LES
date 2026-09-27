"""Read U-Net configuration from BOUT inputs without consulting the environment."""

import configparser
from pathlib import Path


def parse_bool(value, name):
    """Accept BOUT boolean spellings and reject ambiguous values."""
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"[unet] {name} must be true/false or 1/0; got {value!r}")


def load_unet_config(input_path="data/BOUT.inp", embedded=None):
    """Return options and their sources; never read or change environment variables.

    Embedded options are already resolved by BOUT++, including command-line
    overrides. Standalone imports read [unet] from input_path. Relative paths
    retain run-directory semantics. Missing entries use callback/checkpoint
    defaults, and an empty path remains distinct from an omitted setting.
    """
    if embedded is not None:
        values = {key: value for key, value in embedded.items() if not key.startswith("__")}
        source = "BOUT++ [unet]"
    else:
        path = Path(input_path).expanduser()
        parser = configparser.ConfigParser(
            interpolation=None, inline_comment_prefixes=("#", ";"), strict=True
        )
        # BOUT permits top-level options before its first section.
        parser.read_string("[__bout_root__]\n" + path.read_text(), source=str(path))
        values = {}
        if parser.has_section("unet"):
            for key, value in parser.items("unet"):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key] = value
        values["bout_inp"] = str(path)
        source = f"{path} [unet]"
    for key in ("implicit", "omega_from_potential"):
        if key in values:
            values[key] = "1" if parse_bool(values[key], key) else "0"
    return values, dict.fromkeys(values, source)
