"""Validate the UI palette for contrast and colour-vision deficiency.

The design brief asks for a palette that is colourblind-safe with >= 4.5:1 text
contrast. Asserting that in a README is cheap; this script measures it, and the README
quotes its actual output -- including the one pair that does NOT clear the bar.

What it checks, and against which standard
------------------------------------------
* **Text colours** vs the page: WCAG 2.1 SC 1.4.3, floor 4.5:1.
* **Status swatches** (chip backgrounds, map lines) vs the page: WCAG 2.1 SC 1.4.11
  treats these as graphical objects, floor 3:1. Holding a decorative swatch to the
  text threshold would be the wrong test.
* **CIEDE2000 distance** between every pair of status swatches, under normal vision
  and simulated protanopia, deuteranopia and tritanopia. Status colours that collapse
  together for a red-blind reader are the real failure mode in a traffic UI where red
  and green carry opposite meanings.

The deltaE floor of 8 is not a formal standard; it is a working threshold meaning
"clearly different at a glance". A pair below it is reported as an **advisory**, not a
silent pass: the UI tells those states apart with a glyph, a text label and a distinct
line dash, so colour is never the only channel.

    python -m ml.export.check_palette
"""

from __future__ import annotations

import json

import numpy as np

from config.segments import REPO_ROOT

# --- the palette under test (must match frontend/src/styles/tokens.css) --------
PAPER = "#FBF7EF"
SURFACE = "#FFFFFF"
INK = "#16231C"
MUTED = "#50605A"

# Status SWATCHES: chip backgrounds and map line colours (graphical objects, 3:1).
STATUS_SWATCH = {
    "smooth": "#1B7A4B",
    "slow": "#B87D04",
    "disruption": "#D4581F",
    "jammed": "#A3231B",
}

# Status TEXT: the same four states rendered as coloured words on paper (4.5:1).
STATUS_TEXT = {
    "smooth": "#14603A",
    "slow": "#7A5300",
    "disruption": "#8F3410",
    "jammed": "#8C1D16",
}

ACCENTS = {
    "primary": "#1F5C47",
    "rain": "#1F5C8C",
    "flood": "#6B3FA0",
}

TEXT_ON_PAPER = {**STATUS_TEXT, **ACCENTS, "ink": INK, "muted": MUTED}

# How each status is coded WITHOUT colour. Mirrored in the frontend's statusMeta.
NON_COLOUR_CODING = {
    "smooth": {"glyph": "check", "label": "Smooth", "dash": "solid"},
    "slow": {"glyph": "triangle", "label": "Slow", "dash": "dashed 10 6"},
    "disruption": {"glyph": "diamond", "label": "Disruption likely", "dash": "dotted 2 7"},
    "jammed": {"glyph": "square", "label": "Jammed", "dash": "dash-dot 14 5 3 5"},
}

DELTA_E_FLOOR = 8.0
TEXT_CONTRAST_FLOOR = 4.5
GRAPHIC_CONTRAST_FLOOR = 3.0


def hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _linearise(channel: np.ndarray) -> np.ndarray:
    channel = channel / 255.0
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb: np.ndarray) -> float:
    return float(np.dot(_linearise(rgb), [0.2126, 0.7152, 0.0722]))


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = relative_luminance(hex_to_rgb(foreground))
    darker = relative_luminance(hex_to_rgb(background))
    if lighter < darker:
        lighter, darker = darker, lighter
    return (lighter + 0.05) / (darker + 0.05)


# --- colour-vision deficiency simulation (LMS based) --------------------------
RGB_TO_LMS = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ]
)
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)

SIMULATIONS = {
    "protanopia": np.array([[0.0, 1.05118294, -0.05116099], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "deuteranopia": np.array([[1.0, 0.0, 0.0], [0.9513092, 0.0, 0.04866992], [0.0, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.86744736, 1.86727089, 0.0]]),
}


def simulate_cvd(hex_colour: str, matrix: np.ndarray) -> np.ndarray:
    linear = _linearise(hex_to_rgb(hex_colour)) * 255.0
    simulated = LMS_TO_RGB @ (matrix @ (RGB_TO_LMS @ linear))
    simulated = np.clip(simulated / 255.0, 0, 1)
    srgb = np.where(
        simulated <= 0.0031308, simulated * 12.92, 1.055 * simulated ** (1 / 2.4) - 0.055
    )
    return np.clip(srgb * 255.0, 0, 255)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = matrix @ _linearise(rgb)
    ratio = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(ratio > 0.008856, np.cbrt(ratio), 7.787 * ratio + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7))) if avg_C > 0 else 0.0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2
    else:
        avg_hp = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )
    SL = 1 + (0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T
    RT = (
        -2
        * np.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
        * np.sin(np.radians(60 * np.exp(-(((avg_hp - 275) / 25) ** 2))))
        if avg_Cp > 0
        else 0.0
    )
    return float(
        np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))
    )


def swatch_distance(a: str, b: str, mode: str) -> float:
    if mode == "normal":
        rgb_a, rgb_b = hex_to_rgb(a), hex_to_rgb(b)
    else:
        rgb_a = simulate_cvd(a, SIMULATIONS[mode])
        rgb_b = simulate_cvd(b, SIMULATIONS[mode])
    return ciede2000(rgb_to_lab(rgb_a), rgb_to_lab(rgb_b))


def main() -> int:
    problems: list[str] = []
    advisories: list[str] = []
    report: dict = {
        "paper": PAPER,
        "ink": INK,
        "status_swatch": STATUS_SWATCH,
        "status_text": STATUS_TEXT,
        "contrast_text": {},
        "contrast_swatch_vs_paper": {},
        "delta_e": {},
        "floors": {
            "text_contrast_ratio": TEXT_CONTRAST_FLOOR,
            "graphic_contrast_ratio": GRAPHIC_CONTRAST_FLOOR,
            "delta_e": DELTA_E_FLOOR,
        },
        "non_colour_coding": NON_COLOUR_CODING,
    }

    print(f"=== text colours vs paper {PAPER} (WCAG 1.4.3, floor {TEXT_CONTRAST_FLOOR}:1) ===")
    for name, colour in TEXT_ON_PAPER.items():
        ratio = contrast_ratio(colour, PAPER)
        report["contrast_text"][name] = round(ratio, 2)
        passed = ratio >= TEXT_CONTRAST_FLOOR
        if not passed:
            problems.append(f"text {name} ({colour}) contrast {ratio:.2f} < {TEXT_CONTRAST_FLOOR}")
        print(f"  {name:<12} {colour}  {ratio:5.2f}:1  {'PASS' if passed else 'FAIL'}")

    print(f"\n=== status swatches vs paper (WCAG 1.4.11, floor {GRAPHIC_CONTRAST_FLOOR}:1) ===")
    for name, colour in STATUS_SWATCH.items():
        ratio = contrast_ratio(colour, PAPER)
        report["contrast_swatch_vs_paper"][name] = round(ratio, 2)
        passed = ratio >= GRAPHIC_CONTRAST_FLOOR
        if not passed:
            problems.append(
                f"swatch {name} ({colour}) vs paper {ratio:.2f} < {GRAPHIC_CONTRAST_FLOOR}"
            )
        print(f"  {name:<12} {colour}  {ratio:5.2f}:1  {'PASS' if passed else 'FAIL'}")

    print(f"\n=== CIEDE2000 between status swatches (working floor {DELTA_E_FLOOR}) ===")
    names = list(STATUS_SWATCH)
    for mode in ("normal", *SIMULATIONS):
        print(f"\n  -- {mode} --")
        report["delta_e"][mode] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                distance = swatch_distance(STATUS_SWATCH[a], STATUS_SWATCH[b], mode)
                report["delta_e"][mode][f"{a}|{b}"] = round(distance, 2)
                if distance < DELTA_E_FLOOR:
                    advisories.append(
                        f"{a} vs {b}: deltaE {distance:.1f} under {mode}. Below the floor, so "
                        f"these two are separated by glyph "
                        f"({NON_COLOUR_CODING[a]['glyph']} vs {NON_COLOUR_CODING[b]['glyph']}), "
                        f"text label and line dash rather than by colour."
                    )
                verdict = "PASS" if distance >= DELTA_E_FLOOR else "below floor"
                print(f"    {a:<11} vs {b:<11} deltaE {distance:6.2f}  {verdict}")

    report["problems"] = problems
    report["advisories"] = advisories
    report["passed"] = not problems

    out = REPO_ROOT / "ml" / "reports" / "palette_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + ("CONTRAST: all checks pass" if not problems else "CONTRAST PROBLEMS:"))
    for problem in problems:
        print(f"  - {problem}")

    if advisories:
        print("\nCOLOUR-SEPARATION ADVISORIES (reported, not hidden):")
        for advisory in advisories:
            print(f"  - {advisory}")
    else:
        print("\nAll status pairs clear the deltaE floor in every vision mode.")

    print(
        "\nStatus is never carried by colour alone: every chip renders colour + glyph + "
        "text label, and every map line adds a distinct dash pattern."
    )
    print(f"\nsaved {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
