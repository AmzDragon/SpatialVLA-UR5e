"""Create and persist the fixed 60+60 ID/OOD evaluation suite."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Callable, Iterable, Sequence

from dataset_record.config import RecordConfig
from evaluation.config import COLOR_RGBA, EvaluationConfig


CASES_VERSION = 3
OBJECTS = (
    "red_cube",
    "yellow_cylinder",
    "cyan_cuboid",
    "white_square_sheet",
    "black_rectangular_sheet",
)
MOVABLE_OBJECTS = OBJECTS[:3]
SHEET_OBJECTS = frozenset(OBJECTS[3:])
NON_CENTER_POSITIONS = ("up", "down", "left", "right")

SHAPES = {
    "red_cube": "cube",
    "yellow_cylinder": "cylinder",
    "cyan_cuboid": "cuboid",
    "white_square_sheet": "square paper",
    "black_rectangular_sheet": "rectangular paper",
}
BASE_COLORS = dict(zip(OBJECTS, ("red", "yellow", "cyan", "white", "black")))
RELATION_PHRASES = {
    "up": ("above", "on top of", "over"),
    "down": ("below", "under", "beneath"),
    "left": ("to the left of", "on the left side of", "left of"),
    "right": ("to the right of", "on the right side of", "right of"),
    "center": ("at the center of", "in the centre of", "in the middle of"),
}
LANGUAGE_STYLES = (
    ("Place", "place", ", then "),
    ("Move", "put", ", then "),
    ("Transfer", "carry", ", then "),
    ("Pick and place", "move", ". Then "),
)

PoseSampler = Callable[[], Iterable[tuple[str, Sequence[float]]]]


def load_or_create_cases(
    cfg: EvaluationConfig,
    record_cfg: RecordConfig,
    pose_sampler: PoseSampler,
    *,
    regenerate: bool = False,
) -> dict[str, object]:
    if cfg.cases_path.exists() and not regenerate:
        payload = json.loads(cfg.cases_path.read_text(encoding="utf-8"))
        if payload.get("version") == CASES_VERSION:
            return payload

    payload = generate_cases(cfg, record_cfg, pose_sampler)
    cfg.cases_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.cases_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def generate_cases(
    cfg: EvaluationConfig,
    record_cfg: RecordConfig,
    pose_sampler: PoseSampler,
) -> dict[str, object]:
    rng = random.Random(cfg.task_seed)
    descriptions = json.loads(
        cfg.source_tasks_path.read_text(encoding="utf-8")
    )["descriptions"]
    seen_items = rng.sample(
        descriptions[: cfg.language_pool_size],
        cfg.language_cases_per_split,
    )
    unseen_items = rng.sample(
        descriptions[-cfg.language_pool_size :],
        cfg.language_cases_per_split,
    )

    independent: list[dict[str, object]] = []
    for group, distribution, items in (
        ("seen_language", "seen", seen_items),
        ("unseen_language", "unseen", unseen_items),
    ):
        for item in items:
            independent.append(
                _source_case(
                    _next_id("independent", independent),
                    group,
                    distribution,
                    item,
                    _sample_poses(pose_sampler),
                )
            )

    seen_bases = independent[: cfg.factor_cases_per_distribution]
    scene_pairs = {
        "camera": _camera_scene_pairs(cfg),
        "table_color": _table_scene_pairs(cfg, record_cfg),
        "lighting": _lighting_scene_pairs(cfg),
    }
    independent_factor_cases: dict[
        tuple[str, str, int], dict[str, object]
    ] = {}
    for group, pairs in scene_pairs.items():
        for distribution, scene_index in (("in_distribution", 0), ("ood", 1)):
            for pair_index, (base_case, pair) in enumerate(
                zip(seen_bases, pairs, strict=True),
                start=1,
            ):
                case = _variant_case(
                    _next_id("independent", independent),
                    "independent",
                    group,
                    distribution,
                    f"{group}_{pair_index:02d}",
                    base_case,
                    pair[scene_index],
                )
                independent.append(case)
                independent_factor_cases[(group, distribution, pair_index)] = case

    color_specs = [
        {
            "commands": _independent_commands(rng),
            "object_poses": _sample_poses(pose_sampler),
            "style": index % len(LANGUAGE_STYLES),
            "phrase_variant": index % 3,
        }
        for index in range(cfg.factor_cases_per_distribution)
    ]
    canonical_assignment = tuple(BASE_COLORS[name] for name in OBJECTS)
    independent_color_id: list[dict[str, object]] = []
    for pair_index, spec in enumerate(color_specs, start=1):
        case = _generated_color_case(
            _next_id("independent", independent),
            "independent",
            "in_distribution",
            f"object_color_{pair_index:02d}",
            spec,
            canonical_assignment,
        )
        independent.append(case)
        independent_color_id.append(case)
        independent_factor_cases[("object_color", "in_distribution", pair_index)] = case

    for pair_index, (base_case, assignment) in enumerate(
        zip(
            independent_color_id,
            cfg.novel_color_assignments,
            strict=True,
        ),
        start=1,
    ):
        case = _recolor_case(
            _next_id("independent", independent),
            base_case,
            assignment,
            paired_case_id=None,
        )
        independent.append(case)
        independent_factor_cases[("object_color", "ood", pair_index)] = case

    chained: list[dict[str, object]] = []
    independent_language = independent[: 2 * cfg.language_cases_per_split]
    for base_case in independent_language:
        chained.append(
            _chained_case(
                _next_id("chained", chained),
                base_case,
                rng,
                phrase_variant=len(chained) % 3,
            )
        )

    chained_seen_bases = chained[: cfg.factor_cases_per_distribution]
    for group, pairs in scene_pairs.items():
        for distribution, scene_index in (("in_distribution", 0), ("ood", 1)):
            for pair_index, (base_case, pair) in enumerate(
                zip(chained_seen_bases, pairs, strict=True),
                start=1,
            ):
                case = _variant_case(
                    _next_id("chained", chained),
                    "chained",
                    group,
                    distribution,
                    f"{group}_{pair_index:02d}",
                    base_case,
                    pair[scene_index],
                )
                case["paired_case_id"] = independent_factor_cases[
                    (group, distribution, pair_index)
                ]["id"]
                chained.append(case)

    chained_color_id: list[dict[str, object]] = []
    for pair_index, base_case in enumerate(independent_color_id, start=1):
        case = _chained_case(
            _next_id("chained", chained),
            base_case,
            rng,
            phrase_variant=(pair_index - 1) % 3,
        )
        chained.append(case)
        chained_color_id.append(case)

    for pair_index, (base_case, assignment) in enumerate(
        zip(chained_color_id, cfg.novel_color_assignments, strict=True),
        start=1,
    ):
        case = _recolor_case(
            _next_id("chained", chained),
            base_case,
            assignment,
            paired_case_id=independent_factor_cases[
                ("object_color", "ood", pair_index)
            ]["id"],
        )
        chained.append(case)

    return {
        "version": CASES_VERSION,
        "seed": cfg.task_seed,
        "source_tasks_path": str(Path(cfg.source_tasks_path).resolve()),
        "pose_format": "xyz_rpy_radians",
        "structure_per_suite": {
            "seen_language": cfg.language_cases_per_split,
            "unseen_language": cfg.language_cases_per_split,
            "camera": 2 * cfg.factor_cases_per_distribution,
            "table_color": 2 * cfg.factor_cases_per_distribution,
            "lighting": 2 * cfg.factor_cases_per_distribution,
            "object_color": 2 * cfg.factor_cases_per_distribution,
        },
        "factor_distribution": {
            "in_distribution": cfg.factor_cases_per_distribution,
            "ood": cfg.factor_cases_per_distribution,
        },
        "training_distribution": _training_distribution(record_cfg),
        "cases": independent + chained,
    }


def _source_case(
    case_id: str,
    group: str,
    distribution: str,
    item: dict[str, object],
    object_poses: dict[str, list[float]],
) -> dict[str, object]:
    return {
        "id": case_id,
        "suite": "independent",
        "group": group,
        "distribution": distribution,
        "source_task_id": item["id"],
        "prompt": item["english"],
        "commands": copy.deepcopy(item["commands"]),
        "object_poses": object_poses,
        "scene": {},
    }


def _variant_case(
    case_id: str,
    suite: str,
    group: str,
    distribution: str,
    comparison_pair: str,
    base_case: dict[str, object],
    scene: dict[str, object],
) -> dict[str, object]:
    case = {
        "id": case_id,
        "suite": suite,
        "group": group,
        "distribution": distribution,
        "comparison_pair": comparison_pair,
        "base_case_id": base_case["id"],
        "prompt": base_case["prompt"],
        "commands": copy.deepcopy(base_case["commands"]),
        "object_poses": copy.deepcopy(base_case["object_poses"]),
        "scene": copy.deepcopy(scene),
    }
    if "source_task_id" in base_case:
        case["source_task_id"] = base_case["source_task_id"]
    return case


def _generated_color_case(
    case_id: str,
    suite: str,
    distribution: str,
    comparison_pair: str,
    spec: dict[str, object],
    assignment: Sequence[str],
) -> dict[str, object]:
    scene = _color_scene(assignment)
    return {
        "id": case_id,
        "suite": suite,
        "group": "object_color",
        "distribution": distribution,
        "comparison_pair": comparison_pair,
        "prompt": _render_prompt(
            spec["commands"],
            _display_names(scene),
            style=spec["style"],
            phrase_variant=spec["phrase_variant"],
        ),
        "commands": copy.deepcopy(spec["commands"]),
        "object_poses": copy.deepcopy(spec["object_poses"]),
        "scene": scene,
    }


def _chained_case(
    case_id: str,
    base_case: dict[str, object],
    rng: random.Random,
    *,
    phrase_variant: int,
) -> dict[str, object]:
    first_command = copy.deepcopy(base_case["commands"][0])
    second_command = _chained_second_command(first_command, rng)
    case = {
        "id": case_id,
        "suite": "chained",
        "group": base_case["group"],
        "distribution": base_case["distribution"],
        "paired_case_id": base_case["id"],
        "prompt": _replace_second_clause(
            base_case["prompt"],
            second_command,
            _display_names(base_case["scene"]),
            phrase_variant,
        ),
        "commands": [first_command, second_command],
        "object_poses": copy.deepcopy(base_case["object_poses"]),
        "scene": copy.deepcopy(base_case["scene"]),
    }
    for key in ("source_task_id", "comparison_pair"):
        if key in base_case:
            case[key] = base_case[key]
    return case


def _recolor_case(
    case_id: str,
    base_case: dict[str, object],
    assignment: Sequence[str],
    *,
    paired_case_id: str | None,
) -> dict[str, object]:
    scene = _color_scene(assignment)
    prompt = base_case["prompt"]
    old_names = _display_names(base_case["scene"])
    new_names = _display_names(scene)
    for object_name in OBJECTS:
        prompt = prompt.replace(old_names[object_name], new_names[object_name])

    case = {
        "id": case_id,
        "suite": base_case["suite"],
        "group": "object_color",
        "distribution": "ood",
        "comparison_pair": base_case["comparison_pair"],
        "base_case_id": base_case["id"],
        "prompt": prompt,
        "commands": copy.deepcopy(base_case["commands"]),
        "object_poses": copy.deepcopy(base_case["object_poses"]),
        "scene": scene,
    }
    if paired_case_id is not None:
        case["paired_case_id"] = paired_case_id
    return case


def _sample_poses(pose_sampler: PoseSampler) -> dict[str, list[float]]:
    return {
        object_name: [float(value) for value in pose]
        for object_name, pose in pose_sampler()
    }


def _camera_scene_pairs(
    cfg: EvaluationConfig,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    return [
        ({"camera": asdict(id_variant)}, {"camera": asdict(ood_variant)})
        for id_variant, ood_variant in zip(
            cfg.camera_in_distribution_variants,
            cfg.camera_ood_variants,
            strict=True,
        )
    ]


def _table_scene_pairs(
    cfg: EvaluationConfig,
    record_cfg: RecordConfig,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    pairs = []
    for palette_index, ood_variant in zip(
        cfg.table_in_distribution_indices,
        cfg.table_ood_variants,
        strict=True,
    ):
        pairs.append(
            (
                {
                    "table_color": {
                        "name": f"table_palette_{palette_index}",
                        "palette_index": palette_index,
                        "rgba": list(record_cfg.table_color_palette[palette_index]),
                    }
                },
                {
                    "table_color": {
                        "name": ood_variant.name,
                        "palette_index": None,
                        "rgba": list(ood_variant.rgba),
                    }
                },
            )
        )
    return pairs


def _lighting_scene_pairs(
    cfg: EvaluationConfig,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    return [
        ({"lighting": asdict(id_variant)}, {"lighting": asdict(ood_variant)})
        for id_variant, ood_variant in zip(
            cfg.lighting_in_distribution_variants,
            cfg.lighting_ood_variants,
            strict=True,
        )
    ]


def _color_scene(assignment: Sequence[str]) -> dict[str, object]:
    return {
        "object_colors": {
            object_name: {
                "color": color,
                "rgba": list(COLOR_RGBA[color]),
            }
            for object_name, color in zip(OBJECTS, assignment, strict=True)
        }
    }


def _display_names(scene: dict[str, object]) -> dict[str, str]:
    colors = BASE_COLORS.copy()
    for object_name, value in scene.get("object_colors", {}).items():
        colors[object_name] = value["color"]
    return {
        object_name: f"{colors[object_name]} {SHAPES[object_name]}"
        for object_name in OBJECTS
    }


def _independent_commands(rng: random.Random) -> list[dict[str, str]]:
    sources = rng.sample(MOVABLE_OBJECTS, 2)
    targets = rng.sample(
        [object_name for object_name in OBJECTS if object_name not in sources],
        2,
    )
    return [
        {
            "source_object": source,
            "target_object": target,
            "target_position": rng.choice(_positions_for_target(target)),
        }
        for source, target in zip(sources, targets, strict=True)
    ]


def _chained_second_command(
    first_command: dict[str, str],
    rng: random.Random,
) -> dict[str, str]:
    excluded = {
        first_command["source_object"],
        first_command["target_object"],
    }
    source = rng.choice(
        [name for name in MOVABLE_OBJECTS if name not in excluded]
    )
    return {
        "source_object": source,
        "target_object": first_command["source_object"],
        "target_position": rng.choice(NON_CENTER_POSITIONS),
    }


def _positions_for_target(target: str) -> tuple[str, ...]:
    if target in SHEET_OBJECTS:
        return (*NON_CENTER_POSITIONS, "center")
    return NON_CENTER_POSITIONS


def _render_prompt(
    commands: Sequence[dict[str, str]],
    display_names: dict[str, str],
    *,
    style: int,
    phrase_variant: int,
) -> str:
    first_verb, second_verb, separator = LANGUAGE_STYLES[style]
    first = _render_clause(commands[0], display_names, first_verb, phrase_variant)
    second = _render_clause(commands[1], display_names, second_verb, phrase_variant)
    return f"{first}{separator}{second}."


def _replace_second_clause(
    prompt: str,
    command: dict[str, str],
    display_names: dict[str, str],
    phrase_variant: int,
) -> str:
    style = next(
        index
        for index, (first_verb, _, _) in enumerate(LANGUAGE_STYLES)
        if prompt.startswith(first_verb + " ")
    )
    _, second_verb, separator = LANGUAGE_STYLES[style]
    first_clause = prompt.split(separator, 1)[0]
    second_clause = _render_clause(
        command,
        display_names,
        second_verb,
        phrase_variant,
    )
    return f"{first_clause}{separator}{second_clause}."


def _render_clause(
    command: dict[str, str],
    display_names: dict[str, str],
    verb: str,
    phrase_variant: int,
) -> str:
    relation = RELATION_PHRASES[command["target_position"]][phrase_variant]
    source = display_names[command["source_object"]]
    target = display_names[command["target_object"]]
    return f"{verb} the {source} {relation} the {target}"


def _training_distribution(record_cfg: RecordConfig) -> dict[str, object]:
    camera_translation = record_cfg.episode_camera_translation_limit_m
    camera_rotation = record_cfg.episode_camera_rotation_limit_deg
    frame_translation = record_cfg.frame_camera_translation_radius_m
    frame_rotation = record_cfg.frame_camera_rotation_limit_deg
    lighting_low, lighting_high = record_cfg.lighting_intensity_scale_range
    rgb_jitter = record_cfg.lighting_rgb_jitter_limit
    return {
        "camera": {
            "ood_reference": "episode_translation_per_axis_m",
            "episode_translation_per_axis_m": [
                -camera_translation,
                camera_translation,
            ],
            "frame_translation_radius_m": frame_translation,
            "combined_translation_per_axis_bound_m": [
                -camera_translation - frame_translation,
                camera_translation + frame_translation,
            ],
            "episode_rotation_per_axis_deg": [-camera_rotation, camera_rotation],
            "frame_rotation_per_axis_deg": [-frame_rotation, frame_rotation],
            "combined_rotation_per_axis_bound_deg": [
                -camera_rotation - frame_rotation,
                camera_rotation + frame_rotation,
            ],
        },
        "table_color": {
            "type": "discrete_palette",
            "rgba": [list(color) for color in record_cfg.table_color_palette],
        },
        "lighting": {
            "intensity_scale": [lighting_low, lighting_high],
            "rgb_scale_per_channel": [1.0 - rgb_jitter, 1.0 + rgb_jitter],
        },
        "object_color": {
            "type": "fixed_color_shape_pairs",
            "pairs": BASE_COLORS,
        },
    }


def _next_id(suite: str, cases: Sequence[dict[str, object]]) -> str:
    return f"{suite}_{len(cases) + 1:03d}"
