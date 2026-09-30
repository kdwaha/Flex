"""Export finalized reports and compact measurements without model state or text outputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


TEXT_OUTPUT_KEYS = {"predictions", "references", "generation_records"}


def without_text_outputs(value):
    if isinstance(value, dict):
        return {key: without_text_outputs(item) for key, item in value.items()
                if key not in TEXT_OUTPUT_KEYS}
    if isinstance(value, list):
        return [without_text_outputs(item) for item in value]
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    source, destination = Path(source), Path(destination)
    runs = sorted(source.glob("*/*/diagnostics.json"))
    if not runs or not (source / "report" / "comparison.md").is_file():
        raise ValueError("Expected a completed suite with diagnostics and a comparison report.")
    for path in runs:
        if not (path.parent / "metrics.json").is_file():
            raise ValueError(f"Refusing to export unfinished run: {path.parent.name}")
    # Never overwrite either existing published results or raw experiments.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"source_suite": source.name,
                "omitted_json_fields": sorted(TEXT_OUTPUT_KEYS), "files": {}}

    def record(original, target):
        manifest["files"][str(target.relative_to(destination))] = {
            "source": str(original.relative_to(source)),
            "source_sha256": sha256(original), "published_sha256": sha256(target),
        }

    for original in sorted((source / "report").iterdir()):
        if original.is_file() and original.suffix in {".md", ".csv", ".png"}:
            target = destination / original.name
            if original.name == "comparison.md":
                content = original.read_text().replace(
                    "각 조건 종료 후 자동 갱신됩니다.",
                    "이 파일은 완료된 실험의 고정된 공개 스냅샷입니다.")
                target.write_text(content)
            else:
                shutil.copyfile(original, target)
            record(original, target)

    for path in runs:
        target_dir = destination / "measurements" / path.parent.parent.name
        target_dir.mkdir(parents=True)
        # Scalars, trace samples and Lanczos nodes/weights suffice to rebuild plots.
        for name in ("args.json", "client_partition.json", "loss_spec.json",
                     "diagnostics.json", "metrics.json", "diagnostics.csv"):
            original, target = path.parent / name, target_dir / name
            if not original.is_file():
                continue
            if original.suffix == ".json":
                value = without_text_outputs(json.loads(original.read_text()))
                target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            else:
                shutil.copyfile(original, target)
            record(original, target)

    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(runs)} completed conditions and {len(manifest['files'])} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
