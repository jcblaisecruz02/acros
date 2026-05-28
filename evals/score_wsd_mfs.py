#!/usr/bin/env python
"""Parse Raganato-style WSD data and score an MFS baseline.

This is intentionally model-free. It validates the WSD data plumbing before
we plug in induced-sense activations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.etree import ElementTree as ET


WN_POS_TO_CODE = {"n": "1", "v": "2", "a": "3", "s": "5", "r": "4"}
SENSE_KEY_POS_TO_WN = {v: k for k, v in WN_POS_TO_CODE.items()}
SENSE_KEY_POS_TO_WN["5"] = "a"

POS_ALIASES = {
    "n": "n",
    "noun": "n",
    "nn": "n",
    "nns": "n",
    "np": "n",
    "v": "v",
    "verb": "v",
    "vb": "v",
    "vbd": "v",
    "vbg": "v",
    "vbn": "v",
    "vbp": "v",
    "vbz": "v",
    "a": "a",
    "s": "a",
    "adj": "a",
    "adjective": "a",
    "jj": "a",
    "jjr": "a",
    "jjs": "a",
    "r": "r",
    "adv": "r",
    "adverb": "r",
    "rb": "r",
    "rbr": "r",
    "rbs": "r",
}


@dataclass(frozen=True)
class WSDInstance:
    dataset: str
    instance_id: str
    sentence_id: str
    target_index: int
    target_text: str
    lemma: str
    pos: str
    tokens: Tuple[str, ...]
    lemmas: Tuple[str, ...]
    pos_tags: Tuple[str, ...]
    gold_sense_keys: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raganato-root",
        type=Path,
        default=None,
        help="Optional root with Training_Corpora/SemCor and Evaluation_Datasets.",
    )
    p.add_argument(
        "--train",
        action="append",
        default=None,
        help="Repeatable label=path/to/data.xml:path/to/gold.key.txt spec.",
    )
    p.add_argument(
        "--eval",
        action="append",
        default=None,
        help="Repeatable label=path/to/data.xml:path/to/gold.key.txt spec.",
    )
    p.add_argument(
        "--wordnet-index-sense",
        type=Path,
        default=None,
        help="Optional WordNet index.sense file for first-sense fallback.",
    )
    p.add_argument(
        "--mfs-source",
        choices=["semcor", "wordnet", "semcor_then_wordnet"],
        default="semcor_then_wordnet",
        help=(
            "Source for MFS predictions. Use wordnet for the standard first-sense "
            "baseline; semcor_then_wordnet is useful for checking training-derived MFS."
        ),
    )
    p.add_argument("--out", type=Path, default=Path("eval_logs/wsd/mfs_baseline.json"))
    p.add_argument("--include-predictions", action="store_true")
    return p.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_lemma(lemma: str) -> str:
    return lemma.strip().lower().replace(" ", "_")


def normalize_pos(pos: str) -> str:
    return POS_ALIASES.get(pos.strip().lower(), pos.strip().lower())


def sense_key_pos(sense_key: str) -> str | None:
    if "%" not in sense_key:
        return None
    after_percent = sense_key.split("%", 1)[1]
    if not after_percent:
        return None
    return SENSE_KEY_POS_TO_WN.get(after_percent[0])


def sense_key_lemma(sense_key: str) -> str:
    return normalize_lemma(sense_key.split("%", 1)[0])


def lemma_pos_key(lemma: str, pos: str) -> str:
    return f"{normalize_lemma(lemma)}.{normalize_pos(pos)}"


def parse_gold_keys(path: Path) -> Dict[str, Tuple[str, ...]]:
    gold: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no}: expected instance_id plus at least one sense key")
            gold[parts[0]] = tuple(parts[1:])
    return gold


def parse_wsd_xml(path: Path, gold: Mapping[str, Sequence[str]], dataset: str) -> List[WSDInstance]:
    tree = ET.parse(path)
    root = tree.getroot()
    instances: List[WSDInstance] = []

    for sentence in root.iter():
        if local_name(sentence.tag) != "sentence":
            continue
        sentence_id = sentence.attrib.get("id", "")
        token_nodes = [node for node in list(sentence) if local_name(node.tag) in {"wf", "instance"}]
        tokens = tuple((node.text or "").strip() for node in token_nodes)
        lemmas = tuple(node.attrib.get("lemma", token).strip() for node, token in zip(token_nodes, tokens))
        pos_tags = tuple(normalize_pos(node.attrib.get("pos", "")) for node in token_nodes)

        for target_index, node in enumerate(token_nodes):
            if local_name(node.tag) != "instance":
                continue
            instance_id = node.attrib.get("id")
            if not instance_id:
                raise ValueError(f"{path}: encountered <instance> without id")
            target_text = tokens[target_index]
            lemma = node.attrib.get("lemma", target_text)
            pos = normalize_pos(node.attrib.get("pos", ""))
            instances.append(
                WSDInstance(
                    dataset=dataset,
                    instance_id=instance_id,
                    sentence_id=sentence_id,
                    target_index=target_index,
                    target_text=target_text,
                    lemma=normalize_lemma(lemma),
                    pos=pos,
                    tokens=tokens,
                    lemmas=lemmas,
                    pos_tags=pos_tags,
                    gold_sense_keys=tuple(gold.get(instance_id, ())),
                )
            )

    return instances


def parse_dataset_spec(spec: str) -> Tuple[str, Path, Path]:
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        label, rest = "", spec
    if ":" not in rest:
        raise ValueError(f"Dataset spec must be label=xml:gold, got {spec!r}")
    xml_path, gold_path = rest.split(":", 1)
    xml = Path(xml_path)
    gold = Path(gold_path)
    label = label.strip() or xml.stem.replace(".data", "")
    return label, xml, gold


def autodiscover_raganato(root: Path) -> Tuple[List[str], List[str]]:
    train_specs: List[str] = []
    eval_specs: List[str] = []

    semcor_dir = root / "Training_Corpora" / "SemCor"
    semcor_xml = semcor_dir / "semcor.data.xml"
    semcor_gold = semcor_dir / "semcor.gold.key.txt"
    if semcor_xml.exists() and semcor_gold.exists():
        train_specs.append(f"semcor={semcor_xml}:{semcor_gold}")

    eval_root = root / "Evaluation_Datasets"
    all_dir = eval_root / "ALL"
    all_xml = all_dir / "ALL.data.xml"
    all_gold = all_dir / "ALL.gold.key.txt"
    if all_xml.exists() and all_gold.exists():
        eval_specs.append(f"ALL={all_xml}:{all_gold}")
        return train_specs, eval_specs

    for dataset_dir in sorted(eval_root.iterdir()) if eval_root.exists() else []:
        if not dataset_dir.is_dir():
            continue
        xmls = sorted(dataset_dir.glob("*.data.xml"))
        golds = sorted(dataset_dir.glob("*.gold.key.txt"))
        if not xmls or not golds:
            continue
        label = dataset_dir.name
        eval_specs.append(f"{label}={xmls[0]}:{golds[0]}")

    return train_specs, eval_specs


def load_instances(specs: Sequence[str], kind: str) -> List[WSDInstance]:
    all_instances: List[WSDInstance] = []
    for spec in specs:
        label, xml_path, gold_path = parse_dataset_spec(spec)
        if not xml_path.exists():
            raise FileNotFoundError(f"{kind} XML not found: {xml_path}")
        if not gold_path.exists():
            raise FileNotFoundError(f"{kind} gold not found: {gold_path}")
        gold = parse_gold_keys(gold_path)
        instances = parse_wsd_xml(xml_path, gold, dataset=label)
        all_instances.extend(instances)
    return all_instances


def build_mfs_from_training(instances: Iterable[WSDInstance]) -> Dict[str, str]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for inst in instances:
        if not inst.gold_sense_keys:
            continue
        key = lemma_pos_key(inst.lemma, inst.pos)
        for sense_key in inst.gold_sense_keys:
            counts[key][sense_key] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items() if counter}


def load_wordnet_first_sense(path: Path) -> Dict[str, str]:
    first_sense: Dict[str, Tuple[int, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected index.sense row")
            sense_key = parts[0]
            try:
                sense_number = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid sense_number {parts[2]!r}") from exc
            pos = sense_key_pos(sense_key)
            if pos is None:
                continue
            key = lemma_pos_key(sense_key_lemma(sense_key), pos)
            old = first_sense.get(key)
            if old is None or sense_number < old[0]:
                first_sense[key] = (sense_number, sense_key)
    return {key: sense_key for key, (_, sense_key) in first_sense.items()}


def predict_mfs(
    inst: WSDInstance,
    train_mfs: Mapping[str, str],
    first_sense: Mapping[str, str],
    mfs_source: str,
) -> str | None:
    key = lemma_pos_key(inst.lemma, inst.pos)
    if mfs_source == "semcor":
        return train_mfs.get(key)
    if mfs_source == "wordnet":
        return first_sense.get(key)
    if key in train_mfs:
        return train_mfs[key]
    return first_sense.get(key)


def score_instances(
    instances: Sequence[WSDInstance],
    train_mfs: Mapping[str, str],
    first_sense: Mapping[str, str],
    mfs_source: str,
    include_predictions: bool,
) -> Dict:
    by_dataset: Dict[str, Dict] = {}
    predictions: List[Dict] = []

    correct = 0
    answered = 0
    total = 0
    missing_gold = 0
    fallback_first_sense = 0
    missing_prediction = 0

    dataset_totals: Dict[str, Counter[str]] = defaultdict(Counter)

    for inst in instances:
        if not inst.gold_sense_keys:
            missing_gold += 1
            continue
        total += 1
        pred = predict_mfs(inst, train_mfs, first_sense, mfs_source)
        key = lemma_pos_key(inst.lemma, inst.pos)
        used_first_sense = pred is not None and (mfs_source == "wordnet" or key not in train_mfs)
        is_correct = pred in inst.gold_sense_keys if pred is not None else False

        if pred is None:
            missing_prediction += 1
        else:
            answered += 1
            fallback_first_sense += int(used_first_sense)
            correct += int(is_correct)

        bucket = dataset_totals[inst.dataset]
        bucket["total"] += 1
        bucket["answered"] += int(pred is not None)
        bucket["correct"] += int(is_correct)
        bucket["first_sense_fallback"] += int(used_first_sense)
        bucket["missing_prediction"] += int(pred is None)

        if include_predictions:
            predictions.append(
                {
                    "dataset": inst.dataset,
                    "instance_id": inst.instance_id,
                    "sentence": " ".join(inst.tokens),
                    "target_index": inst.target_index,
                    "target_text": inst.target_text,
                    "lemma": inst.lemma,
                    "pos": inst.pos,
                    "gold_sense_keys": list(inst.gold_sense_keys),
                    "prediction": pred,
                    "correct": is_correct,
                    "source": (
                        "wordnet_first_sense"
                        if mfs_source == "wordnet" or (pred and key not in train_mfs)
                        else "semcor_mfs"
                        if pred
                        else "missing"
                    ),
                }
            )

    precision = correct / answered if answered else 0.0
    recall = correct / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct / total if total else 0.0

    for dataset, counts in sorted(dataset_totals.items()):
        d_total = counts["total"]
        d_answered = counts["answered"]
        d_correct = counts["correct"]
        d_precision = d_correct / d_answered if d_answered else 0.0
        d_recall = d_correct / d_total if d_total else 0.0
        d_f1 = 2 * d_precision * d_recall / (d_precision + d_recall) if d_precision + d_recall else 0.0
        by_dataset[dataset] = {
            "total": d_total,
            "answered": d_answered,
            "correct": d_correct,
            "precision": d_precision,
            "recall": d_recall,
            "f1": d_f1,
            "accuracy": d_correct / d_total if d_total else 0.0,
            "first_sense_fallback": counts["first_sense_fallback"],
            "missing_prediction": counts["missing_prediction"],
        }

    result = {
        "overall": {
            "total": total,
            "answered": answered,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "missing_gold": missing_gold,
            "missing_prediction": missing_prediction,
            "first_sense_fallback": fallback_first_sense,
        },
        "by_dataset": by_dataset,
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def summarize_training(instances: Sequence[WSDInstance], train_mfs: Mapping[str, str]) -> Dict:
    tagged = [inst for inst in instances if inst.gold_sense_keys]
    unique_senses = {sense for inst in tagged for sense in inst.gold_sense_keys}
    return {
        "num_instances": len(instances),
        "num_tagged_instances": len(tagged),
        "num_lemma_pos_entries": len(train_mfs),
        "num_unique_senses": len(unique_senses),
        "datasets": dict(Counter(inst.dataset for inst in instances)),
    }


def main() -> None:
    args = parse_args()
    train_specs = list(args.train or [])
    eval_specs = list(args.eval or [])

    if args.raganato_root is not None:
        auto_train, auto_eval = autodiscover_raganato(args.raganato_root)
        train_specs = train_specs or auto_train
        eval_specs = eval_specs or auto_eval

    if not train_specs:
        raise ValueError("No training data supplied. Use --train or --raganato-root.")
    if not eval_specs:
        raise ValueError("No eval data supplied. Use --eval or --raganato-root.")

    train_instances = load_instances(train_specs, kind="train")
    eval_instances = load_instances(eval_specs, kind="eval")
    train_mfs = build_mfs_from_training(train_instances)
    first_sense = load_wordnet_first_sense(args.wordnet_index_sense) if args.wordnet_index_sense else {}
    if args.mfs_source in {"wordnet", "semcor_then_wordnet"} and not first_sense:
        raise ValueError(f"--mfs-source {args.mfs_source!r} requires --wordnet-index-sense")
    scores = score_instances(eval_instances, train_mfs, first_sense, args.mfs_source, args.include_predictions)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": f"mfs_{args.mfs_source}",
        "mfs_source": args.mfs_source,
        "train_specs": train_specs,
        "eval_specs": eval_specs,
        "wordnet_index_sense": str(args.wordnet_index_sense) if args.wordnet_index_sense else None,
        "train_summary": summarize_training(train_instances, train_mfs),
        "wordnet_first_sense_entries": len(first_sense),
        "scores": scores,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    overall = scores["overall"]
    print(
        "MFS WSD: "
        f"F1={overall['f1']:.4f} "
        f"accuracy={overall['accuracy']:.4f} "
        f"answered={overall['answered']}/{overall['total']} "
        f"first_sense_fallback={overall['first_sense_fallback']} "
        f"missing_prediction={overall['missing_prediction']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
