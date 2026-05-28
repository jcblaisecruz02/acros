#!/usr/bin/env python
"""Build a stable CoInCo lexical-substitution case file.

The raw CoInCo XML is kept as the provenance source. This script only converts
the XML into a JSON payload that is convenient for steering experiments; it does
not apply tokenizer-specific filtering.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_XML_GZ = Path("evals/data/coinco/coinco.xml.gz")
DEFAULT_DEV_IDS = Path("evals/data/coinco/devset-tokenIDs.txt")
DEFAULT_TEST_IDS = Path("evals/data/coinco/testset-tokenIDs.txt")
DEFAULT_OUT = Path("evals/data/coinco/coinco_lexsub_cases.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-gz", type=Path, default=DEFAULT_XML_GZ)
    p.add_argument("--dev-ids", type=Path, default=DEFAULT_DEV_IDS)
    p.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--unique-wordform-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only targets whose wordform occurs once in the CoInCo tokenized target sentence.",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def split_for_id(token_id: str, dev_ids: set[str], test_ids: set[str]) -> str:
    if token_id in dev_ids:
        return "dev"
    if token_id in test_ids:
        return "test"
    return "unlisted"


def build_cases(args: argparse.Namespace) -> dict:
    dev_ids = load_ids(args.dev_ids)
    test_ids = load_ids(args.test_ids)

    cases = []
    raw_target_count = 0
    problematic_count = 0
    no_substitution_count = 0
    repeated_wordform_count = 0
    split_counts: Counter[str] = Counter()
    pos_counts: Counter[str] = Counter()

    with gzip.open(args.xml_gz, "rt", encoding="utf-8") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    for sent_idx, sent in enumerate(root.findall("sent")):
        masc_file = sent.attrib.get("MASCfile", "")
        masc_sent_id = sent.attrib.get("MASCsentID", "")
        precontext = clean_text(sent.findtext("precontext"))
        sentence = clean_text(sent.findtext("targetsentence"))
        postcontext = clean_text(sent.findtext("postcontext"))
        tokens_elem = sent.find("tokens")
        token_elems = list(tokens_elem) if tokens_elem is not None else []
        token_wordforms = [tok.attrib.get("wordform", "") for tok in token_elems]
        wordform_counts = Counter(w.lower() for w in token_wordforms)

        for token_idx, tok in enumerate(token_elems):
            token_id = tok.attrib.get("id", "")
            if not token_id or token_id == "XXX":
                continue
            raw_target_count += 1

            wordform = tok.attrib.get("wordform", "")
            lemma = tok.attrib.get("lemma", "")
            pos_masc = tok.attrib.get("posMASC", "")
            pos_tt = tok.attrib.get("posTT", "")
            problematic = tok.attrib.get("problematic", "no")
            if problematic == "yes":
                problematic_count += 1

            substs = []
            substitutions = tok.find("substitutions")
            if substitutions is not None:
                for subst in substitutions.findall("subst"):
                    subst_lemma = subst.attrib.get("lemma", "").strip()
                    if not subst_lemma:
                        continue
                    try:
                        freq = int(subst.attrib.get("freq", "1"))
                    except ValueError:
                        freq = 1
                    substs.append(
                        {
                            "lemma": subst_lemma,
                            "pos": subst.attrib.get("pos", ""),
                            "freq": freq,
                        }
                    )
            if not substs:
                no_substitution_count += 1

            unique_wordform = wordform_counts[wordform.lower()] == 1
            if args.unique_wordform_only and not unique_wordform:
                repeated_wordform_count += 1
                continue

            split = split_for_id(token_id, dev_ids, test_ids)
            split_counts[split] += 1
            pos_counts[pos_tt or pos_masc or ""] += 1
            cases.append(
                {
                    "id": f"coinco:{token_id}",
                    "source_token_id": token_id,
                    "split": split,
                    "sent_index": sent_idx,
                    "token_index": token_idx,
                    "MASCfile": masc_file,
                    "MASCsentID": masc_sent_id,
                    "precontext": precontext,
                    "sentence": sentence,
                    "postcontext": postcontext,
                    "wordform": wordform,
                    "lemma": lemma,
                    "posMASC": pos_masc,
                    "posTT": pos_tt,
                    "problematic": problematic,
                    "unique_wordform_in_sentence": unique_wordform,
                    "substitutes": substs,
                }
            )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder": "evals/build_coinco_lexsub_cases.py",
        "source": {
            "name": "CoInCo: Concepts in Context",
            "citation": "Kremer, Erk, Pado, and Thater. 2014. What Substitutes Tell Us - Analysis of an All-Words Lexical Substitution Corpus. EACL.",
            "data_url": "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/coinco/coinco.xml.gz",
            "readme_url": "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/coinco/README.txt",
            "license": "CC-BY-3.0-US",
            "xml_gz": str(args.xml_gz),
            "xml_gz_sha256": sha256(args.xml_gz),
            "dev_ids": str(args.dev_ids),
            "dev_ids_sha256": sha256(args.dev_ids),
            "test_ids": str(args.test_ids),
            "test_ids_sha256": sha256(args.test_ids),
        },
        "filters": {
            "unique_wordform_only": bool(args.unique_wordform_only),
        },
        "stats": {
            "raw_target_count": raw_target_count,
            "problematic_count": problematic_count,
            "no_substitution_count": no_substitution_count,
            "repeated_wordform_filtered_count": repeated_wordform_count,
            "kept_case_count": len(cases),
            "split_counts": dict(sorted(split_counts.items())),
            "pos_counts": dict(sorted(pos_counts.items())),
        },
        "cases": cases,
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = build_cases(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(payload['cases'])} CoInCo cases to {args.out}")
    print(json.dumps(payload["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
