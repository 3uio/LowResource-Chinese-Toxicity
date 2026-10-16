"""Portable corpus isolation check; raw benchmark files are not needed on GPU hosts."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import zhconv

ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]")


def normalized_hash(text):
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    text = "".join(c for c in text if not c.isspace() and unicodedata.category(c)[0] != "P")
    return hashlib.sha256(zhconv.convert(text.lower(), "zh-cn").encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_corpus(corpus, audit_path, frozen_path):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if (audit.get("schema_version") != 1 or audit.get("passed") is not True
            or audit.get("policy") != "official_dev_test_and_full_CNTP_and_table2_reserved"):
        raise ValueError("Invalid corpus isolation policy/audit")
    if sha256_file(corpus) != audit["corpus_sha256"]:
        raise ValueError("Corpus fingerprint changed; rebuild and independently verify first")
    if sha256_file(frozen_path) != audit["frozen_sha256"]:
        raise ValueError("Frozen hash fingerprint changed")
    frozen = set(json.loads(Path(frozen_path).read_text(encoding="utf-8")))
    rows, ids = [], set()
    with open(corpus, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Empty/invalid text at line %d" % line_number)
            if not row.get("id") or row["id"] in ids:
                raise ValueError("Missing/duplicate id at line %d" % line_number)
            if (row.get("source") == "CNTP" or row.get("source_split") in {"dev", "test"}
                    or normalized_hash(row["text"]) in frozen):
                raise ValueError("Frozen test/seed leaked into corpus at line %d" % line_number)
            ids.add(row["id"])
            rows.append(row)
    if not rows or len(rows) != audit["rows"]:
        raise ValueError("Corpus row count does not match audit")
    return rows, audit
