"""Regulator-validated therapeutic relations from FDA and EMA orphan designations.

An orphan designation is granted only after a regulator accepts the sponsor's
scientific case that a drug is plausible for a rare disease (FDA: 21 CFR
316.20, "scientific rationale" with supporting data; EMA: Regulation (EC)
No 141/2000, "medical plausibility" reviewed by the COMP).  Two rare diseases
become *therapeutically related* on the date the same drug first holds a
designation for both, i.e. when an independent expert body has accepted that
one molecule plausibly treats both.

The dates make the relations usable prospectively: relations established
before a cutoff can train a model, and relations first established after it
measure whether the similarity graph anticipated them.  None of these
relations is visible to the models through their inputs (see
``data_sources``); dated designations enter only as drug *history* before the
cutoff, with the drugs a pair shares removed (``features.pair_drug_features``).

Steps:

1. Parse FDA (1983-) and EMA (2000-) designations; EMA refusals are dropped.
2. Identify drugs: the ChEMBL parent molecule via Open Targets molecule names,
   synonyms and trade names (salt forms stripped), else a normalized-name key.
3. Map the designation wording to Orphanet cohort diseases with a tiered
   matcher over Orphanet names/synonyms and the exact synonyms of Mondo terms
   that Mondo declares equivalent to an Orphanet entity.  Small Orphanet groups
   expand to their member diseases; vague or broad wording stays unmapped.
4. Build dated disease-pair relations from distinct designation records.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd

from common import DATABASE_DIR, FEATURE_DIR, REGULATORY_DIR, REGULATORY_OUT_DIR, pair_key, timed, write_json
from data_sources import EXCLUDED_FLAGS, Bundle, attach_groups

MIN_SINGLE_TOKEN_MATCH = 7
MIN_KEY_LENGTH = 4
MAX_DISEASES_PER_DRUG = 12
MATCH_TIERS = ("exact", "qualifier", "list", "contained")

SPELLING = (
    (r"aem", "em"),
    (r"oedem", "edem"),
    (r"oesoph", "esoph"),
    (r"diarrhoea", "diarrhea"),
    (r"foet", "fet"),
    (r"oestr", "estr"),
    (r"coeliac", "celiac"),
    (r"amoeb", "ameb"),
    (r"dyspnoea", "dyspnea"),
    (r"apnoea", "apnea"),
    (r"tumour", "tumor"),
    (r"paediatric", "pediatric"),
    (r"orthopaed", "orthoped"),
    (r"gynaec", "gynec"),
    (r"sulph", "sulf"),
    (r"leucod", "leukod"),
    (r"leucoc", "leukoc"),
    (r"faec", "fec"),
)
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
PURPOSE_WORDS = frozenset(
    """treatment treatments treating treat prevention preventing prevent prophylaxis prophylactic diagnosis
    diagnostic diagnosing management managing of the and or for in adjunctive adjunct therapy therapies use as an a
    to reduction reducing reduce delay delaying improvement improving inhibition control maintenance induction
    amelioration relief slowing suppression palliation patients patient children adults adult pediatric infants
    neonates newborns subjects individuals people persons with suffering from who have has having those""".split()
)
QUALIFIERS = (
    ("associated", "with"),
    ("due", "to"),
    ("secondary", "to"),
    ("caused", "by"),
    ("resulting", "from"),
    ("other", "than"),
    ("at", "risk"),
    ("in",),
    ("following",),
    ("after",),
    ("including",),
    ("with",),
    ("who",),
    ("that",),
    ("requiring",),
    ("undergoing",),
    ("not",),
    ("except",),
)
LOOSE_DROP = frozenset({"disease", "syndrome", "disorder", "the", "of", "and", "with", "due", "to", "in", "a"})
# Leading words Orphanet adds that regulators usually omit ("Proximal spinal
# muscular atrophy", "Primary myelofibrosis").  Words that select a subset of
# the disease (hereditary gastric cancer, idiopathic bronchiectasis,
# congenital, juvenile, acute, ...) are deliberately absent.
# "non-small cell lung cancer" must not match "small cell lung cancer".
NEGATING_PREFIXES = frozenset({"non", "not", "without", "excluding", "except"})
OPTIONAL_MODIFIERS = frozenset(
    {"primary", "proximal", "inherited", "classic", "classical", "isolated", "sporadic", "malignant", "skeletal", "typical"}
)


def normalize_text(value: str) -> str:
    """Lowercase ASCII words with US spelling, no possessives, and 'type 2' for 'type II'."""

    value = unicodedata.normalize("NFKD", str(value)).replace("\u2019", "'").replace("\u2018", "'")
    value = value.encode("ascii", "ignore").decode().lower()
    value = re.sub(r"'s\b", "", value)
    value = re.sub(r"s'(?=\s|$)", "s", value)
    for pattern, replacement in SPELLING:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\btype (i{1,3}|iv|vi{0,3}|ix|x)\b", lambda m: "type " + ROMAN[m.group(1)], value)
    return " ".join(value.split())


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is", "us", "os")):
        return token[:-1]
    return token


def loose_key(normalized: str) -> str:
    """Order-free key without filler words, plurals, or apostrophe-less possessives."""

    tokens = [_singular(t) for t in normalized.split() if t not in LOOSE_DROP]
    return " ".join(sorted(tokens))


# --------------------------------------------------------------------------- disease lexicon


@dataclass
class OrphanetCatalog:
    """Orphanet entities: names, node candidates (diseases and groups), and descendants."""

    names: dict[str, str]
    diseases: frozenset[str]
    groups: frozenset[str]
    excluded: frozenset[str]
    children: dict[str, set[str]]
    _descendants: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def descendants(self, orpha: str) -> frozenset[str]:
        cached = self._descendants.get(orpha)
        if cached is not None:
            return cached
        found: set[str] = set()
        stack = [orpha]
        while stack:
            for child in self.children.get(stack.pop(), ()):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        result = frozenset(found)
        self._descendants[orpha] = result
        return result

    def nested(self, a: str, b: str) -> bool:
        """True when one entity is an Orphanet descendant of the other."""

        return b in self.descendants(a) or a in self.descendants(b)

    def resolve(self, orpha: str) -> tuple[set[str], str]:
        """Graph nodes for a matched entity, and how they were reached."""

        if orpha in self.diseases:
            return {orpha}, ""
        if orpha in self.groups:
            return {orpha}, "group"
        if orpha in self.excluded:
            return set(), "excluded entity"
        members = self.descendants(orpha) & self.diseases
        if len(members) == 1:
            return set(members), "single member"
        return set(), "no cohort member" if not members else f"group with {len(members)} members"


def _parse_mondo_orphanet_synonyms(path) -> dict[str, set[str]]:
    """Mondo label and exact (non-abbreviation) synonyms for terms equivalent to an Orphanet entity."""

    result: dict[str, set[str]] = defaultdict(set)
    term: dict[str, Any] = {}

    def flush() -> None:
        if term.get("id", "").startswith("MONDO:") and not term.get("obsolete") and term.get("orpha"):
            for orpha in term["orpha"]:
                result[orpha].update(term["names"])

    in_term = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if in_term:
                    flush()
                in_term = line == "[Term]"
                term = {"names": set(), "orpha": set(), "obsolete": False}
                continue
            if not in_term or not line:
                continue
            key, _, value = line.partition(": ")
            if key == "id":
                term["id"] = value.strip()
            elif key == "name":
                term["names"].add(value.strip())
            elif key == "synonym" and " EXACT" in value and "ABBREVIATION" not in value:
                match = re.match(r'"(.*)"\s+EXACT', value)
                if match:
                    term["names"].add(match.group(1))
            elif key == "xref" and value.startswith("Orphanet:") and "MONDO:equivalentTo" in value:
                term["orpha"].add("ORPHA:" + value.split()[0].split(":")[1])
            elif key == "is_obsolete" and value.strip() == "true":
                term["obsolete"] = True
    if in_term:
        flush()
    return result


class DiseaseMatcher:
    """Tiered mapping of free-text indications to graph nodes (Orphanet diseases or groups).

    Names are searched in priority order -- Orphanet preferred names, then
    Orphanet synonyms, then Mondo exact synonyms -- so a looser source can never
    override or make ambiguous what Orphanet itself names.
    """

    def __init__(self, catalog: OrphanetCatalog, name_levels: list[dict[str, set[str]]], heads: Iterable[str]):
        self.catalog = catalog
        heads = set(heads)
        self.exact: list[dict[str, set[str]]] = []
        self.loose: list[dict[str, set[str]]] = []
        for level in name_levels:
            exact: dict[str, set[str]] = defaultdict(set)
            loose: dict[str, set[str]] = defaultdict(set)
            for orpha, names in level.items():
                if orpha in heads:
                    continue
                for name in names:
                    key = normalize_text(name)
                    if len(key) < MIN_KEY_LENGTH or key.isdigit():
                        continue
                    exact[key].add(orpha)
                    loose_value = loose_key(key)
                    if len(loose_value) >= MIN_KEY_LENGTH:
                        loose[loose_value].add(orpha)
            self.exact.append(dict(exact))
            self.loose.append(dict(loose))
        self.any_exact: dict[str, set[str]] = {}
        for level in reversed(self.exact):
            self.any_exact.update(level)
        self.max_key_tokens = max(len(k.split()) for k in self.any_exact)

    def _resolve_ids(self, ids: set[str]) -> tuple[set[str], str]:
        resolved: dict[str, tuple[set[str], str]] = {}
        notes = []
        for orpha in ids:
            members, how = self.catalog.resolve(orpha)
            if members:
                resolved[orpha] = (members, how)
            else:
                notes.append(how)
        if not resolved:
            return set(), notes[0] if len(notes) == 1 else "unresolvable"
        if len(resolved) == 1:
            return next(iter(resolved.values()))
        # Several entities share the name: accept one that contains all the others.
        for orpha, (members, how) in resolved.items():
            if all(other == orpha or other in self.catalog.descendants(orpha) for other in resolved):
                return members, how
        return set(), "ambiguous"

    def lookup(self, normalized: str) -> tuple[set[str], str]:
        for level in self.exact:
            if normalized in level:
                return self._resolve_ids(level[normalized])
        loose = loose_key(normalized)
        if len(loose) >= MIN_KEY_LENGTH:
            for level in self.loose:
                if loose in level:
                    members, how = self._resolve_ids(level[loose])
                    return members, ("loose " + how).strip() if members else how
        return set(), "no match"

    @staticmethod
    def purpose_variants(tokens: list[str]) -> list[list[str]]:
        """The token list with 0, 1, 2, ... leading purpose words removed (longest first)."""

        variants = [tokens]
        i = 0
        while i < len(tokens) and tokens[i] in PURPOSE_WORDS:
            i += 1
            if i < len(tokens):
                variants.append(tokens[i:])
        return variants

    @staticmethod
    def qualifier_split(tokens: list[str]) -> list[list[str]]:
        """Head before the first qualifier and every tail after a qualifier."""

        parts: list[list[str]] = []
        for i in range(1, len(tokens)):
            for qualifier in QUALIFIERS:
                n = len(qualifier)
                if tuple(tokens[i : i + n]) == qualifier:
                    if not parts:
                        parts.append(tokens[:i])
                    if i + n < len(tokens):
                        parts.append(tokens[i + n :])
                    break
        return parts

    def map_text(self, text: str) -> tuple[set[str], str, str]:
        """Return (cohort diseases, tier, note) for one designation wording."""

        normalized = normalize_text(text)
        tokens = normalized.split()
        if not tokens:
            return set(), "", "empty"
        variants = self.purpose_variants(tokens)
        notes: list[str] = []

        for variant in variants:
            members, how = self.lookup(" ".join(variant))
            if members:
                return members, "exact", how
            notes.append(how)

        for variant in variants[-1:]:
            for part in self.qualifier_split(variant):
                for sub in self.purpose_variants(part):
                    members, how = self.lookup(" ".join(sub))
                    if members:
                        return members, "qualifier", how

        # "X and Y": every multi-word piece must name a disease, otherwise an
        # incidental mention could stand in for an unmapped main indication.
        core = " ".join(variants[-1])
        pieces = [p.split() for p in re.split(r"\b(?:and|or)\b", core) if p.strip()]
        if len(pieces) > 1:
            found: set[str] = set()
            complete = True
            for piece in pieces:
                matched = False
                for sub in self.purpose_variants(piece):
                    members, how = self.lookup(" ".join(sub))
                    if members:
                        found |= members
                        matched = True
                        break
                if not matched and len(piece) > 1:
                    complete = False
            if found and complete:
                return found, "list", ""

        found = self._contained(variants[-1])
        if found:
            return found, "contained", ""
        informative = [n for n in notes if n != "no match"]
        return set(), "", informative[0] if informative else "no match"

    def _contained(self, tokens: list[str]) -> set[str]:
        found: set[str] = set()
        covered = [False] * len(tokens)
        for length in range(min(self.max_key_tokens, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - length + 1):
                if any(covered[start : start + length]):
                    continue
                if start > 0 and tokens[start - 1] in NEGATING_PREFIXES:
                    continue
                key = " ".join(tokens[start : start + length])
                if length == 1 and len(key) < MIN_SINGLE_TOKEN_MATCH:
                    continue
                if key not in self.any_exact:
                    continue
                members, _ = self._resolve_ids(self.any_exact[key])
                if members:
                    found |= members
                    for i in range(start, start + length):
                        covered[i] = True
        return found


def load_catalog(bundle: Bundle) -> tuple[OrphanetCatalog, list[dict[str, set[str]]]]:
    """The Orphanet catalog and disease names in priority order
    (Orphanet preferred names, Orphanet synonyms, Mondo exact synonyms)."""

    diseases = pd.read_parquet(FEATURE_DIR / "diseases.parquet", columns=["orpha_id", "name", "synonyms", "flags"])
    flags = diseases["flags"].map(lambda v: set(v.tolist()) if v is not None else set())
    excluded = set(diseases.loc[flags.map(lambda v: bool(v & EXCLUDED_FLAGS)), "orpha_id"])
    children: dict[str, set[str]] = defaultdict(set)
    for parent, child in bundle.labels.orphanet_edges:
        children[parent].add(child)
    groups = frozenset(bundle.group_members)
    catalog = OrphanetCatalog(
        names=dict(zip(diseases["orpha_id"], diseases["name"])),
        diseases=frozenset(d for d in bundle.records if d not in groups),
        groups=groups,
        excluded=frozenset(excluded - set(bundle.records)),
        children=dict(children),
    )
    preferred: dict[str, set[str]] = defaultdict(set)
    synonyms: dict[str, set[str]] = defaultdict(set)
    for row in diseases.itertuples(index=False):
        preferred[row.orpha_id].add(row.name)
        for synonym in row.synonyms if row.synonyms is not None else []:
            if synonym:
                synonyms[row.orpha_id].add(synonym)
    with timed("Reading Mondo exact synonyms of Orphanet-equivalent terms"):
        mondo = _parse_mondo_orphanet_synonyms(DATABASE_DIR / "mondo" / "mondo.obo")
    unmodified: dict[str, set[str]] = defaultdict(set)
    for orpha in catalog.diseases | catalog.groups:
        words = catalog.names.get(orpha, "").split(maxsplit=1)
        if len(words) == 2 and words[0].lower() in OPTIONAL_MODIFIERS:
            unmodified[orpha].add(words[1])
    return catalog, [dict(preferred), dict(synonyms), dict(mondo), dict(unmodified)]


# --------------------------------------------------------------------------- designations


def _parse_date(values: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format="%d/%m/%Y" if dayfirst else "%m/%d/%Y")


def read_fda() -> pd.DataFrame:
    table = pd.read_html(REGULATORY_DIR / "fda_orphan_designations.xls")[0]
    status = table["Orphan Designation Status"].fillna("")
    frame = pd.DataFrame(
        {
            "source": "FDA",
            "drug_name": table["Generic Name"].fillna("").astype(str).str.strip(),
            "trade_name": table["Trade Name"].fillna("").astype(str).str.strip(),
            "date": _parse_date(table["Date Designated"], dayfirst=False),
            "text": table["Orphan Designation"].fillna("").astype(str).str.strip(),
            "status": status,
            "approved": status.str.contains("Approved"),
            "approval_date": _parse_date(table["Marketing Approval Date"], dayfirst=False),
        }
    )
    # Approved designations repeat once per marketing approval; keep the earliest approval.
    frame = frame.sort_values("approval_date").drop_duplicates(["drug_name", "date", "text"], keep="first")
    frame["record_id"] = [f"FDA:{i}" for i in range(len(frame))]
    return frame


def read_ema() -> pd.DataFrame:
    document = json.loads((REGULATORY_DIR / "ema_orphan_designations.json").read_text(encoding="utf-8"))
    table = pd.DataFrame(document["data"])
    table = table[table["status"].ne("Negative")]
    return pd.DataFrame(
        {
            "source": "EMA",
            "drug_name": table["active_substance"].fillna("").astype(str).str.strip(),
            "trade_name": table["medicine_name"].fillna("").astype(str).str.strip(),
            "date": _parse_date(table["date_of_designation_or_refusal"], dayfirst=True),
            "text": table["intended_use"].fillna("").astype(str).str.strip(),
            "status": table["status"],
            "approved": table["medicine_name"].fillna("").ne(""),
            "approval_date": pd.NaT,
            "record_id": "EMA:" + table["eu_designation_number"].astype(str),
        }
    )


def build_designations(bundle: Bundle) -> pd.DataFrame:
    """All designations with drug identifiers and mapped cohort diseases."""

    catalog, name_levels = load_catalog(bundle)
    matcher = DiseaseMatcher(catalog, name_levels, bundle.knowledge.orphanet_heads)
    with timed("Reading FDA and EMA orphan designations"):
        frame = pd.concat([read_fda(), read_ema()], ignore_index=True)
        frame = frame[frame["date"].notna() & frame["text"].ne("") & frame["drug_name"].ne("")].reset_index(drop=True)
    knowledge = bundle.knowledge
    with timed(f"Identifying drugs and mapping {len(frame):,} designations to Orphanet"):
        drug_ids, resolved, mapped, tiers, notes = [], [], [], [], []
        cache: dict[str, tuple[set[str], str, str]] = {}
        for row in frame.itertuples(index=False):
            chembl = knowledge.resolve_drug(row.drug_name) or (knowledge.resolve_drug(row.trade_name) if row.trade_name else None)
            drug_ids.append(chembl or knowledge.drug_identifier(row.drug_name))
            resolved.append(chembl is not None)
            if row.text not in cache:
                cache[row.text] = matcher.map_text(row.text)
            members, tier, note = cache[row.text]
            mapped.append(sorted(members))
            tiers.append(tier)
            notes.append(note)
    frame["drug_id"] = drug_ids
    frame["drug_resolved"] = resolved
    frame["orpha_ids"] = mapped
    frame["match_tier"] = tiers
    frame["match_note"] = notes
    frame["mapped_names"] = [[catalog.names.get(o, o) for o in ids] for ids in mapped]
    frame = frame[frame["drug_id"].notna()].reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- relations


@dataclass
class RegulatoryData:
    designations: pd.DataFrame
    relations: pd.DataFrame
    # disease -> drug -> first designation date (all designations, any breadth).
    history: dict[str, dict[str, pd.Timestamp]]
    broad_drugs: frozenset[str]
    catalog: OrphanetCatalog

    def known_before(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        return self.relations[self.relations["date"] < cutoff]

    def new_after(self, cutoff: pd.Timestamp, until: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        keep = self.relations["date"] >= cutoff
        if until is not None:
            keep &= self.relations["date"] < until
        return self.relations[keep]

    def drugs_before(self, cutoff: Optional[pd.Timestamp]) -> dict[str, dict[str, float]]:
        """Designated drugs per disease before ``cutoff`` (all drugs when None)."""

        result: dict[str, dict[str, float]] = {}
        for disease, drugs in self.history.items():
            kept = {drug: 1.0 for drug, date in drugs.items() if cutoff is None or date < cutoff}
            if kept:
                result[disease] = kept
        return result


def build_relations(
    designations: pd.DataFrame,
    catalog: OrphanetCatalog,
    max_diseases_per_drug: int = MAX_DISEASES_PER_DRUG,
) -> tuple[pd.DataFrame, frozenset[str]]:
    """Dated disease pairs sharing a designated drug, from distinct designation records.

    A pair formed only inside one multi-disease designation record is skipped:
    the relation must be backed by two separate regulatory decisions.  Pairs
    where one node contains the other (a group and its member) are trivial and
    skipped.  Drugs designated for more than ``max_diseases_per_drug`` nodes
    (broad agents such as checkpoint inhibitors) carry little pair-specific
    signal and do not create relations.
    """

    mapped = designations[designations["orpha_ids"].map(len) > 0]
    per_drug: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in mapped.itertuples(index=False):
        for disease in row.orpha_ids:
            entry = per_drug[row.drug_id].setdefault(
                disease, {"date": row.date, "records": set(), "sources": set(), "approved": False}
            )
            entry["date"] = min(entry["date"], row.date)
            entry["records"].add(row.record_id)
            entry["sources"].add(row.source)
            entry["approved"] |= bool(row.approved)
    broad = frozenset(d for d, diseases in per_drug.items() if len(diseases) > max_diseases_per_drug)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for drug, diseases in per_drug.items():
        if drug in broad or len(diseases) < 2:
            continue
        for a, b in itertools.combinations(sorted(diseases), 2):
            ea, eb = diseases[a], diseases[b]
            if ea["records"] == eb["records"] and len(ea["records"]) == 1:
                continue
            if catalog.nested(a, b):
                continue
            date = max(ea["date"], eb["date"])
            entry = pairs.setdefault(
                pair_key(a, b), {"date": date, "drugs": [], "sources": set(), "approved_both": False}
            )
            entry["date"] = min(entry["date"], date)
            entry["drugs"].append(drug)
            entry["sources"] |= ea["sources"] | eb["sources"]
            entry["approved_both"] |= ea["approved"] and eb["approved"]
    rows = [
        {
            "a": a,
            "b": b,
            "date": v["date"],
            "year": v["date"].year,
            "n_drugs": len(v["drugs"]),
            "drugs": sorted(v["drugs"]),
            "sources": "+".join(sorted(v["sources"])),
            "approved_both": v["approved_both"],
        }
        for (a, b), v in pairs.items()
    ]
    relations = pd.DataFrame(rows).sort_values(["date", "a", "b"]).reset_index(drop=True)
    return relations, broad


def build_history(designations: pd.DataFrame) -> dict[str, dict[str, pd.Timestamp]]:
    history: dict[str, dict[str, pd.Timestamp]] = defaultdict(dict)
    for row in designations[designations["orpha_ids"].map(len) > 0].itertuples(index=False):
        for disease in row.orpha_ids:
            current = history[disease].get(row.drug_id)
            if current is None or row.date < current:
                history[disease][row.drug_id] = row.date
    return dict(history)


def tree_catalog(bundle: Bundle) -> OrphanetCatalog:
    """A catalog without names, enough for containment checks."""

    children: dict[str, set[str]] = defaultdict(set)
    for parent, child in bundle.labels.orphanet_edges:
        children[parent].add(child)
    groups = frozenset(bundle.group_members)
    return OrphanetCatalog(
        names={},
        diseases=frozenset(d for d in bundle.records if d not in groups),
        groups=groups,
        excluded=frozenset(),
        children=dict(children),
    )


def load_regulatory(bundle: Bundle, rebuild: bool = False) -> RegulatoryData:
    """Designations and dated relations; designated Orphanet groups become nodes of ``bundle``."""

    path = REGULATORY_OUT_DIR / "designations.parquet"
    if path.is_file() and not rebuild:
        designations = pd.read_parquet(path)
        designations["orpha_ids"] = designations["orpha_ids"].map(list)
    else:
        designations = build_designations(bundle)
        REGULATORY_OUT_DIR.mkdir(parents=True, exist_ok=True)
        designations.to_parquet(path, index=False)
    attach_groups(bundle, {o for ids in designations["orpha_ids"] for o in ids if bundle.is_group(o)})
    catalog = tree_catalog(bundle)
    relations, broad = build_relations(designations, catalog)
    return RegulatoryData(designations, relations, build_history(designations), broad, catalog)


def with_history(records: dict[str, dict[str, Any]], drugs: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Shallow copies of the records carrying the given designated drugs."""

    return {d: {**r, "drugs": dict(drugs.get(d, {}))} for d, r in records.items()}


def summarize(data: RegulatoryData) -> dict[str, Any]:
    d = data.designations
    mapped = d[d["orpha_ids"].map(len) > 0]
    r = data.relations
    by_year = r.groupby("year").size()
    nodes = {o for ids in mapped["orpha_ids"] for o in ids}
    return {
        "designations": int(len(d)),
        "by_source": d["source"].value_counts().to_dict(),
        "mapped_designations": int(len(mapped)),
        "mapped_fraction": round(len(mapped) / max(len(d), 1), 3),
        "mapped_by_tier": mapped["match_tier"].value_counts().to_dict(),
        "unmapped_reasons": d.loc[d["orpha_ids"].map(len) == 0, "match_note"].value_counts().head(12).to_dict(),
        "drugs": int(d["drug_id"].nunique()),
        "drugs_resolved_to_chembl": round(float(d["drug_resolved"].mean()), 3),
        "diseases_with_designation": len(nodes),
        "groups_with_designation": len(nodes & data.catalog.groups),
        "broad_drugs_excluded": len(data.broad_drugs),
        "relations": int(len(r)),
        "relation_diseases": int(len(set(r["a"]) | set(r["b"]))),
        "relations_approved_both": int(r["approved_both"].sum()),
        "relations_by_year": {int(k): int(v) for k, v in by_year.items()},
    }


def main() -> int:
    import argparse

    from common import configure_stdout
    from data_sources import load_bundle

    configure_stdout()
    parser = argparse.ArgumentParser(description="Build the dated regulatory relation ground truth.")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    bundle = load_bundle()
    data = load_regulatory(bundle, rebuild=args.rebuild)
    summary = summarize(data)
    write_json(REGULATORY_OUT_DIR / "summary.json", summary)
    d = data.designations
    unmapped = d[d["orpha_ids"].map(len) == 0]
    unmapped_counts = (
        unmapped.assign(normalized=unmapped["text"].map(normalize_text))
        .groupby("normalized")
        .agg(count=("text", "size"), example=("text", "first"), reason=("match_note", "first"))
        .sort_values("count", ascending=False)
    )
    unmapped_counts.to_csv(REGULATORY_OUT_DIR / "unmapped_indications.csv")
    mapped = d[d["orpha_ids"].map(len) > 0]
    audit = mapped.sample(min(200, len(mapped)), random_state=7)[
        ["record_id", "source", "drug_name", "drug_id", "date", "text", "match_tier", "match_note", "mapped_names"]
    ]
    audit.to_csv(REGULATORY_OUT_DIR / "mapping_audit_sample.csv", index=False)
    data.relations.to_csv(REGULATORY_OUT_DIR / "relations.csv", index=False)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
