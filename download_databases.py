"""
This script downloads the databases for the rare disease similarity detection project,
and puts them into .data/databases/

The default download is the relatively small set needed by the initial features in
PROJECT.md. Large corpora and full database mirrors require explicit command-line
flags; run ``python download_databases.py --help`` for details.
"""

import argparse
import concurrent.futures
import hashlib
import html.parser
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urljoin, urlparse

import requests


USER_AGENT = (
    "rare-disease-similarity-downloader/1.0 "
    "(research use; https://github.com/)"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / ".data" / "databases"
CORE_GROUPS = (
    "orphadata",
    "hpo",
    "mondo",
    "clingen",
    "clinvar",
    "opentargets",
    "regulatory",
)
OPEN_TARGETS_DATASETS = (
    "disease",
    "disease_hpo",
    "disease_phenotype",
    "target",
    "association_overall_direct",
    "drug_molecule",
    "clinical_indication",
    "drug_mechanism_of_action",
    "reactome",
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_REQUEST_ATTEMPTS = 4
CHUNK_SIZE = 1024 * 1024
thread_local = threading.local()


@dataclass(frozen=True)
class RemoteFile:
    """A remote file and its destination relative to the database directory."""

    source: str
    url: str
    relative_path: str
    estimated_size: Optional[int] = None
    checksum_url: Optional[str] = None
    method: str = "GET"
    form_data: Optional[Dict[str, str]] = None


@dataclass
class DownloadResult:
    """The outcome of downloading one file."""

    source: str
    url: str
    relative_path: str
    status: str
    size: int
    error: Optional[str] = None


@dataclass(frozen=True)
class ListingEntry:
    """One file or directory found in an HTTP directory listing."""

    href: str
    size: Optional[int]


class LinkParser(html.parser.HTMLParser):
    """Collect links without depending on BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def core_static_files() -> Dict[str, List[RemoteFile]]:
    """Return the fixed portion of the default manifest."""

    orphadata_base = "https://www.orphadata.com/data"
    hpo_base = "https://purl.obolibrary.org/obo/hp"
    clinvar_base = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar"
    clinvar_tab = f"{clinvar_base}/tab_delimited"

    clinvar_names = (
        "allele_gene.txt.gz",
        "hgvs4variation.txt.gz",
        "submission_summary.txt.gz",
        "variant_summary.txt.gz",
        "variation_allele.txt.gz",
    )
    clinvar_files = [
        RemoteFile(
            "clinvar",
            f"{clinvar_tab}/{name}",
            f"clinvar/tab_delimited/{name}",
            checksum_url=f"{clinvar_tab}/{name}.md5",
        )
        for name in clinvar_names
    ]
    clinvar_files.extend(
        [
            RemoteFile(
                "clinvar",
                f"{clinvar_base}/disease_names",
                "clinvar/disease_names",
            ),
            RemoteFile(
                "clinvar",
                f"{clinvar_base}/gene_condition_source_id",
                "clinvar/gene_condition_source_id",
            ),
        ]
    )

    fda_form = {
        "Product_name": "",
        "sponsor_name": "",
        "Designation": "",
        "Designation_Start_Date": "01/01/1983",
        "Designation_End_Date": date.today().strftime("%m/%d/%Y"),
        "Search_param": "DESDATE",
        "Output_Format": "Excel",
        "Sort_order": "GENERIC_NAME",
        "RecordsPerPage": "25",
        "newSearch": "Run Search",
    }

    return {
        "orphadata": [
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product1.xml",
                "orphadata/en_product1.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product4.xml",
                "orphadata/en_product4.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product6.xml",
                "orphadata/en_product6.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product9_prev.xml",
                "orphadata/en_product9_prev.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product9_ages.xml",
                "orphadata/en_product9_ages.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_funct_consequences.xml",
                "orphadata/en_funct_consequences.xml",
            ),
            RemoteFile(
                "orphadata",
                f"{orphadata_base}/xml/en_product7.xml",
                "orphadata/en_product7.xml",
            ),
        ],
        "hpo": [
            RemoteFile(
                "hpo",
                "https://purl.obolibrary.org/obo/hp.obo",
                "hpo/hp.obo",
            ),
            RemoteFile(
                "hpo",
                f"{hpo_base}/phenotype.hpoa",
                "hpo/phenotype.hpoa",
            ),
            RemoteFile(
                "hpo",
                f"{hpo_base}/genes_to_phenotype.txt",
                "hpo/genes_to_phenotype.txt",
            ),
            RemoteFile(
                "hpo",
                f"{hpo_base}/genes_to_disease.txt",
                "hpo/genes_to_disease.txt",
            ),
        ],
        "mondo": [
            RemoteFile(
                "mondo",
                "https://purl.obolibrary.org/obo/mondo.obo",
                "mondo/mondo.obo",
            )
        ],
        "clingen": [
            RemoteFile(
                "clingen",
                "https://search.clinicalgenome.org/kb/gene-validity/download",
                "clingen/gene_disease_validity.csv",
            ),
            RemoteFile(
                "clingen",
                "https://search.clinicalgenome.org/kb/gene-dosage/download",
                "clingen/gene_dosage.csv",
            ),
        ],
        "clinvar": clinvar_files,
        "regulatory": [
            RemoteFile(
                "regulatory",
                (
                    "https://www.ema.europa.eu/en/documents/report/"
                    "medicines-output-orphan_designations-json-report_en.json"
                ),
                "regulatory/ema_orphan_designations.json",
            ),
            RemoteFile(
                "regulatory",
                (
                    "https://www.accessdata.fda.gov/scripts/opdlisting/oopd/"
                    "OOPD_Results.cfm"
                ),
                "regulatory/fda_orphan_designations.xls",
                estimated_size=10 * 1024 * 1024,
                method="POST",
                form_data=fda_form,
            ),
        ],
    }


def get_session() -> requests.Session:
    """Return one configured HTTP session per worker thread."""

    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            }
        )
        thread_local.session = session
    return session


def request_with_retries(
    method: str,
    url: str,
    *,
    stream: bool = False,
    form_data: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    accepted_statuses: Sequence[int] = (),
) -> requests.Response:
    """Make an HTTP request with bounded exponential backoff."""

    last_error: Optional[BaseException] = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        response: Optional[requests.Response] = None
        try:
            response = get_session().request(
                method,
                url,
                data=form_data,
                headers=headers,
                stream=stream,
                timeout=(20, timeout),
                allow_redirects=True,
            )
            if response.status_code in accepted_statuses:
                return response
            if response.status_code in RETRYABLE_STATUS_CODES:
                response.close()
                raise requests.HTTPError(
                    f"temporary HTTP {response.status_code} for {url}"
                )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if response is not None:
                response.close()
            if attempt == MAX_REQUEST_ATTEMPTS - 1:
                break
            delay = (2**attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"request failed for {url}: {last_error}") from last_error


def parse_size(value: str) -> Optional[int]:
    """Parse either an exact byte count or an Apache-style human size."""

    value = value.strip().replace(",", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", value, re.I)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }[unit]
    return int(amount * multiplier)


def parse_directory_listing(document: str) -> List[ListingEntry]:
    """Parse links and approximate sizes from common Apache/NCBI listings."""

    parser = LinkParser()
    parser.feed(document)
    sizes: Dict[str, int] = {}

    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", document, re.I | re.S):
        link_match = re.search(r'href=["\']([^"\']+)["\']', row, re.I)
        if not link_match:
            continue
        cells = [
            re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", "").strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.I | re.S)
        ]
        for cell in reversed(cells):
            size = parse_size(cell)
            if size is not None:
                sizes[link_match.group(1)] = size
                break

    preformatted_pattern = re.compile(
        r'href=["\']([^"\']+)["\'][^>]*>.*?</a>'
        r"\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+([0-9.]+[KMGTPE]?)",
        re.I,
    )
    for href, size_text in preformatted_pattern.findall(document):
        size = parse_size(size_text)
        if size is not None:
            sizes[href] = size

    entries: List[ListingEntry] = []
    seen = set()
    for href in parser.links:
        if href in seen:
            continue
        seen.add(href)
        entries.append(ListingEntry(href=href, size=sizes.get(href)))
    return entries


def discover_directory(
    source: str,
    url: str,
    relative_root: str,
    *,
    suffixes: Optional[Tuple[str, ...]] = None,
    timeout: int = 60,
) -> List[RemoteFile]:
    """Recursively discover files under a public HTTP directory listing."""

    root_url = url.rstrip("/") + "/"
    root = urlparse(root_url)
    root_path = unquote(root.path)
    queue: List[Tuple[str, Path]] = [(root_url, Path())]
    visited = set()
    files: List[RemoteFile] = []

    while queue:
        current_url, current_relative = queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        response = request_with_retries("GET", current_url, timeout=timeout)
        try:
            entries = parse_directory_listing(response.text)
        finally:
            response.close()

        for entry in entries:
            href = entry.href
            if href.startswith("?") or href.startswith("#"):
                continue
            absolute_url = urljoin(current_url, href)
            parsed = urlparse(absolute_url)
            decoded_path = unquote(parsed.path)
            if (
                parsed.netloc != root.netloc
                or not decoded_path.startswith(root_path)
                or parsed.query
            ):
                continue

            name = unquote(decoded_path.rstrip("/").split("/")[-1])
            if not name:
                continue
            relative = current_relative / name
            if decoded_path.endswith("/"):
                queue.append((absolute_url.rstrip("/") + "/", relative))
                continue
            if suffixes and not name.endswith(suffixes):
                continue
            files.append(
                RemoteFile(
                    source=source,
                    url=absolute_url,
                    relative_path=(Path(relative_root) / relative).as_posix(),
                    estimated_size=entry.size,
                )
            )

    return files


def discover_linked_files(
    source: str,
    page_url: str,
    relative_root: str,
    filename_pattern: str,
    *,
    timeout: int = 60,
) -> List[RemoteFile]:
    """Discover data files linked from a normal web page."""

    response = request_with_retries("GET", page_url, timeout=timeout)
    try:
        parser = LinkParser()
        parser.feed(response.text)
    finally:
        response.close()

    files: List[RemoteFile] = []
    seen = set()
    for href in parser.links:
        absolute_url = urljoin(page_url, href)
        parsed = urlparse(absolute_url)
        filename = unquote(parsed.path.split("/")[-1])
        if (
            parsed.scheme not in {"http", "https"}
            or not re.fullmatch(filename_pattern, filename)
            or absolute_url in seen
        ):
            continue
        seen.add(absolute_url)
        files.append(
            RemoteFile(
                source=source,
                url=absolute_url,
                relative_path=(Path(relative_root) / filename).as_posix(),
            )
        )
    if not files:
        raise RuntimeError(f"no matching data files found at {page_url}")
    return files


def optional_static_files(args: argparse.Namespace) -> List[RemoteFile]:
    """Return explicitly requested large, fixed downloads."""

    files: List[RemoteFile] = []
    if args.include_clinical_trials:
        files.append(
            RemoteFile(
                "clinical-trials",
                "https://clinicaltrials.gov/AllPublicXML.zip",
                "clinical_trials/AllPublicXML.zip",
            )
        )
    if args.include_clinvar_xml:
        clinvar_xml = (
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/"
            "ClinVarVCVRelease_00-latest.xml.gz"
        )
        files.append(
            RemoteFile(
                "clinvar-xml",
                clinvar_xml,
                "clinvar/xml/ClinVarVCVRelease_00-latest.xml.gz",
                checksum_url=f"{clinvar_xml}.md5",
            )
        )
    if args.include_reactome:
        reactome_base = "https://reactome.org/download/current"
        for name in (
            "ReactomePathways.txt",
            "ReactomePathwaysRelation.txt",
            "UniProt2Reactome_All_Levels.txt",
        ):
            files.append(
                RemoteFile(
                    "reactome",
                    f"{reactome_base}/{name}",
                    f"reactome/{name}",
                )
            )
    return files


def build_manifest(args: argparse.Namespace) -> List[RemoteFile]:
    """Build the selected core and optional download manifest."""

    selected = set(args.only or CORE_GROUPS)
    selected.difference_update(args.skip or [])
    static_groups = core_static_files()
    files: List[RemoteFile] = []

    for group in CORE_GROUPS:
        if group in selected and group in static_groups:
            files.extend(static_groups[group])

    if "orphadata" in selected:
        files.extend(
            discover_linked_files(
                "orphadata",
                "https://sciences.orphadata.com/classifications/",
                "orphadata/classifications",
                r"en_product3_\d+\.xml",
                timeout=args.timeout,
            )
        )

    open_targets_url = (
        "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output/"
    )
    if args.include_open_targets_full:
        files = [item for item in files if item.source != "opentargets"]
        files.extend(
            discover_directory(
                "opentargets-full",
                open_targets_url,
                "opentargets",
                timeout=args.timeout,
            )
        )
    elif "opentargets" in selected:
        for dataset in OPEN_TARGETS_DATASETS:
            files.extend(
                discover_directory(
                    "opentargets",
                    f"{open_targets_url}{dataset}/",
                    f"opentargets/{dataset}",
                    timeout=args.timeout,
                )
            )

    files.extend(optional_static_files(args))

    if args.include_chembl:
        files.extend(
            discover_directory(
                "chembl",
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/",
                "chembl",
                suffixes=("_sqlite.tar.gz", "checksums.txt", "LICENSE"),
                timeout=args.timeout,
            )
        )
    if args.include_pubmed:
        for directory in ("baseline", "updatefiles"):
            files.extend(
                discover_directory(
                    "pubmed",
                    f"https://ftp.ncbi.nlm.nih.gov/pubmed/{directory}/",
                    f"pubmed/{directory}",
                    suffixes=(".xml.gz", ".xml.gz.md5"),
                    timeout=args.timeout,
                )
            )
    if args.include_europe_pmc:
        files.extend(
            discover_directory(
                "europe-pmc",
                "https://ftp.ebi.ac.uk/pub/databases/pmc/oa/",
                "europe_pmc/oa",
                suffixes=(".xml.gz",),
                timeout=args.timeout,
            )
        )

    deduplicated: Dict[str, RemoteFile] = {}
    for item in files:
        deduplicated[item.relative_path] = item
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.source, item.relative_path),
    )


def probe_size(item: RemoteFile, timeout: int) -> Optional[int]:
    """Get the best available size estimate without downloading a file."""

    if item.estimated_size is not None:
        return item.estimated_size
    if item.method != "GET":
        return None
    try:
        response = request_with_retries("HEAD", item.url, timeout=timeout)
        try:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
        finally:
            response.close()
    except (RuntimeError, ValueError):
        return None


def get_size_estimates(
    files: Sequence[RemoteFile],
    output_dir: Path,
    *,
    workers: int,
    timeout: int,
    force: bool,
) -> Dict[str, Optional[int]]:
    """Estimate pending bytes, probing only files without listing sizes."""

    estimates: Dict[str, Optional[int]] = {}
    to_probe: List[RemoteFile] = []
    for item in files:
        destination = output_dir / item.relative_path
        if destination.exists() and not force:
            estimates[item.relative_path] = 0
        elif item.estimated_size is not None:
            estimates[item.relative_path] = item.estimated_size
        else:
            to_probe.append(item)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(probe_size, item, timeout): item for item in to_probe
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            estimates[item.relative_path] = future.result()
    return estimates


def format_bytes(size: Optional[int]) -> str:
    """Format a byte count using binary units."""

    if size is None:
        return "unknown"
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024 or unit == "PiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} PiB"


def print_plan(
    files: Sequence[RemoteFile], estimates: Dict[str, Optional[int]]
) -> None:
    """Print a compact source-by-source download plan."""

    grouped: Dict[str, List[RemoteFile]] = {}
    for item in files:
        grouped.setdefault(item.source, []).append(item)

    print("Download plan:")
    total = 0
    unknown = 0
    for source, items in sorted(grouped.items()):
        sizes = [estimates.get(item.relative_path) for item in items]
        known = sum(size for size in sizes if size is not None)
        source_unknown = sum(size is None for size in sizes)
        total += known
        unknown += source_unknown
        qualifier = f" + {source_unknown} unknown" if source_unknown else ""
        print(
            f"  {source:20} {len(items):5} files  "
            f"{format_bytes(known):>12}{qualifier}"
        )
    suffix = f" plus {unknown} files of unknown size" if unknown else ""
    print(f"Total pending: {format_bytes(total)}{suffix}")


def nearest_existing_path(path: Path) -> Path:
    """Find an existing parent suitable for a free-space query."""

    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_free_space(
    output_dir: Path,
    estimates: Dict[str, Optional[int]],
    *,
    ignore_check: bool,
) -> None:
    """Ensure that known downloads fit with a modest temporary-file reserve."""

    if ignore_check:
        return
    required = sum(size for size in estimates.values() if size is not None)
    reserve = max(256 * 1024**2, int(required * 0.10))
    free = shutil.disk_usage(nearest_existing_path(output_dir)).free
    if free < required + reserve:
        raise RuntimeError(
            f"insufficient free space: need approximately "
            f"{format_bytes(required + reserve)}, have {format_bytes(free)}. "
            "Use another --output-dir or --ignore-space-check."
        )


def fetch_md5(url: str, timeout: int) -> str:
    """Fetch and parse an NCBI-style MD5 sidecar."""

    response = request_with_retries("GET", url, timeout=timeout)
    try:
        match = re.search(r"\b([0-9a-fA-F]{32})\b", response.text)
    finally:
        response.close()
    if not match:
        raise RuntimeError(f"could not parse MD5 checksum from {url}")
    return match.group(1).lower()


def calculate_md5(path: Path) -> str:
    """Calculate a file's MD5 for verification against publisher sidecars."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_attempt(
    item: RemoteFile,
    destination: Path,
    *,
    timeout: int,
) -> int:
    """Perform one streaming attempt, resuming GET requests when possible."""

    part_path = destination.with_name(f"{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    can_resume = item.method == "GET" and part_path.exists()
    starting_size = part_path.stat().st_size if can_resume else 0
    headers = {"Range": f"bytes={starting_size}-"} if starting_size else None

    response = request_with_retries(
        item.method,
        item.url,
        stream=True,
        form_data=item.form_data,
        headers=headers,
        timeout=timeout,
        accepted_statuses=(416,),
    )
    try:
        if response.status_code == 416:
            total_match = re.search(
                r"\*/(\d+)", response.headers.get("Content-Range", "")
            )
            if total_match and starting_size == int(total_match.group(1)):
                final_size = starting_size
            else:
                part_path.unlink(missing_ok=True)
                raise RuntimeError(f"server rejected resume for {item.url}")
        else:
            append = response.status_code == 206 and starting_size > 0
            if not append:
                starting_size = 0
            mode = "ab" if append else "wb"
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
            final_size = part_path.stat().st_size

            content_range = response.headers.get("Content-Range", "")
            total_match = re.search(r"/(\d+)$", content_range)
            if total_match:
                expected_size = int(total_match.group(1))
            else:
                content_length = response.headers.get("Content-Length")
                expected_size = (
                    starting_size + int(content_length)
                    if content_length
                    else None
                )
            if expected_size is not None and final_size != expected_size:
                raise RuntimeError(
                    f"incomplete download for {item.url}: "
                    f"expected {expected_size} bytes, got {final_size}"
                )
    finally:
        response.close()

    if item.checksum_url:
        expected_md5 = fetch_md5(item.checksum_url, timeout)
        actual_md5 = calculate_md5(part_path)
        if actual_md5 != expected_md5:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {item.url}: "
                f"expected {expected_md5}, got {actual_md5}"
            )

    os.replace(part_path, destination)
    return final_size


def download_one(
    item: RemoteFile,
    output_dir: Path,
    *,
    timeout: int,
    force: bool,
) -> DownloadResult:
    """Download one manifest entry and retain partial files on interruption."""

    destination = output_dir / item.relative_path
    if destination.exists() and not force:
        return DownloadResult(
            item.source,
            item.url,
            item.relative_path,
            "skipped",
            destination.stat().st_size,
        )

    part_path = destination.with_name(f"{destination.name}.part")
    if force:
        part_path.unlink(missing_ok=True)

    last_error: Optional[BaseException] = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            size = download_attempt(item, destination, timeout=timeout)
            return DownloadResult(
                item.source,
                item.url,
                item.relative_path,
                "downloaded",
                size,
            )
        except (OSError, RuntimeError, requests.RequestException) as error:
            last_error = error
            if attempt == MAX_REQUEST_ATTEMPTS - 1:
                break
            time.sleep((2**attempt) + random.uniform(0.0, 0.5))

    return DownloadResult(
        item.source,
        item.url,
        item.relative_path,
        "failed",
        part_path.stat().st_size if part_path.exists() else 0,
        error=str(last_error),
    )


def write_manifest(output_dir: Path, results: Sequence[DownloadResult]) -> None:
    """Write an atomic machine-readable record of this run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.part")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": [asdict(result) for result in results],
    }
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)


def run_downloads(
    files: Sequence[RemoteFile],
    output_dir: Path,
    *,
    workers: int,
    timeout: int,
    force: bool,
) -> List[DownloadResult]:
    """Download the manifest concurrently and report each completed file."""

    results: List[DownloadResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                download_one,
                item,
                output_dir,
                timeout=timeout,
                force=force,
            ): item
            for item in files
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as error:
                result = DownloadResult(
                    item.source,
                    item.url,
                    item.relative_path,
                    "failed",
                    0,
                    error=str(error),
                )
            results.append(result)
            if result.status == "failed":
                print(
                    f"[failed]     {result.relative_path}: {result.error}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[{result.status:10}] {result.relative_path} "
                    f"({format_bytes(result.size)})"
                )
    return sorted(results, key=lambda result: result.relative_path)


def positive_int(value: str) -> int:
    """Argparse validator for positive integer options."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the public datasets used by the rare-disease similarity project. "
            "The default core set is approximately 3 GB."
        ),
        epilog=(
            "NORD and GARD are not scraped because they do not provide unrestricted "
            "bulk downloads. OMIM is omitted because it requires registration or a "
            "commercial licence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=CORE_GROUPS,
        help="download only this core source; repeat to select multiple sources",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=CORE_GROUPS,
        help="omit this core source; repeat to omit multiple sources",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="number of concurrent file downloads (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=120,
        help="per-request read timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files that have already been downloaded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover files and estimate storage without downloading",
    )
    parser.add_argument(
        "--ignore-space-check",
        action="store_true",
        help="download even when the known files exceed available disk space",
    )

    large = parser.add_argument_group("explicit large/optional downloads")
    large.add_argument(
        "--include-chembl",
        action="store_true",
        help="include the latest ChEMBL SQLite archive (~5.8 GB compressed)",
    )
    large.add_argument(
        "--include-clinical-trials",
        action="store_true",
        help="include all ClinicalTrials.gov study XML (~7 GB compressed)",
    )
    large.add_argument(
        "--include-pubmed",
        action="store_true",
        help="include the PubMed baseline and updates (~73 GB compressed)",
    )
    large.add_argument(
        "--include-europe-pmc",
        action="store_true",
        help="include Europe PMC open-access XML (~180 GB compressed)",
    )
    large.add_argument(
        "--include-open-targets-full",
        action="store_true",
        help="replace the focused Open Targets subset with all outputs (~63 GB)",
    )
    large.add_argument(
        "--include-clinvar-xml",
        action="store_true",
        help="include the full ClinVar VCV XML release (~6 GB compressed)",
    )
    large.add_argument(
        "--include-reactome",
        action="store_true",
        help="include Reactome pathway hierarchy and UniProt mappings",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point."""

    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        files = build_manifest(args)
        if not files:
            print("No files selected.", file=sys.stderr)
            return 2

        estimates = get_size_estimates(
            files,
            output_dir,
            workers=args.workers,
            timeout=args.timeout,
            force=args.force,
        )
        print_plan(files, estimates)
        if args.dry_run:
            return 0
        check_free_space(
            output_dir,
            estimates,
            ignore_check=args.ignore_space_check,
        )
        results = run_downloads(
            files,
            output_dir,
            workers=args.workers,
            timeout=args.timeout,
            force=args.force,
        )
        write_manifest(output_dir, results)
        failures = [result for result in results if result.status == "failed"]
        if failures:
            print(
                f"{len(failures)} download(s) failed; partial .part files were kept "
                "for the next run.",
                file=sys.stderr,
            )
            return 1
        downloaded = sum(
            result.size for result in results if result.status == "downloaded"
        )
        print(f"Completed successfully; downloaded {format_bytes(downloaded)}.")
        return 0
    except (OSError, RuntimeError, requests.RequestException) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())