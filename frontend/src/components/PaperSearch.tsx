import { useId, useMemo, useState } from "react";
import type { PaperEntry } from "../data/types";
import { plural } from "../lib/format";
import { InfoTip } from "./InfoTip";

interface Props {
  /** Uploaded, curated and acquisition-release papers; null while the literature file loads. */
  papers: PaperEntry[] | null;
  error: string | null;
  onSelect: (id: string) => void;
}

interface Indexed {
  paper: PaperEntry;
  title: string;
  /** PMID or other identifier, lower case, without the "pmid:" prefix. */
  ref: string;
}

const MAX_RESULTS = 10;

const SET_LABEL: Record<PaperEntry["kind"], string> = {
  upload: "Uploaded",
  literature: "Curated",
  acquired: "Newly acquired",
};

const ACCESS_LABEL: Record<string, string> = {
  abstract_only: "abstract only",
  metadata_only: "metadata only",
  open_full_text_xml: "open full text",
  retracted_or_removed: "retracted",
};

const fold = (text: string) =>
  text.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "");

function meta(p: PaperEntry): string {
  const parts = [p.year ? String(p.year) : null, SET_LABEL[p.kind]];
  if (p.access)
    parts.push(ACCESS_LABEL[p.access] ?? p.access.replace(/_/g, " "));
  if (p.claims.length > 0)
    parts.push(
      plural(
        p.claims.length,
        p.kind === "acquired" ? "candidate pair" : "claim",
      ),
    );
  return parts.filter(Boolean).join(" · ");
}

export function PaperSearch({ papers, error, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();

  const index = useMemo<Indexed[]>(
    () =>
      (papers ?? []).map((p) => ({
        paper: p,
        title: fold(p.title),
        ref: p.id.toLowerCase().replace(/^pmid:/, ""),
      })),
    [papers],
  );

  /** "508 curated + 1,972 newly acquired papers", so the total can be told apart from the curated set's. */
  const breakdown = useMemo(() => {
    if (!papers) return null;
    const count = (kind: PaperEntry["kind"]) =>
      papers.filter((p) => p.kind === kind).length;
    const parts = [
      count("literature") && `${count("literature").toLocaleString()} curated`,
      count("acquired") &&
        `${count("acquired").toLocaleString()} newly acquired`,
      count("upload") && `${count("upload").toLocaleString()} uploaded`,
    ].filter(Boolean);
    return parts.length > 0
      ? `${parts.join(" + ")} ${papers.length === 1 ? "paper" : "papers"}`
      : null;
  }, [papers]);

  const results = useMemo(() => {
    const q = fold(query.trim());
    if (!q) return [];
    const id = q.replace(/^pmid:?\s*/, "");
    const words = q.split(/\s+/).filter(Boolean);
    const wordStart = new RegExp(
      `\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`,
    );
    const scored: { p: PaperEntry; rank: number }[] = [];
    for (const item of index) {
      let rank = -1;
      if (item.ref === id) rank = 0;
      else if (item.title.startsWith(q)) rank = 1;
      else if (wordStart.test(item.title)) rank = 2;
      else if (item.title.includes(q)) rank = 3;
      else if (words.length > 1 && words.every((w) => item.title.includes(w)))
        rank = 4;
      if (rank >= 0) scored.push({ p: item.paper, rank });
    }
    // Papers that can be checked against the graph first, then the newest.
    scored.sort(
      (x, y) =>
        x.rank - y.rank ||
        y.p.claims.length - x.p.claims.length ||
        (y.p.year ?? 0) - (x.p.year ?? 0),
    );
    return scored.slice(0, MAX_RESULTS);
  }, [query, index]);

  const choose = (id: string) => {
    onSelect(id);
    setQuery("");
    setOpen(false);
  };

  return (
    <div className="search">
      <div className="label-row">
        <label htmlFor={`${listId}-input`} className="section-label">
          Find a paper
        </label>
        <InfoTip topic="paperSearch" />
      </div>
      <div className="search-field">
        <input
          id={`${listId}-input`}
          type="search"
          role="combobox"
          aria-expanded={open && results.length > 0}
          aria-controls={listId}
          aria-activedescendant={
            results[active] ? `${listId}-${active}` : undefined
          }
          placeholder="Title or PMID"
          value={query}
          autoComplete="off"
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => Math.min(a + 1, results.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(a - 1, 0));
            } else if (e.key === "Enter" && results[active]) {
              choose(results[active].p.id);
            } else if (e.key === "Escape") {
              setOpen(false);
            }
          }}
        />
        {open && query.trim() && (
          <ul className="search-results" role="listbox" id={listId}>
            {error && <li className="search-empty">{error}</li>}
            {!error && !papers && (
              <li className="search-empty">Loading papers…</li>
            )}
            {papers && results.length === 0 && (
              <li className="search-empty">No matching papers.</li>
            )}
            {results.map(({ p }, i) => (
              <li
                key={p.id}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={i === active}
                className={i === active ? "active" : undefined}
                onMouseDown={(e) => {
                  e.preventDefault();
                  choose(p.id);
                }}
                onMouseEnter={() => setActive(i)}
              >
                <span className="search-name">{p.title}</span>
                <span className="search-meta">{meta(p)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {breakdown && <p className="muted small search-hint">{breakdown}</p>}
    </div>
  );
}
