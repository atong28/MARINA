# Benchmark curation (upstream, human-in-the-loop)

These scripts produce the **inputs** behind the frozen Journal benchmark; they are
**not** part of the reproducible MARINA-DB build and are not called by `run_build.sh`. They are
here so the whole benchmark-creation process lives in one place, but they are non-deterministic
(network fetches, paper availability) and involve **manual** NMR-table extraction.

They operate on the `Benchmark/` working directory (`$MARINA_DATA_ROOT/Benchmark`, paths still
hardcoded inside each script) — harvesting candidates, fetching papers, and staging per-compound
folders. A human then transcribes shifts into `Benchmark/filtered/<NPID>/{1H,13C,HSQC}.csv`, from
which the frozen `benchmark-journal.pkl` was built. The go-forward build consumes that frozen pkl
(`build/7_build_journal.py` only re-derives the prepared view), not `filtered/` directly.

## Rough flow

```
extract_npmrd_index.py     flatten the NP-MRD dump into a compact index
scrape_npmrd.py            scrape NP-Cards for DOIs / spectrum types
build_candidates.py        group un-extracted compounds by source paper -> work queue
enrich_candidates.py       add title/journal/year/OA status
find_papers.py             search/download open-access PDFs
create_paper_links.py      per-NPID link files via PubMed/CrossRef/Unpaywall
assess_evidence.py         grade papers for likely NMR content
fetch_and_check.py         fetch OA PDFs, check for NMR tables
retry_fetch.py             second-pass fetch (PMC/MDPI referers)
fetch_missing_structures.py pull SMILES for NPIDs absent from the local dump
setup_batch.py             stage next queued papers -> filtered/<NPID>/ dirs
copy_info.py               copy paper metadata into filtered/<NPID>/info.txt
update_queue.py            reconcile the manual download queue vs disk
write_links.py write_table.py   render the queue as md/csv
```

Then: human extraction into `filtered/<NPID>/*.csv` → the frozen `benchmark-journal.pkl`.
