import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper
from config.settings import settings

UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

# Realistic browser headers — many publishers (MDPI, Springer, Wiley) reject
# generic Python User-Agents via Cloudflare bot protection.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get_unpaywall_pdf(doi: str, email: str) -> str | None:
    try:
        resp = requests.get(
            UNPAYWALL_URL.format(doi=doi),
            params={"email": email},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("is_oa"):
            loc = data.get("best_oa_location") or {}
            return loc.get("url_for_pdf")
    except Exception:
        return None


def _is_valid_pdf(path: str) -> bool:
    """Check if file starts with the PDF magic bytes (%PDF-)."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _is_pubmed_abstract_url(url: str) -> bool:
    """PubMed URLs go to the abstract page, not a PDF — skip them."""
    return "pubmed.ncbi.nlm.nih.gov" in url


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _download(url: str, dest: str) -> bool:
    resp = requests.get(
        url,
        headers=BROWSER_HEADERS,
        timeout=60,
        stream=True,
        allow_redirects=True,
    )
    resp.raise_for_status()

    # Reject if response is HTML (paywall page, login redirect, etc.)
    content_type = resp.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        return False

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def download_pdf(paper: Paper) -> str | None:
    """
    Try to download the PDF for a paper.
    Returns local file path if successful, None otherwise.
    """
    os.makedirs(settings.pdf_dir, exist_ok=True)
    dest = os.path.join(settings.pdf_dir, f"{paper.id}.pdf")

    # Already downloaded and valid
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 and _is_valid_pdf(dest):
        return dest

    # Build candidate URLs — prioritize Unpaywall for PubMed (abstract URL is useless)
    candidates: list[str] = []

    is_pubmed = paper.source == "pubmed" or (paper.pdf_url and _is_pubmed_abstract_url(paper.pdf_url))

    # Always try Unpaywall first when we have a DOI — it points to actual OA PDFs
    if paper.doi:
        unpaywall_url = _get_unpaywall_pdf(paper.doi, settings.openalex_email)
        if unpaywall_url:
            candidates.append(unpaywall_url)

    # Then the publisher's URL — but skip the PubMed abstract page
    if paper.pdf_url and not _is_pubmed_abstract_url(paper.pdf_url):
        candidates.append(paper.pdf_url)

    # MDPI fallback: try the canonical /pdf URL even if we already have one
    if paper.source == "mdpi" and paper.doi:
        mdpi_alt = f"https://www.mdpi.com/{paper.doi}/pdf"
        if mdpi_alt not in candidates:
            candidates.append(mdpi_alt)

    for url in candidates:
        try:
            ok = _download(url, dest)
            if ok and os.path.getsize(dest) > 1024 and _is_valid_pdf(dest):
                print(f"  [PDF] Downloaded: {paper.id}")
                return dest
            if os.path.exists(dest):
                os.remove(dest)
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            continue

    print(f"  [PDF] Not available: {paper.id}")
    return None
