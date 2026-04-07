import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper
from config.settings import settings

UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _download(url: str, dest: str) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()
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

    # Already downloaded
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest

    # Candidate URLs in priority order
    candidates: list[str] = []

    if paper.pdf_url:
        candidates.append(paper.pdf_url)

    if paper.doi:
        unpaywall_url = _get_unpaywall_pdf(paper.doi, settings.openalex_email)
        if unpaywall_url:
            candidates.append(unpaywall_url)

    for url in candidates:
        try:
            _download(url, dest)
            if os.path.getsize(dest) > 1024:
                print(f"  [PDF] Downloaded: {paper.id}")
                return dest
        except Exception:
            continue

    print(f"  [PDF] Not available: {paper.id}")
    return None
