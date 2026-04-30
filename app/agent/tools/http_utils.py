from urllib import parse


SENSITIVE_QUERY_KEYS = {
    "apikey",
    "api_key",
    "key",
    "token",
    "access_token",
    "secret",
}


def redact_url(url: str | None) -> str | None:
    if not url:
        return url

    parsed = parse.urlsplit(url)
    query_items = parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = [
        (key, "***REDACTED***" if key.lower() in SENSITIVE_QUERY_KEYS else value)
        for key, value in query_items
    ]
    return parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parse.urlencode(redacted_query, doseq=True, safe="*"),
            parsed.fragment,
        )
    )
