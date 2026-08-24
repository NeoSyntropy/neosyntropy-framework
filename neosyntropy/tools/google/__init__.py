__all__ = [
    "AuthConfig",
    "GoogleAuth",
    "GoogleSlidesTools",
    "GoogleBigQueryTools",
    "GoogleCalendarTools",
    "GoogleDriveTools",
    "GmailTools",
    "GoogleMapTools",
    "GoogleSheetsTools",
]


def __getattr__(name: str):
    if name == "AuthConfig":
        from neosyntropy.tools.google.auth import AuthConfig

        return AuthConfig
    if name == "GoogleAuth":
        from neosyntropy.tools.google.auth import GoogleAuth

        return GoogleAuth
    if name == "GoogleSlidesTools":
        from neosyntropy.tools.google.slides import GoogleSlidesTools

        return GoogleSlidesTools
    if name == "GoogleBigQueryTools":
        from neosyntropy.tools.google.bigquery import GoogleBigQueryTools

        return GoogleBigQueryTools
    if name == "GoogleCalendarTools":
        from neosyntropy.tools.google.calendar import GoogleCalendarTools

        return GoogleCalendarTools
    if name == "GoogleDriveTools":
        from neosyntropy.tools.google.drive import GoogleDriveTools

        return GoogleDriveTools
    if name == "GmailTools":
        from neosyntropy.tools.google.gmail import GmailTools

        return GmailTools
    if name == "GoogleMapTools":
        from neosyntropy.tools.google.maps import GoogleMapTools

        return GoogleMapTools
    if name == "GoogleSheetsTools":
        from neosyntropy.tools.google.sheets import GoogleSheetsTools

        return GoogleSheetsTools
    raise AttributeError(f"module 'neosyntropy.tools.google' has no attribute {name!r}")
