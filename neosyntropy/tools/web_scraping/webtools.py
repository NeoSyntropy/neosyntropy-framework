import httpx

from neosyntropy.tools.core.toolkit import Toolkit
import logging
logger = logging.getLogger(__name__)
log_debug = logger.debug
log_error = logger.error
log_warning = logger.warning
log_info = logger.info


class WebTools(Toolkit):
    """
    A toolkit for working with web-related tools.
    """

    def __init__(
        self,
        retries: int = 3,
        enable_expand_url: bool = True,
        all: bool = False,
        **kwargs,
    ):
        self.retries = retries

        tools = []
        if all or enable_expand_url:
            tools.append(self.expand_url)

        super().__init__(name="web_tools", tools=tools, **kwargs)

    def expand_url(self, url: str) -> str:
        """
        Expands a shortened URL to its final destination using HTTP HEAD requests with retries.

        :param url: The URL to expand.

        :return: The final destination URL if successful; otherwise, returns the original URL.
        """
        timeout = 5
        for attempt in range(1, self.retries + 1):
            try:
                response = httpx.head(url, follow_redirects=True, timeout=timeout)
                final_url = response.url
                logger.info(f"expand_url: {url} expanded to {final_url} on attempt {attempt}")
                return str(final_url)
            except Exception:
                logger.exception(f"Error expanding URL {url} on attempt {attempt}")

        return url
