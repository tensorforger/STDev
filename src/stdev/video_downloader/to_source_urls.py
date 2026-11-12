import os
import re
import json
import time
import random
import hashlib
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

COOLDOWN_MIN = 2.0
COOLDOWN_MAX = 5.0
VIDEO_URL_PATTERN = re.compile(r"https://vkvideo\.ru/video-\S+")


class ToSourceUrlsPipeline:
    def __init__(self, cfg: DictConfig):
        dataset_name = cfg.dataset_name
        self.dataset_root = f"./datasets/{dataset_name}"
        self.output_root = os.path.join(self.dataset_root, "source_urls")
        os.makedirs(self.output_root, exist_ok=True)
        self.filename_prefix = cfg.general.filename_prefix
        self.channel_urls = list(cfg.to_source_urls.channel_urls)
        self.max_links_per_channel = cfg.to_source_urls.max_links_per_channel

    def _build_driver(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        driver = webdriver.Chrome(options=options)
        return driver

    def _load_cookies(self, driver, url):
        cookies_path = "./cookies.txt"
        if not os.path.exists(cookies_path):
            return
        # Navigate first so the domain is set before adding cookies
        driver.get(url)
        time.sleep(1)
        with open(cookies_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                domain, _, path, secure, _, name, value = parts[:7]
                try:
                    driver.add_cookie(
                        {
                            "domain": domain.lstrip("."),
                            "path": path,
                            "secure": secure.upper() == "TRUE",
                            "name": name,
                            "value": value,
                        }
                    )
                except Exception:
                    pass

    def _scroll_and_collect(self, driver, channel_url):
        driver.get(channel_url)
        time.sleep(random.uniform(COOLDOWN_MIN, COOLDOWN_MAX))

        collected = set()
        last_height = 0

        while len(collected) < self.max_links_per_channel:
            page_source = driver.page_source
            found = VIDEO_URL_PATTERN.findall(page_source)
            # Strip trailing punctuation/quotes that may be captured
            found = {re.split(r'["\'\s<>]', u)[0] for u in found}
            collected.update(found)

            if len(collected) >= self.max_links_per_channel:
                break

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(COOLDOWN_MIN, COOLDOWN_MAX))

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # No more content loaded
                break
            last_height = new_height

        return list(collected)[: self.max_links_per_channel]

    def _url_to_filename(self, url):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(url.encode()).digest())
            .decode()
            .rstrip("=")
        )
        return f"{self.filename_prefix}_{digest[:43]}.json"

    def process_channel(self, driver, channel_url):
        print(f"Scraping channel: {channel_url}")
        urls = self._scroll_and_collect(driver, channel_url)
        print(f"  Found {len(urls)} video URLs")

        new_count = 0
        for url in urls:
            filename = self._url_to_filename(url)
            output_path = os.path.join(self.output_root, filename)
            if os.path.exists(output_path):
                continue
            data = {"url": url, "status": "PENDING"}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            new_count += 1

        print(f"  Saved {new_count} new URL files")

    def run(self):
        channels = self.channel_urls
        driver = self._build_driver()
        try:
            # Load cookies against the base domain before scraping
            if channels:
                self._load_cookies(driver, "https://vkvideo.ru")
            for channel_url in tqdm(channels, desc="Processing channels"):
                try:
                    self.process_channel(driver, channel_url)
                except Exception as e:
                    print(f"Error processing channel {channel_url}: {e}")
                time.sleep(random.uniform(COOLDOWN_MIN, COOLDOWN_MAX))
        finally:
            driver.quit()


def main():
    cfg = OmegaConf.load(sys.argv[1])
    pipeline = ToSourceUrlsPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
