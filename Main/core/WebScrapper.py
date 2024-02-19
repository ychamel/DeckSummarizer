import json
import tldextract
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup


class ContentSpider(CrawlSpider):
    name = "content_spider"
    # allowed_domains = ["synapse-analytics.io"]
    # start_urls = ["https://www.synapse-analytics.io/"]

    rules = (
        Rule(LinkExtractor(allow=(), unique=True), callback='parse_item', follow=True),
    )
    DICT = {}
    def __init__(self, url='', allowed_domains=[], DICT={}, *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [url]
        self.allowed_domains = allowed_domains
        self.DICT = DICT

    def parse_item(self, response):
        url_name = response.url
        html = response.body
        soup = BeautifulSoup(html, "html.parser")
        self.DICT[url_name] = soup.get_text().strip()


def ScrapWeb(url: str):
    process = CrawlerProcess(settings={
        "FEEDS": {
            "items.json": {"format": "json"},
        },
    })
    extracted_info = tldextract.extract(url)
    domain_url = f"{extracted_info.domain}.{extracted_info.suffix}"
    process.crawl(ContentSpider, url=url, allowed_domains=[domain_url], DICT={})
    process.start()
    return ContentSpider.DICT
