import json
import tldextract
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue

from twisted.internet import reactor

class ContentSpider(CrawlSpider):
    name = "content_spider"
    rules = (
        Rule(LinkExtractor(allow=(), unique=True), callback='parse_item', follow=True),
    )
    DICT = {}
    def __init__(self, url='', allowed_domains=[], *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [url]
        self.allowed_domains = allowed_domains

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
    process.join()
    return ContentSpider.DICT

def run_spider(url):
    spider = ContentSpider
    extracted_info = tldextract.extract(url)
    domain_url = f"{extracted_info.domain}.{extracted_info.suffix}"

    def f(q):
        try:
            runner = CrawlerProcess(settings={
                "FEEDS": {
                    "items.json": {"format": "json"},
                },
            })
            deferred = runner.crawl(spider, url=url, allowed_domains=[domain_url])
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(spider.DICT)
        except Exception as e:
            q.put(None)

    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()

    return result