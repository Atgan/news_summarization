from pathlib import Path
from ..items import NewsItem 
import scrapy
import re 
from datetime import datetime 



class QuotesSpider(scrapy.Spider):
    name = "cnn_crawler"
    
    

    def start_requests(self):
        urls = [
            "https://edition.cnn.com",
            
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        links = []
        for href in response.css('div[data-uri] a::attr(href)').getall():
            href = "https://edition.cnn.com" + href
            links.append(response.urljoin(href))

        yield from self.parse_link(response, links)

    def parse_link(self, response, links):
        
        for link in links:
            yield scrapy.Request(url=link, callback=self.parse_individual_link)

    def parse_individual_link(self, response):
        article_body = response.css('script[type="application/ld+json"]').get()

        if article_body:

            article_match = re.search(r'"articleBody":"([^"]+)"', article_body)
            article_content = article_match.group(1) if article_match else ""

            date_written_match = re.search(r'"datePublished":"(.+?)"', article_body)
            date_written = datetime.strptime(date_written_match.group(1), '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S') if date_written_match else ""


            authors_match = re.findall(r'"@type":"Person","name":"([^"]+)"', article_body)
            authors = authors_match if authors_match else ["unknown author"]


            summary = ""
            summary_date = ""


            item = NewsItem(
            content=article_content,
            date_written = date_written,
            authors = authors,
            summary = summary,
            summary_date = summary_date
            )
            yield item
        else:
            self.logger.warning(f"No JSON data found on {response.url}")


#scrapy crawl -s MONGODB_URI="mongodb://localhost:27017/news_scraping" -s MONGODB_DATABASE="news_scraping" cnn_crawler






