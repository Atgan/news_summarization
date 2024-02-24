import scrapy


class NewsItem(scrapy.Item):
    content = scrapy.Field()
    date_written = scrapy.Field()
    authors = scrapy.Field()
    summary = scrapy.Field()
    summary_date = scrapy.Field()

    collection = 'news_document'