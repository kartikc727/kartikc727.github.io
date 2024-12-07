---
title: "Personalized News Summarizer"
excerpt: "A personalized news page that summarizes real-time news using the BART model from HuggingFace, using a MongoDB backend and hosted on Google Cloud Platform"
collection: portfolio
date: 2022-12-20 20:30:00 -0500
last_modified_at: 2023-07-24 08:30:00 -0500
---
[![Github forks][gh-fork-shield]][github-repo]

Staying up to date with the latest news is necessary in today's world, but it is 
also becoming increasingly difficult to do so. There are so many news sources, 
with difficult-to-navigate websites, and clickbait titles and descriptions that
make it har to determine which articles are worth reading.

The Personalized News Summarizer tries to solve this problem by creating a 
simple web page with a clean UI that displays the latest news articles from a 
variety of sources and provides a summary of each article. The app is 
hosted on the Google Cloud Platform and can be accessed [here][1]{:target="_blank"}
(if my GCP credits have not expired yet).

# Getting the News Articles

The first step is to get the news articles. There are many news APIs available, 
often with a free tier that allows us to retrieve a small number of articles per
day, which is sufficient for our use case. I tried several APIs, including

- [Bing News API][2]
- [News Org API][3]
- [NewsData API][4]

Finally, I settled on the NewsData API, which provides real-time news articles 
from a variety of sources and allows us to filter the articles by country, 
category, and language. We use the API to query for the latest articles based on
the categories and number of articles specified by the user. The following code
snippet shows how we query the API for the latest articles.

```python
class NewsDataAPI(NewsAPI):
    
    ...

    def news(self, news_query:NewsQuery, **kwargs)->list[Article]:
        category = self.CATEGORIES[news_query.category]['name']

        headers = {'X-ACCESS-KEY' : self.api_key}
        params = {'category':category, 'language':self.language, 'country':self.country}
        r = requests.get(self.news_url, headers=headers, params=params)
        r.raise_for_status()
        return [self._parse_article(a) for a in r.json()['results']]
```

These articles are then stored in a [MongoDB][5] database, which is hosted on 
[MongoDB Atlas][6]. We use the PyMongo library to connect to the database and 
store the articles in a collection. The article fetching function is hosted on 
[GCP][7] as a [Cloud Function][8], which is triggered by a [Cloud Scheduler][9]
job every day in the morning that pushes a message to a [Pub/Sub topic][10] 
to fetch the latest articles.

The cloud function gets information about the user's preferences from the
cloud event that triggered it, while the MongoDB credentials are provided as
environment variables by the function's secret manager, as shown below.

```python
@functions_framework.cloud_event
def fetch_news(cloud_event):
    request_b64 = cloud_event.data['message']['data']
    request_str = base64.b64decode(request_b64).decode('utf-8')
    request_json = json.loads(request_str)

    # generate data
    data = generate_data({
        'num_articles' : request_json.get('num_articles', 2),
        'categories' : request_json.get('categories', 'General').split(',')
    })
    
    # write data to MongoDB
    with get_client('db_info.json', 'conn_info.json') as client:
        db = client['news']
        collection = db['articles']

        for cateogry, articles in data.items():
            documents = []
            for article in articles:
                documents.append({
                    'category' : cateogry,
                    'api_src' : article.api_src,
                    'url' : article.url,
                    'title' : article.title,
                    'source' : article.source,
                    'time' : article.time,
                    'description' : article.description,
                    'img_url' : article.img_url,
                    'article_text' : article.text
                })
            collection.insert_many(documents)
```

# Summarizing the Articles

The next step is to summarize the articles. For this, we use the [BART model][11]
from [HuggingFace][12], which is fine-tuned on the [CNN/DailyMail dataset][13] 
to be able to summarize news articles. While the model is free to download and 
host, it becomes costly to host it on GCP, so we use the [HuggingFace inference API][14]
to summarize the articles.

Again, we create a Cloud Function that is triggered by a Cloud Scheduler job every 
minute for an hour in the morning after the articles have been fetched. Each 
function call retrieves an unsummarized article from the database, summarizes it
using the BART model, and updates the article in the database with the summary.

```python
@functions_framework.cloud_event
def summarize(cloud_event):

    summarizer = NewsSummarizer(**summarizer_api_params)

    with get_client('db_info.json', 'conn_info.json') as client:
        collection = client['news']['articles']

        article = collection.find_one({'summary' : {'$exists':False}})

        if article is None:
            print('No articles to summarize')
            return
        
        article_text = article['article_text']
        summary = summarizer(article_text)

        update_result = collection.update_one(
            {'_id':article['_id']}, 
            {"$set": {'summary':summary}},
            upsert=False)
        
        assert update_result.matched_count == 1, 'No document found'
        assert update_result.modified_count == 1, 'No document modified'

    return f'Summarized article with id {article["_id"]}'
```

# Creating the Web Page

Finally, we create a simple static web page that displays the summarized articles. 
We first create a [Flask][15] app that connects to the MongoDB database and 
retrieves the latest articles to display on the web page. We then use the 
[Frozen Flask][16] library to convert the Flask app into a static web page, 
which is then hosted on GCP as a [Cloud Storage bucket][17].

```python
@app.route('/')
def homepage():
    with open('site_args.json', 'r') as f:
        args = json.load(f)

    date = datetime.now().strftime('%B %d, %Y')

    with get_client('db_info.json', 'conn_info.json') as client:
        collection = client['news']['articles']
        docs = collection.find({'time' : {'$gt' : datetime.now() - timedelta(days=1)}})
        article_holders = {}
        for doc in docs:
            category = doc['category']
            article_holder = article_holders.get(category, ArticleHolder(category, []))
            article_holder.articles.append(Article(
                title=doc['title'],
                source=doc['source'],
                time_str=doc['time'].strftime('%l:%M %p on %b %d, %Y'),
                url=doc['url'],
                summary=doc.get('summary', 'Summary not available'),
                img_url=doc['img_url']
            ))
            article_holders[category] = article_holder

    article_holders = list(article_holders.values())

    bucket_name = args.get('bucket_name')
    return render_template('index.html',
        curr_date = date,
        article_holders = article_holders,
        stylesheet_url = get_url(bucket_name, 'static/main.css'),
        favicon_url = get_url(bucket_name, 'static/favicon.ico'))
```

If the user wants to read the full article, they can click on the article title,
which will redirect them to the original article on the news website. 

<!-- Links -->
[gh-fork-shield]: <https://img.shields.io/github/forks/kartikc727/ml-projects.svg?style=social&label=Fork&maxAge=2592000>
[github-repo]: <https://github.com/kartikc727/ml-projects/tree/master/personalized-news-summarizer> "Github repository"
[1]: <https://storage.googleapis.com/news-site-holder-bucket/build/index.html> "Personalized News Summarizer"
[2]: <https://www.microsoft.com/en-us/bing/apis/bing-news-search-api> "Bing News API"
[3]: <https://newsapi.org/> "News Org API"
[4]: <https://newsdata.io/> "NewsData API"
[5]: <https://www.mongodb.com/> "MongoDB"
[6]: <https://www.mongodb.com/atlas> "MongoDB Atlas"
[7]: <https://cloud.google.com/> "Google Cloud Platform"
[8]: <https://cloud.google.com/functions> "Cloud Functions - GCP"
[9]: <https://cloud.google.com/scheduler> "Cloud Scheduler - GCP"
[10]: <https://cloud.google.com/pubsub> "Pub/Sub - GCP"
[11]: <https://huggingface.co/facebook/bart-large-cnn> "BART model"
[12]: <https://huggingface.co/> "HuggingFace"
[13]: <https://huggingface.co/datasets/cnn_dailymail> "CNN/DailyMail dataset"
[14]: <https://huggingface.co/inference-api> "HuggingFace inference API"
[15]: <https://flask.palletsprojects.com/en/2.0.x/> "Flask"
[16]: <https://flask.palletsprojects.com/en/2.3.x/> "Frozen Flask"
[17]: <https://cloud.google.com/storage> "Cloud Storage - GCP"
