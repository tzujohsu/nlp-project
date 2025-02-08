# nlp-project

```bash
docker build -t news-categorize-image .
docker run -d --name news-categorize-container -p 80:80 news-categorize-image

docker stop news-categorize-container 
docker rm news-categorize-container


```