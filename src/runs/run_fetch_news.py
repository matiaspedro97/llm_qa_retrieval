from src.data.parse import NewsFetcher


fetcher = NewsFetcher(
    websites=[
        "https://www.jn.pt/ultimas/",
        "https://www.publico.pt/",
        "https://expresso.pt/",
        "https://observador.pt/", 
        "https://www.dn.pt/",
        "https://sicnoticias.pt/",
        #"https://www.rtp.pt/noticias/",
        "https://www.sapo.pt/",
        "https://www.abola.pt/",
        "https://www.bbc.com/news/world",
    ]
)


fetcher.fetch_all(max_news=80)





