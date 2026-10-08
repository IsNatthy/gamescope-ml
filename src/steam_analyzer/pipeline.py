class Pipeline:
    def __init__(self, loader, cleaner, profiler):
        self.loader = loader
        self.cleaner = cleaner
        self.profiler = profiler

    def run(self):
        games = self.loader.load_games()
        reviews = self.loader.load_reviews()

        games_clean = self.cleaner.clean_games(games)
        reviews_clean = self.cleaner.clean_reviews(reviews)

        self.profiler.generate_report(games_clean, "games")
        self.profiler.generate_report(reviews_clean, "reviews")

        return games_clean, reviews_clean