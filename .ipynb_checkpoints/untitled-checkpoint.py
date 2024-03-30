# Open and read the fake news CSV file
fake_articles = []
with open('fake_news.csv', 'r', encoding='utf-8') as file:
    for line in file:
        fake_articles.append(line.strip().split(','))

# Open and read the real news CSV file
real_articles = []
with open('real_news.csv', 'r', encoding='utf-8') as file:
    for line in file:
        real_articles.append(line.strip().split(','))
