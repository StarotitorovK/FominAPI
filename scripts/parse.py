import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urljoin, urlparse


def get_all_links(base_url, blog_prefix):
    visited = set()
    to_visit = [base_url]
    all_links = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue

        print(f"Посещение: {url}")
        visited.add(url)

        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    full_url = urljoin(url, href)
                    if full_url.startswith(blog_prefix):
                        if full_url not in visited:
                            to_visit.append(full_url)
                        if full_url not in all_links:
                            all_links.append(full_url)
            else:
                print(f'Не удалось получить страницу. Код состояния: {response.status_code}')
        except Exception as e:
            print(f'Ошибка при обработке {url}: {e}')

        time.sleep(1)

    return all_links


def save_links_to_csv(links, filename='links.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['URL'])
        for link in links:
            writer.writerow([link])
    print(f"Сохранено {len(links)} ссылок в {filename}")



base_url = 'https://fomin-clinic.ru/blog/'
blog_prefix = 'https://fomin-clinic.ru/blog/'
all_links = get_all_links(base_url, blog_prefix)
save_links_to_csv(all_links)