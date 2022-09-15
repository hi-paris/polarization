# LIBS IMPORT
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time



# SELENIUM SETTINGS
PATH = "./chromedriver.exe"
# driver = webdriver.Chrome(PATH)
option = webdriver.ChromeOptions()
chrome_prefs = {}
option.experimental_options["prefs"] = chrome_prefs
chrome_prefs["profile.default_content_settings"] = {"images": 2}
chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}
# option.add_argument('headless')
driver = webdriver.Chrome(options=option)

# Get to the main page
driver.get('https://www.nzz.ch/')
# Get away with cookie windows
try:
    main = WebDriverWait(driver, 10).until(  # wait driver 10 sec jusqu'a detection de l'element
        EC.presence_of_element_located((By.XPATH, '//*[@id="cmpwelcomebtnyes"]/a'))
    )
    driver.find_element(by=By.XPATH,
                        value='//*[@id="cmpwelcomebtnyes"]/a').click()
except:
    pass

# Identify yourself, traveller
# Click sur Anmelden/Connecter after waiting for it to load
main = WebDriverWait(driver, 10).until(  # wait driver 10 sec jusqu'a detection de l'element
    EC.presence_of_element_located((By.XPATH, '//*[@id="header"]/div[2]/div/div[1]/div/div/div[2]/div/a/span'))
)
driver.find_element(by=By.XPATH,
                    value='//*[@id="header"]/div[2]/div/div[1]/div/div/div[2]/div/a/span').click()
time.sleep(3)
# Get the field for ID
InputID = driver.find_element(by=By.XPATH, value='//*[@id="c1-login-field"]')
time.sleep(3)
# Send ID to field
InputID.send_keys('millerk@hec.fr')
time.sleep(3)
# Validation button
driver.find_element(by=By.XPATH,
                    value='/html/body/main/div[2]/form/section[1]/div/div/button').click()
time.sleep(3)
# Get the field for PW
InputPW = driver.find_element(by=By.XPATH, value='//*[@id="c1-password-field"]')
time.sleep(3)
# Send PW to field
InputPW.send_keys('hec2022paris')
time.sleep(3)
# Validation button
driver.find_element(by=By.XPATH,
                    value='//*[@id="c1-submit-button-login"]').click()

time.sleep(3)
Headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}
driver_cookies = driver.get_cookies()

s = requests.Session()
# Set correct user agent
selenium_user_agent = driver.execute_script("return navigator.userAgent;")
s.headers.update({"user-agent": selenium_user_agent})

for cookie in driver.get_cookies():
    s.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

print('cookies', driver.get_cookies())

# Data
df = pd.read_csv(r'C:\Users\pblancha\PycharmProjects\scraping_newspaper\nzz_all_articles_gathered.csv', sep=',', encoding='utf-8')
df['cleaned_url'].replace(np.nan, 'NO', inplace=True)
list_links = list(df['cleaned_url'].drop_duplicates())
list_links = list_links[94196:100000]
path_to_results_features = r'C:\Users\pblancha\PycharmProjects\scraping_newspaper\article_features.csv'
path_to_results_meta = r'C:\Users\pblancha\PycharmProjects\scraping_newspaper\article_meta.csv'

df_features = []
df_meta = []
start_time = time.time()
articles = []

# Scrap function
def scrapping_nzz_ch(list_links, s):
    # list initalization
    news_contents = []
    list_titles = []
    list_category = []
    list_author = []
    list_subtitle = []
    list_links_200 = []
    list_dates = []
    list_id = []
    list_para_length = []
    last_save_time = time.time()

    try:
        for n in np.arange(0, len(list_links) - 1):
            # Reading the content

            if len(list_links[n]) > 4 and s.get(list_links[n]):
                article = s.get(list_links[n])
                status_code = article.status_code

                if status_code == 200:
                    article_content = article.content
                    soup_article = BeautifulSoup(article_content, 'html.parser')

                    # update our list of links with working ones i.e not leading to 404
                    id_split = list_links[n]
                    id_split = id_split.split('.')[-1]
                    list_links_200.append(list_links[n])
                    list_id.append(id_split.split('.')[-1])

                    # Getting the title
                    title = soup_article.find('h1', class_="headline__title")
                    if title is not None:
                        list_titles.append(title.get_text())
                    else:
                        list_titles.append('Not found')

                    # Authors
                    author = soup_article.find('span', class_="metainfo__item metainfo__item--author")
                    if author is None:
                        list_author.append('Not found')
                    else:
                        list_author.append(author.get_text())

                    # Subtitles
                    subtitle = soup_article.find('p', class_='headline__lead')
                    if subtitle is not None:
                        list_subtitle.append(subtitle.get_text())
                    else:
                        list_subtitle.append('None')

                    # Category is the first info in the URL
                    category = list_links[n].split('/')[3]
                    list_category.append(category)

                    # Publication date
                    date = soup_article.find('time')
                    if date is not None:
                        list_dates.append(date.get('datetime'))
                    else:
                        list_dates.append('No date found')

                    # Merging paragraphs
                    body = soup_article.find_all('div')
                    x = body[0].find_all('p')
                    list_paragraphs = []
                    list_para_length.append(str(len(x)))

                    final_article = ''
                    for p in np.arange(0, len(x)):
                        paragraph = x[p].get_text()
                        list_paragraphs.append(paragraph)
                        final_article = " ".join(list_paragraphs)

                    news_contents.append(final_article)

                    if (time.time() - last_save_time) > 1000:
                        print('saving {n} / {total}'.format(n=n, total=len(list_links) - 1))
                        # df_features contains link + content
                        df_features = pd.DataFrame(
                            {'Article Content': news_contents,
                             'Article Link': list_links_200
                             })

                        # df_show_info contains all the meta information
                        df_meta = pd.DataFrame(
                            {'Article ID': list_id,
                             'Article Title': list_titles,
                             'Article Link': list_links_200,
                             'Authors': list_author,
                             'Category': list_category,
                             'Content Length characters': df_features['Article Content'].str.len(),
                             'Content Length words': df_features['Article Content'].str.split().str.len(),
                             'Content paragraph length': list_para_length,
                             'Date': list_dates})

                        with open(path_to_results_features, 'a', newline='', encoding='utf8') as f:
                            pd.DataFrame(df_features).to_csv(f, header=False, encoding='utf8', sep=';')
                        df_features = []

                        with open(path_to_results_meta, 'a', newline='', encoding='utf8') as f:
                            pd.DataFrame(df_meta).to_csv(f, header=False, encoding='utf8', sep=';')
                        df_meta = []

                        last_save_time = time.time()

                else:
                    continue

    except Exception as e:
        print(e)


def main():
    start = time.time()
    scrapping_nzz_ch(list_links, s)
    if not pd.DataFrame(df_features).empty:
        with open(path_to_results_features, 'a', newline='', encoding='utf8') as f:
            pd.DataFrame(df_features).to_csv(f, header=False, encoding='utf8')
    if not pd.DataFrame(df_meta).empty:
        with open(path_to_results_meta, 'a', newline='', encoding='utf8') as f:
            pd.DataFrame(df_meta).to_csv(f, header=False, encoding='utf8')
    end = time.time()
    time_elapsed = end - start
    print("Computing time is {}".format(time_elapsed))


if __name__ == "__main__":
    main()
