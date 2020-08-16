# import requests
# import random
# import shutil
# import bs4
# import ssl
# import os
# ssl._create_default_https_context = ssl._create_unverified_context
# def image(data):
#     Res = requests.get("https://www.google.com/search?hl=jp&q=" + data + "&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
#     print(Res)
#     Html = Res.text
#     Soup = bs4.BeautifulSoup(Html,'lxml')
#     links = Soup.find_all("img")
#     link = random.choice(links).get("src")
#     return link
# def download_img(url, file_name):
#     base_dir="train_data/web"
#     os.makedirs(base_dir,exist_ok=True)
#     try:
#         r = requests.get(url, stream=True)
#         print(url)
#         if r.status_code == 200:
#             with open(os.path.join(base_dir,file_name+".png"), 'wb') as f:
#                 print(os.path.join(base_dir,file_name+".png"))
#                 r.raw.decode_content = True
#                 shutil.copyfileobj(r.raw, f)
#     except:
#         print("Error")
# def code():
#     code = ""
#     for i in range(10):
#         code += random.choice("aaaaaaaaaaaa")
#     return code
# while True:
#     num = 1000
#     data = "車ナンバープレート"
#     for i in range(int(num)):
#         link = image(data)
#         download_img(link, str(i))
#     print("OK")


  
# import requests
# import os, time, sys
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import chromedriver_binary # need to designate driver

# # launch chrome browser
# driver = webdriver.Chrome()
# # google image search
# driver.get('https://www.google.co.jp/imghp?hl=ja&tab=wi&ogbl')
# # execute search
# keyword = "フロントナンバープレート"
# driver.find_element_by_name('q').send_keys(keyword, Keys.ENTER)

# current_url = driver.current_url
# html = requests.get(current_url)
# bs = BeautifulSoup(html.text, 'lxml')
# images = bs.find_all('img', limit=10)

# base_dir="train_data/web"
# os.makedirs(base_dir,exist_ok=True)

# WAIT_TIME = 1

# for i, img in enumerate(images, 1):
#     src = img.get('src')
#     try:
#         response = requests.get(src)
#         with open(base_dir + '/' + '{}.jpg'.format(i), 'wb') as f:
#             f.write(response.content)
#         time.sleep(WAIT_TIME)
#     except:
#         print("Error")

# driver.quit()

import os
import time
from selenium import webdriver
from PIL import Image
import io
import requests
import hashlib
import chromedriver_binary # need to designate driver
from selenium.webdriver.chrome.options import Options


# クリックなど動作後に待つ時間(秒)
sleep_between_interactions = 2
# ダウンロードする枚数
download_num = 1000
# 検索ワード
query ="車画像 横"# "フロントグリル ナンバープレート"
# 画像検索用のurl
search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

# chromedriverの設定
options = Options()
options.add_argument('--headless')
# driver = webdriver.Chrome('C:/Users/oono/Downloads/chromedriver_win32/chromedriver')

# サムネイル画像のURL取得
wd = webdriver.Chrome(chrome_options=options)
wd.get(search_url.format(q=query))

#適当に下までスクロールしてる--
for t in range(10):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(1.5)

# サムネイル画像のリンクを取得(ここでコケる場合はセレクタを実際に確認して変更する)
thumbnail_results = wd.find_elements_by_css_selector("img.rg_i")

# サムネイルをクリックして、各画像URLを取得
image_urls = set()
for img in thumbnail_results[:download_num]:
    print(img)
    try:
        img.click()
        time.sleep(sleep_between_interactions)
    except Exception:
        continue
    # 一発でurlを取得できないので、候補を出してから絞り込む(やり方あれば教えて下さい)
    # 'n3VNCb'は変更されることあるので、クリックした画像のエレメントをみて適宜変更する
    url_candidates = wd.find_elements_by_class_name('n3VNCb')
    for candidate in url_candidates:
        url = candidate.get_attribute('src')
        if url and 'https' in url:
            image_urls.add(url)
# 少し待たないと正常終了しなかったので3秒追加
time.sleep(sleep_between_interactions+3)
wd.quit()

# 画像のダウンロード
image_save_folder_path = "train_data/web"
os.makedirs(image_save_folder_path,exist_ok=True)

for url in image_urls:
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(image_save_folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=90)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
