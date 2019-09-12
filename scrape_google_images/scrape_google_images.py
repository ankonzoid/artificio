"""

 scrape_google_images.py  (author: Anson Wong / git: ankonzoid)

 Scrapes the top k images from Google Images for a particular query.

"""
import os, json, time
import requests
import urllib.request as urllibreq
from bs4 import BeautifulSoup

def scrape_google_images(query, k, outDir, dt_stall):
    print("Scraping top k={} Google images for query: '{}'".format(k, query))

    # Create output directory
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # Construct image url to scrape from
    url = "https://www.google.co.in/search?q=" + \
          '+'.join(query.split()) + "&source=lnms&tbm=isch"
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

    # Download html + parse it to get all image urls
    html = urllibreq.urlopen(urllibreq.Request(url, headers=header))
    soup = BeautifulSoup(html,'html.parser')
    urls = [json.loads(x.text)["ou"]
            for x in soup.find_all("div", {"class": "rg_meta"})]
    print("Parsed {} google image urls!".format(len(urls)))

    # Download top-k images
    for i, url in enumerate(urls[:min(k, len(urls))]):
        imgFile = os.path.join(outDir, query + "-" + str(i+1) + ".jpg")
        print("[{}/{}] Downloading image {} to '{}'...".format(i+1, k, url, imgFile))
        try:
            with open(imgFile, 'wb') as handle:
                response = requests.get(url, stream=True)
                if not response.ok:
                    print(response)
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
            time.sleep(dt_stall)  # force pause
        except:
            raise Exception("  -> Failed downloading: " + url)

query = "drake"  # google images query text
k = 5  # top-k images will be scraped
outDir = os.path.join(os.getcwd(), "output")  # output directory to save images
dt_stall = 2  # number of seconds to stall between image scrapes
scrape_google_images(query=query, k=k, outDir=outDir, dt_stall=dt_stall)