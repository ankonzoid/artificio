"""

 scrape_google_images.py  (author: Anson Wong / git: ankonzoid)

 Scrapes the top k images from Google Images for a particular query.

"""
def main():
    query_text = "drake"  # google images query text
    k = 5  # top k images will be scraped
    output_dir = "output"  # output directory to save images
    dt_wait = 1  # number of seconds to wait between image scrapes
    scrape_google_images(query_text=query_text, k=k,
                         output_dir=output_dir, dt_wait=dt_wait)


# ***************************
#
# Functions
#
# ***************************
from bs4 import BeautifulSoup
import urllib.request as urllibreq
import os, json, time

def scrape_google_images(query_text="deepmind",
                         k=5, output_dir="output", dt_wait=1):

    print("Downloading top k={} Google images for query: '{}'".format(k, query_text))

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct url to scrape from
    query_processed = '+'.join(query_text.split())
    url = "https://www.google.co.in/search?q=" + query_processed + "&source=lnms&tbm=isch"
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

    # Download html and soup it to get all image urls
    html = urllibreq.urlopen(urllibreq.Request(url, headers=header))
    soup = BeautifulSoup(html,'html.parser')
    image_urls = []
    for a in soup.find_all("div",{"class": "rg_meta"}):
        link = json.loads(a.text)["ou"]
        image_urls.append(link)

    print("Found {} google image links!".format(len(image_urls)))

    # Download top k images
    for i, (image_url) in enumerate(image_urls[:min(k, len(image_urls))]):
        image_name = query_text + "-" + str(i+1) + ".jpg"
        image_filename = os.path.join(output_dir, image_name)

        print("[{}/{}] Downloading image to: '{}'...".format(i+1, k, image_filename))
        try:
            urllibreq.urlretrieve(image_url, image_filename)
            time.sleep(dt_wait)  # force pause
        except Exception as error_message:
            print("  -> Failed download".format(image_url))

# Driver
if __name__ == '__main__':
    main()