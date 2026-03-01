
Med Device Topic Sepecialization
Author
Jackson McDowell

import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin, urlparse
import time, random
import lda
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

Zimmer scraping Scraped the product and solution pages. Extracted a product name and a short description. I keep only rows with enough text to be useful for topic modeling.

headers = {"User-Agent": "Mozilla/5.0"}

company = "Zimmer Biomet"
start_url = "https://www.zimmerbiomet.com/en/products-and-solutions.html"

r = requests.get(start_url, headers=headers)
soup = BeautifulSoup(r.text, "html.parser")


links = []
for a in soup.select("a[href]"):
    href = a.get("href", "").strip()
    full = urljoin(start_url, href)

    if urlparse(full).netloc != "www.zimmerbiomet.com":
        continue

    if any(x in full.lower() for x in ["products-and-solutions", "product", "solutions"]) and "#" not in full:
        links.append(full)

links = list(dict.fromkeys(links))

print("Links found:", len(links))

products = []
##80 was a good balance of runtime and variety of products, I was still able to notice a pattern with this number. This reasoning is the same for all the slicing I did with other websites. 
for url in links[:80]: 
    try:
        rr = requests.get(url, headers=headers, timeout=20)
        ss = BeautifulSoup(rr.text, "html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else ss.title.get_text(strip=True)

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content", "").strip() if meta else ""

        if not desc:
            main = ss.select_one("main") or ss
            p = main.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if name and len(name) >= 4 and len(desc) >= 20:
            products.append({
                "company": company,
                "product": name,
                "description": desc,
                "url": url
            })

    except Exception:
        continue

df = pd.DataFrame(products)

df = df.drop_duplicates(subset=["company", "product"])

if "url" in df.columns:
    df = df.drop(columns=["url"])

df = df[df["description"].str.len() > 40]

df["text"] = (df["product"] + " " + df["description"]).str.lower()

df = df.reset_index(drop=True)

print(df.head())
print("Rows:", len(df))

df.to_csv("zimmer_products_clean.csv", index=False)

import pandas as pd

df = pd.read_csv("zimmer_products_clean.csv")
df.head()

company	product	description	text
0	Zimmer Biomet	OrthoGrid	OrthoGrid’s digital platform creates a robust ...	orthogrid orthogrid’s digital platform creates...
1	Zimmer Biomet	Biologics Solutions	Zimmer Biomet Biologics has a range of solutio...	biologics solutions zimmer biomet biologics ha...
2	Zimmer Biomet	Bone Cement for Arthroplasty Procedures	Zimmer Biomet offers surgeons a complete portf...	bone cement for arthroplasty procedures zimmer...
3	Zimmer Biomet	Craniomaxillofacial (CMF)	Zimmer Biomet offers a wide array of devices f...	craniomaxillofacial (cmf) zimmer biomet offers...
4	Zimmer Biomet	mymobility®Care Management Platform	mymobility with Apple Watch is a digital care ...	mymobility®care management platform mymobility...
Stryker I filtered out non-product pages (careers, investors, policies, etc.), sample a subset for runtime, and extract names + descriptions. This website had a lot of broad coverage. I also kept only rows with enough text to be useful for topic modeling.

headers = {"User-Agent": "Mozilla/5.0"}

company = "Stryker"
start_url = "https://www.stryker.com/us/en/site-map.html"

r = requests.get(start_url, headers=headers, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")

#links that look like product pages
links = []
for a in soup.select("a[href]"):
    href = a.get("href", "").strip()
    full = urljoin(start_url, href)

    if urlparse(full).netloc != "www.stryker.com":
        continue

    full_l = full.lower()

    if any(bad in full_l for bad in [
        "product-inquiry", "contact", "careers", "investors",
        "privacy", "terms", "/about/", "/news/", "/events/",
        "/training", "/services/", "/company", "/locations"
    ]):
        continue

    if any(x in full_l for x in [
        "/products", "implant", "system", "platform", "instrument"
    ]) and "#" not in full:
        links.append(full)

links = list(dict.fromkeys(links))
print("Links found:", len(links))

random.seed(1)
links = random.sample(links, min(120, len(links)))

products = []

for url in links:
    try:
        time.sleep(random.uniform(.1, .4))

        rr = requests.get(url, headers=headers, timeout=20)
        ss = BeautifulSoup(rr.text, "html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else (ss.title.get_text(strip=True) if ss.title else "")

        if len(name.split()) <= 1:
            continue

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content", "").strip() if meta else ""

        if not desc:
            og = ss.select_one("meta[property='og:description']")
            desc = og.get("content", "").strip() if og else ""

        if not desc:
            main = ss.select_one("main") or ss
            p = main.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if name and len(name) >= 4 and len(desc) >= 20:
            products.append({
                "company": company,
                "product": name,
                "description": desc,
                "url": url
            })

    except Exception:
        continue

df = pd.DataFrame(products)

df["product"] = df["product"].str.replace("| Stryker","", regex=False)



df = df.drop_duplicates(subset=["company", "product"])

if "url" in df.columns:
    df = df.drop(columns=["url"])

df = df[df["description"].astype(str).str.len() > 40]

df["text"] = (df["product"].astype(str) + " " + df["description"].astype(str)).str.lower()

df = df.reset_index(drop=True)

print(df.head())
print("Rows:", len(df))

df.to_csv("stryker_no_noise_2.csv", index=False)

import pandas as pd
df = pd.read_csv("stryker_no_noise_2.csv")
df.head()

company	product	description	text
0	Stryker	LITe Pedicle Access Solution	A less invasive pedicle access solution design...	lite pedicle access solution a less invasive p...
1	Stryker	IsoFlex SE	Our premier stretcher surface, IsoFlex SE, add...	isoflex se our premier stretcher surface, isof...
2	Stryker	High Speed Drills	The most comprehensive and customizable neuros...	high speed drills the most comprehensive and c...
3	Stryker	Aleutian Lateral	The Aleutian Lateral Interbody System is desig...	aleutian lateral the aleutian lateral interbod...
4	Stryker	Vitoss BA	A synthetic bone graft substitute with bioacti...	vitoss ba a synthetic bone graft substitute wi...
Olympus Organizes products by category pages. I collect category links, then collect individual product links from those categories, and extract product name + description.

headers = {"User-Agent": "Mozilla/5.0"}

company = "Olympus"
start_url = "https://medical.olympusamerica.com/products"

r = requests.get(start_url, headers=headers)
soup = BeautifulSoup(r.text, "html.parser")

category_links = []

for a in soup.select("a[href]"):
    href = a.get("href","")
    full = urljoin(start_url, href)

    if "/products/" in full and "#" not in full:
        category_links.append(full)

category_links = list(set(category_links))
print("Category pages:", len(category_links))


product_links = []

for cat in category_links:
    try:
        rr = requests.get(cat, headers=headers)
        ss = BeautifulSoup(rr.text,"html.parser")

        for a in ss.select("a[href]"):
            href = a.get("href","")
            full = urljoin(cat, href)

            if "/products/" in full and full != cat:
                product_links.append(full)

    except:
        pass

product_links = list(set(product_links))
print("Product pages:", len(product_links))


products = []

for url in product_links:
    try:
        time.sleep(random.uniform(.2,.5))

        rr = requests.get(url, headers=headers)
        ss = BeautifulSoup(rr.text,"html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else ""

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content","").strip() if meta else ""

        if not desc:
            p = ss.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if len(name) > 3 and len(desc) > 25:
            products.append({
                "company": company,
                "product": name,
                "description": desc
            })

    except:
        pass

df = pd.DataFrame(products)

df = df[~df["product"].str.contains("standard|innovation|solutions", case=False, na=False)]

df = df.drop_duplicates()

df["text"] = (df["product"] + " " + df["description"]).str.lower()

print("Rows:", len(df))

df.to_csv("olympus_clean_3.csv", index=False)

import pandas as pd
df = pd.read_csv("olympus_clean_3.csv")
df.head()

company	product	description	text
0	Olympus	ERCP Guidewires	ERCP Guidewires ERCP access solutions tailored...	ercp guidewires ercp guidewires ercp access so...
1	Olympus	Contained Tissue Extraction System	Overview Pneumoliner GUARDENIA Videos ...	contained tissue extraction system overview ...
2	Olympus	Resection in Saline Electrodes	Olympus Resection in Saline Electrodes include...	resection in saline electrodes olympus resecti...
3	Olympus	VisiGlide™ Guidewires	Single-use Olympus VisiGlide Guidewires are pa...	visiglide™ guidewires single-use olympus visig...
4	Olympus	TJF-Q190V Duodenoscope	The TJF-Q190V is the newest generation of Olym...	tjf-q190v duodenoscope the tjf-q190v is the ne...
I combined the cleaned company datasets into one file for the topic model.

import glob

folder = r"C:\Users\JacksonMcDowell\Desktop\Unstructured\final_data"

files = glob.glob(folder + r"\*.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df.to_csv(folder + r"\combined_med_device.csv", index=False)

print("Combined", len(files), "files")

Combined 4 files
Topic Modeling I set k=6 to balance interpretability few enough topics to label with variety enough to separate major device themes.

path = r"C:\Users\JacksonMcDowell\Desktop\Unstructured\final_data\combined_med_device.csv"
df = pd.read_csv(path)

if "text" not in df.columns:
    df["text"] = (df["product"].astype(str) + " " + df["description"].astype(str)).str.lower()
else:
    df["text"] = df["text"].astype(str).str.lower()

df = df.dropna(subset=["company", "text"]).copy()

df["text_clean"] = (
    df["text"]
      .str.replace(r"\s+", " ", regex=True)
      .str.replace(r"[^a-z\s]", " ", regex=True)  
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)
df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)

vectorizer = CountVectorizer(
    stop_words=list(ENGLISH_STOP_WORDS), 
    min_df=2,      
    max_df=0.90    
)

X = vectorizer.fit_transform(df["text_clean"])
vocab = vectorizer.get_feature_names_out()

K = 6 
model = lda.LDA(n_topics=K, n_iter=800, random_state=1)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 10

for k, topic_dist in enumerate(topic_word):
    top_idx = np.argsort(topic_dist)[-n_top_words:][::-1]
    top_terms = vocab[top_idx]
    print(f"Topic {k}: " + ", ".join(top_terms))

doc_topic = model.doc_topic_ 
for k in range(K):
    df[f"topic_{k}"] = doc_topic[:, k]

company_topic_means = df.groupby("company")[[f"topic_{k}" for k in range(K)]].mean()
print(company_topic_means)

for comp, row in company_topic_means.iterrows():
    top = row.sort_values(ascending=False)
    top_str = ", ".join([f"{t} ({v:.3f})" for t, v in top.items()])
    print(f"Company: {comp}, Topics: {top_str}")

INFO:lda:n_documents: 578
INFO:lda:vocab_size: 1634
INFO:lda:n_words: 13927
INFO:lda:n_topics: 6
INFO:lda:n_iter: 800
INFO:lda:<0> log likelihood: -149124
INFO:lda:<10> log likelihood: -100491
INFO:lda:<20> log likelihood: -98259
INFO:lda:<30> log likelihood: -97315
INFO:lda:<40> log likelihood: -96553
INFO:lda:<50> log likelihood: -96055
INFO:lda:<60> log likelihood: -95783
INFO:lda:<70> log likelihood: -95334
INFO:lda:<80> log likelihood: -94946
INFO:lda:<90> log likelihood: -94762
INFO:lda:<100> log likelihood: -94602
INFO:lda:<110> log likelihood: -94484
INFO:lda:<120> log likelihood: -94362
INFO:lda:<130> log likelihood: -94160
INFO:lda:<140> log likelihood: -94160
INFO:lda:<150> log likelihood: -94217
INFO:lda:<160> log likelihood: -94166
INFO:lda:<170> log likelihood: -94090
INFO:lda:<180> log likelihood: -94015
INFO:lda:<190> log likelihood: -93859
INFO:lda:<200> log likelihood: -93752
INFO:lda:<210> log likelihood: -93799
INFO:lda:<220> log likelihood: -93771
INFO:lda:<230> log likelihood: -93816
INFO:lda:<240> log likelihood: -93677
INFO:lda:<250> log likelihood: -93830
INFO:lda:<260> log likelihood: -93658
INFO:lda:<270> log likelihood: -93662
INFO:lda:<280> log likelihood: -93676
INFO:lda:<290> log likelihood: -93550
INFO:lda:<300> log likelihood: -93386
INFO:lda:<310> log likelihood: -93410
INFO:lda:<320> log likelihood: -93435
INFO:lda:<330> log likelihood: -93213
INFO:lda:<340> log likelihood: -93381
INFO:lda:<350> log likelihood: -93359
INFO:lda:<360> log likelihood: -93431
INFO:lda:<370> log likelihood: -93398
INFO:lda:<380> log likelihood: -93405
INFO:lda:<390> log likelihood: -93307
INFO:lda:<400> log likelihood: -93387
INFO:lda:<410> log likelihood: -93331
INFO:lda:<420> log likelihood: -93277
INFO:lda:<430> log likelihood: -93315
INFO:lda:<440> log likelihood: -93275
INFO:lda:<450> log likelihood: -93250
INFO:lda:<460> log likelihood: -93237
INFO:lda:<470> log likelihood: -93243
INFO:lda:<480> log likelihood: -93259
INFO:lda:<490> log likelihood: -93346
INFO:lda:<500> log likelihood: -93189
INFO:lda:<510> log likelihood: -93371
INFO:lda:<520> log likelihood: -93313
INFO:lda:<530> log likelihood: -93362
INFO:lda:<540> log likelihood: -93364
INFO:lda:<550> log likelihood: -93312
INFO:lda:<560> log likelihood: -93376
INFO:lda:<570> log likelihood: -93316
INFO:lda:<580> log likelihood: -93269
INFO:lda:<590> log likelihood: -93333
INFO:lda:<600> log likelihood: -93264
INFO:lda:<610> log likelihood: -93294
INFO:lda:<620> log likelihood: -93289
INFO:lda:<630> log likelihood: -93261
INFO:lda:<640> log likelihood: -93274
INFO:lda:<650> log likelihood: -93229
INFO:lda:<660> log likelihood: -93300
INFO:lda:<670> log likelihood: -93324
INFO:lda:<680> log likelihood: -93254
INFO:lda:<690> log likelihood: -93338
INFO:lda:<700> log likelihood: -93279
INFO:lda:<710> log likelihood: -93274
INFO:lda:<720> log likelihood: -93246
INFO:lda:<730> log likelihood: -93231
INFO:lda:<740> log likelihood: -93337
INFO:lda:<750> log likelihood: -93295
INFO:lda:<760> log likelihood: -93257
INFO:lda:<770> log likelihood: -93271
INFO:lda:<780> log likelihood: -93292
INFO:lda:<790> log likelihood: -93319
INFO:lda:<799> log likelihood: -93285
Topic 0: olympus, use, guidewire, imaging, single, endoscopic, endoscope, videoscope, flexible, high
Topic 1: surgical, camera, platform, advanced, control, patient, head, delivers, technology, visualization
Topic 2: platform, tissue, procedures, bone, multiple, elite, designed, specialties, contained, patients
Topic 3: bf, bronchoscope, olympus, evis, iii, exera, empower, bone, laser, diagnostic
Topic 4: performance, designed, blades, guidewires, visiglide, locking, device, line, instruments, efficiency
Topic 5: zimmer, biomet, products, solutions, patient, hip, surgeons, knee, offers, healthcare
                topic_0   topic_1   topic_2   topic_3   topic_4   topic_5
company                                                                  
Olympus        0.256554  0.114551  0.092291  0.264626  0.228305  0.043672
Stryker        0.065149  0.224870  0.266143  0.154632  0.185029  0.104178
Zimmer Biomet  0.005818  0.154324  0.042854  0.008194  0.020314  0.768496
Company: Olympus, Topics: topic_3 (0.265), topic_0 (0.257), topic_4 (0.228), topic_1 (0.115), topic_2 (0.092), topic_5 (0.044)
Company: Stryker, Topics: topic_2 (0.266), topic_1 (0.225), topic_4 (0.185), topic_3 (0.155), topic_5 (0.104), topic_0 (0.065)
Company: Zimmer Biomet, Topics: topic_5 (0.768), topic_1 (0.154), topic_2 (0.043), topic_4 (0.020), topic_3 (0.008), topic_0 (0.006)
Visualization for presentation

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
company_topic_means = df.groupby("company")[[f"topic_{k}" for k in range(K)]].mean().reset_index()
company_topic_means_melted = company_topic_means.melt(id_vars="company",
                                                    var_name="topic",
                                                    value_name="proportion")
plt.figure(figsize=(10, 6))
sns.barplot(data=company_topic_means_melted, x="company", y="proportion", hue="topic")
plt.title("Average Topic Proportions by Company")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

