# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19"}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings  
warnings.filterwarnings('ignore')

import os
from pathlib import Path
print(Path.cwd())
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train = pd.read_csv("../input/train.csv")
sales = pd.read_csv("../input/sales.csv")
navigation = pd.read_csv("../input/navigation.csv")
vimages = pd.read_csv("../input/vimages.csv")

# %% {"_uuid": "821b22453a1f931532e9fdef5d2ee444983e78c3"}
train.head()

# %% {"_uuid": "b68814a28eea6cbe136bb0d2739342026812ca63"}
train.info()

# %% {"_uuid": "e607ecd3e710007360a909b54f1b16a20b8db7a9"}
train.describe()

# %% {"_uuid": "43ecdcae96bcb50c0e79f87f05a8a7c81a6459df"}
train.nunique()

# %% {"_uuid": "297b5ad7ff535e9cec9937b648d4f970b2411d3a"}
train.isna().sum()

# %% [markdown] {"_uuid": "b6e549a8a65a0850a7a42f27fb46709b0a5bc70e"}
# # Target

# %% {"_uuid": "209ceabaa4194bfb2168879f7f7ed09892f4a029"}
sns.distplot(train.target)

# %% {"_uuid": "dfa6ac40b19b5ee3883702261a86a973e179b049"}
sns.boxplot(train.target, orient='v')

# %% {"_uuid": "36b75bea2d2019bc4d44661b2d613870fee08d6c"}
plt.scatter(range(train.shape[0]), np.sort(train.target.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Quantity', fontsize=12)

# %% [markdown] {"_uuid": "37038033b04fed29f96093461017b631f73b864b"}
# # Other variables in the train dataset

# %% [markdown] {"_uuid": "555ca375781733c35941a89adb8bbe98e7540da5"}
# ### product_type and product_gender

# %% {"_uuid": "5cdac0c67c73253ca509751fe0b2d260c3f4ba81"}
f, axes = plt.subplots(2, 2, figsize=(15, 5))

for ii,i in enumerate(['product_type','product_gender']):
    sns.countplot(train[i], ax =axes[ii][0])
    sns.barplot(train[i], train.target, ax=axes[ii][1])

# %% [markdown] {"_uuid": "e3cc5b1dbfc163ed15d463dcb83c2a93e7948a8c"}
# ### macro_function

# %% {"_uuid": "8c7483733a547d4fe7bfcf8151b8f0c8f48af209"}
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.macro_function, orient='v', ax=axes[0])
sns.barplot(y=train.macro_function, x=train.target, ax=axes[1])

# %% [markdown] {"_uuid": "5048fabe74df2c4320bd66c950014950abd13f20"}
# ### function

# %% {"_uuid": "0f9615f2d0369b62572441b817887ea75e124431"}
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.function, orient='v', ax=axes[0])
sns.barplot(y=train.function, x=train.target, ax=axes[1])

# %% [markdown] {"_uuid": "964c858706aea5f4c289c1d4de4427fe92e35057"}
# ### sub_function

# %% {"_uuid": "a3ae1f4f07e691140f7a43a19a89c6c5133ff9cd"}
plt.figure(figsize=(15, 10))
sns.countplot(y=train.sub_function, orient='h')


# %% [markdown] {"_uuid": "77a582c215818157b9ab417c978cb38018fc75f4"}
# ### model

# %% {"_uuid": "982360ab08a793529d3ae4d7c81ff2f7cf6bb636"}
train.model.value_counts(ascending=False).head(30)

# %% [markdown] {"_uuid": "f27351fc435bb78a9c465eaa21051c7b79810d28"}
# ### aesthetic_sub_line

# %% {"_uuid": "c6a0211e0783e9d9b755a488e1c3b47bfbf1ea07"}
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.aesthetic_sub_line, orient='v', ax=axes[0])
sns.barplot(y=train.aesthetic_sub_line, x=train.target, ax=axes[1])

# %% [markdown] {"_uuid": "3a2b812ab03d24f8d314641969df2435e48cc96b"}
# ### macro_material

# %% {"_uuid": "8c141ed91767718f214a26209ad07598e20f0b63"}
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.macro_material, orient='v', ax=axes[0])
sns.barplot(y=train.macro_material, x=train.target, ax=axes[1])

# %% [markdown] {"_uuid": "3ede66eb52d0a6d29b208c876479f790310a4647"}
# ### color

# %% {"_uuid": "de9192a1e6ce149318f2356ae0fb4b527a8592c7"}
train.color.value_counts(ascending=False).head(20)

# %% [markdown] {"_uuid": "402a137d6dfbc1653353bb183970c888cca071cb"}
# ### fr_FR_price

# %% {"_uuid": "6474de63da8c1fac5af4f6408a68fcdb9f95510c"}
plt.figure(figsize=(15, 10))
sns.distplot(train.fr_FR_price)

# %% {"_uuid": "88e91331d79513221a0904322e180b67feecd972"}
plt.figure(figsize=(15, 10))
sns.scatterplot(train.fr_FR_price, train.target)

# %% {"_uuid": "1bba8aaf4e328f809ce200ce2371e9a80d14db9e"}
plt.figure(figsize=(15,10))
sns.violinplot(x=train.product_type, y=train.target, inner="points")

# %% {"_uuid": "ec55a28fa267def8a679a19c6120d2313b5ca105"}
plt.figure(figsize=(15,10))
sns.violinplot(x=train.product_gender, y=train.target, inner="points")

# %% {"_uuid": "01630839fc431936c2c85ebfdcfeff9dd6e89cf3"}
plt.figure(figsize=(15,10))
sns.violinplot(y=train.macro_function, x=train.target, inner="points")

# %% [markdown] {"_uuid": "1a896b871083e80e03ffa6209eca13f7208eb494"}
# # Sales dataset

# %% {"_uuid": "a8efdf511058652a875c1b974713fdd965637b92"}
sales.head()

# %% {"_uuid": "8eb5d63bf2b525a3f11c94ef31e5ecbe35a3e022"}
sales.info()

# %% {"_uuid": "75e56b9547ba8ad03fd250cfeb3eaedf159e3d0b"}
sales.nunique()

# %% {"_uuid": "a6d1645e1907d538dc00caf5d1bafd91dda49180"}
sales.describe()

# %% {"_uuid": "a368fb80f8433ec267102edc0654accae461222f"}
sales.isna().sum()

# %% [markdown] {"_uuid": "cb8d555b561e42b8b5a5db3816d388a19a99bc78"}
# ### country_number

# %% {"_uuid": "6571894d4c9c32c14f069d7a09db386e52ea67e6"}
sales.country_number.value_counts(ascending=False).head()

# %% {"_uuid": "f3aa2a618ab465b8065d60be2c27457f0eeef791"}
sales.country_number.value_counts(ascending=False).head()

# %% [markdown] {"_uuid": "353acdfad517fc046f9624c9e182913a812b45f5"}
# ### sales_quantity

# %% {"_uuid": "19c8fd5ea14b43efedc9ebfc20acac386c868bb0"}
plt.figure(figsize=(15,5))
sns.distplot(sales.sales_quantity)

# %% [markdown] {"_uuid": "04c3772cea0c268de1d6858ebf1119f98430aa2f"}
# ### currency_rate_USD

# %% {"_uuid": "8086f0fac2018adc2bb31c32dd1343633a667e53"}
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_USD, ax=axes[0])
sns.scatterplot(sales.currency_rate_USD, sales.sales_quantity, ax=axes[1]) 

# %% [markdown] {"_uuid": "761038be62cb07695f7b8d57505b655c5ab35082"}
# ### currency_rate_GBP

# %% {"_uuid": "881a15ae593683b4227313e0ae11f372ed2064f9"}
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_GBP, ax=axes[0])
sns.scatterplot(sales.currency_rate_GBP, sales.sales_quantity, ax=axes[1])

# %% [markdown] {"_uuid": "3e5ef132583ed169b6acdceadde0396a7ce13e4a"}
# ### currency_rate_CNY

# %% {"_uuid": "9fa07224dd0de163c5525d0217771c0662691c91"}
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_CNY, ax=axes[0])
sns.scatterplot(sales.currency_rate_CNY, sales.sales_quantity, ax=axes[1])

# %% [markdown] {"_uuid": "f3d756d79d8ddf44743ab4b3106b3b24baba7b93"}
# ### currency_rate_JPY

# %% {"_uuid": "2cf6b06cdea90c1105e67c1be49f1b848d320822"}
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_JPY, ax=axes[0])
sns.scatterplot(sales.currency_rate_JPY, sales.sales_quantity, ax=axes[1])

# %% [markdown] {"_uuid": "f060966cd711161410243c8453bbe6946924d3fa"}
# ### currency_rate_KRW

# %% {"_uuid": "7bd9f62ce5cafb1b5a2651a7af31e3c9525f40cb"}
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_KRW, ax=axes[0])
sns.scatterplot(sales.currency_rate_KRW, sales.sales_quantity, ax=axes[1])

# %% {"_uuid": "6ee78307ef2ad9915955f547c49cfaf56734d487"}
var = ['TotalBuzzPost', 'TotalBuzz', 'NetSentiment',
       'PositiveSentiment', 'NegativeSentiment', 'Impressions']

f, axes = plt.subplots(3,2, figsize=(15,5))

for ii, i in enumerate(var):
    axes = axes.flatten()
    sns.distplot(sales[i], ax =axes[ii]).set(xlabel=i)

# %% {"_uuid": "9865d1d8f1c5ba7566ae9c7ee9703bea307d215a"}
f, axes = plt.subplots(3,2, figsize=(15,5))

for ii, i in enumerate(var):
    axes = axes.flatten()
    sns.scatterplot(sales[i], sales.sales_quantity, ax =axes[ii]).set(xlabel=i)

# %% [markdown] {"_uuid": "d5e32c572e5b4b346409feb7d0b82d0aa0a12fac"}
# # Navigation dataset

# %% {"_uuid": "370754487a740d426ed62332f51c50ea973f2d7e"}
navigation.head()

# %% {"_uuid": "0fd4a9ff40238ef63cac72735169bcb784d18ad7"}
navigation.info()

# %% {"_uuid": "aa0a71720445065db9ccdd367b41203fe1d0ac6f"}
navigation.isna().sum()

# %% {"_uuid": "6151cbc1eb1f7a901fba23b39e737da97f834764"}
navigation.nunique()

# %% [markdown] {"_uuid": "375c33dd88b530e01a0764b80421e070d4a4947b"}
# ### page_views

# %% {"_uuid": "2e9e8cf309c51c173d7b1bae89d8dd6cc2dc02c0"}
plt.figure(figsize=(15,5))
sns.distplot(navigation.page_views)

# %% {"_uuid": "1e90bb1c3373ae0e57f5d4f45a3515f5354f9c9b"}
plt.figure(figsize=(15,5))
sns.boxplot(navigation.page_views)

# %% [markdown] {"_uuid": "c2a7ce05bbbd117cfe8d409bafeacf56ab7ac08c"}
# ### traffic_source and traffic_source vs page_views

# %% {"_uuid": "e5db829e2047bca2565db7cd5e884916f425b264"}
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.traffic_source, ax=axes[0])
sns.barplot(navigation.traffic_source, navigation.page_views, ax=axes[1])

# %% [markdown] {"_uuid": "48c32554288d1f234143dd386880537569ae6f09"}
# ### addtocart

# %% {"_uuid": "ca873d85774c8172fce03f10a9f1854b8266c640"}
navigation.addtocart.value_counts()

# %% [markdown] {"_uuid": "1bf7bcf9619a91ded6d8d5e2f2a362435daad73e"}
# ### day_visit and day_visit vs page_views

# %% {"_uuid": "d8837e0344a1ca3d3e347d827b30ebf1a3210b69"}
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.day_visit, ax=axes[0])
sns.barplot(navigation.day_visit, navigation.page_views, ax=axes[1])

# %% [markdown] {"_uuid": "820cac2b241fb037a6a3655e42b5be69ff09f75d"}
# ### month_visit and month_visit vs page_views

# %% {"_uuid": "84d26fc85a998934bbeec89f795ebc89564f6eb3"}
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.month_visit, ax=axes[0])
sns.barplot(navigation.month_visit, navigation.page_views, ax=axes[1])

# %% [markdown] {"_uuid": "f62ba151734d8d91564ff8a12712ce50ac6d6c75"}
# ### website_version_zone_number and website_version_zone_number vs page_views

# %% {"_uuid": "cdf11487807e0e686d5232798703f537c107eee1"}
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.website_version_zone_number, ax=axes[0])
sns.barplot(navigation.website_version_zone_number, navigation.page_views, ax=axes[1])

# %% [markdown] {"_uuid": "1ce19c0f3225b7c99ba7af58dfb4d262f577901e"}
# ### website_version_country_number and website_version_country_number vs page_views

# %% {"_uuid": "40c008a0fd84c2b44371bfb07926b7cde8b7c0ce"}
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.website_version_country_number, ax=axes[0])
sns.barplot(navigation.website_version_country_number, navigation.page_views, ax=axes[1])
