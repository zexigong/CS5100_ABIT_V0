from abit_transforms import Pattern, TransformCascade, generate_cookie_sales
import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abit_transforms.transforms.hht import HHTTransform
from abit_transforms.transforms.stft import STFTTransform
from abit_transforms.transforms.wavelet import WaveletTransform
from abit_transforms.cascades import TripleTransformCascade
import matplotlib.ticker as mtick
from abit_transforms import pattern_detection


def compute_correlations(walmart_weekly, trends_weekly):

    # Combine dataframes aligned by date
    combined = pd.concat([walmart_weekly, trends_weekly], axis=1, join='inner')

    # Rename Walmart column if it exists
    if 'Weekly_Sales' in combined.columns:
        combined = combined.rename(columns={'Weekly_Sales': 'Walmart_Sales'})
    elif 'Walmart_Sales' not in combined.columns:
        raise ValueError("No Walmart sales column found!")

    # Compute correlation
    corr = combined.corr()['Walmart_Sales'].drop('Walmart_Sales')
    return corr


def main():
    walmart_keywords = ['black friday', 'cyber monday', 'christmas gifts', 'toys for kids', 'back to school']
    walmart_keywords_nonseasonal = ['walmart laptop', 'walmart headphones', 'walmart smartphone', 'walmart blender', 'walmart coffee maker']
    walmart_keywords_seasonal_two = ['walmart swimsuit', 'walmart wool', 'walmart jacket', 'walmart gingerbread', 'walmart cookie']
    zara_keywords = ['cotton', 'jacket', 'wool', 'summer dress', 'jeans']
    # load sales data in using pandas
    walmart_data = pd.read_csv('walmart_sales.csv', parse_dates=['Date'])
    # zara_data = pd.read_csv('zara_sales.csv', parse_dates=['D'])
    walmart_data = walmart_data.set_index('Date')

    total_sales = walmart_data.groupby("Date")[["Weekly_Sales"]].sum()
    total_sales = total_sales.reset_index()  # make 'Date' a column
    total_sales = total_sales.rename(columns={'Date': 'date'})

    # plt.figure(figsize=(12, 5))
    # plt.plot(total_sales['date'], total_sales['Weekly_Sales'])
    # plt.title("Walmart Weekly Sales (All Stores Aggregated)")
    # plt.xlabel("Date")
    # plt.ylabel("Sales")
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    # # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("sales_plot.png")


    # get google trends data for keywords
    # pytrends = TrendReq(hl='en-US', tz=360)
    # pytrends.build_payload(walmart_keywords, timeframe='2010-01-01 2012-12-31', geo='US')

    # trends_data = pytrends.interest_over_time()
    # trends_data.to_csv("google_trends_walmart.csv")

    # run correlation between raw weekly sales data and keyword search patterns

    total_searches = pd.read_csv("google_trends_walmart_nonseasonal.csv", parse_dates=['date'])

    if "isPartial" in total_searches.columns:
        total_searches = total_searches.drop(columns=["isPartial"])

    for col in total_searches.columns:
        if col != 'date':
            total_searches[col] = pd.to_numeric(total_searches[col], errors='coerce')

    total_sales['date'] = pd.to_datetime(total_sales['date'], format="mixed")
    total_searches['date'] = pd.to_datetime(total_searches['date'])
    # Ensure 'date' is index in both dataframes
    total_sales = total_sales.set_index('date')
    total_searches = total_searches.set_index('date')

    # # Optionally, resample both to the same weekly frequency (Monday)
    total_sales = total_sales.resample('W-MON').sum()
    total_searches = total_searches.resample('W-MON').mean()

    correlations = compute_correlations(total_sales, total_searches)

    print("\n=== Correlation between Walmart Sales and Google Trends ===")
    print(correlations)

    # run both through cascade of transforms then correlate output again
    triple_cascade = TripleTransformCascade('wavelet', 'stft', 'hht')

    print(triple_cascade.analyze(total_sales['Weekly_Sales'].values))
    # patterns_trends = dict()
    # for keyword in total_searches.columns:
    #     patterns_trends[keyword], _ = triple_cascade.analyze(total_searches[keyword].values)

    # print(patterns_trends)
    # 

    # print(pattern_detection.detect_all_patterns(total_sales['Weekly_Sales'].values, 'WHS', total_sales.index, total_sales['Weekly_Sales'].values))

    # print(pattern_detection.detect_all_patterns(total_searches['walmart jacket'].values, 'WHS', total_searches.index, total_searches['walmart jacket'].values))


    plt.figure(figsize=(12, 5))
    plt.plot(total_searches.index, total_searches['laptop'])
    plt.title("Laptop Searches")
    plt.xlabel("Date")
    plt.ylabel("Searches")
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("laptop_plot.png")

if __name__ == "__main__":
    main()


