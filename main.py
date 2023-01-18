"""
This file calls all needed methods to retrieve the dataset and train the machine learning model

By Cole Koryto
Inspired from: https://oxylabs.io/blog/python-web-scraping
"""

import pprint
import traceback
from DataExtraction import *
import pandas as pd
from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager  # pip3 install webdriver_manager

# gets a new dataset or updates existing dataset file
def updateDataset():

    # sets up main selenium driver
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.minimize_window()

    # gets all FIPS codes from file
    fipsCodesDf = pd.read_excel("US_FIPS_Codes.xlsx", dtype=object)

    # extracts all data from every FIPS code
    for codeNum, code in enumerate(fipsCodesDf["FIPS Code"]):
        print(f"\nGetting FIPS code: {code} {codeNum + 1} of {len(fipsCodesDf['FIPS Code'])}")
        try:
            DataExtract.extractData(driver, code)
        except Exception as e:
            print(f"Error encountered with FIPS code {code}")
            traceback.print_exc()

    # exits chrome webpage
    driver.quit()

    # finds latitude and longitude from addresses
    # todo very slow right now 1 sec per request, maybe use Google maps api?
    """print("Finding latitudes and longitudes")
    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    dataExtract.totalRentalDf['Location'] = dataExtract.totalRentalDf['Addresses'].apply(geocode)
    dataExtract.totalRentalDf['Point'] = dataExtract.totalRentalDf['Location'].apply(lambda loc: tuple(loc.point) if loc else None)
    dataExtract.totalRentalDf[['Latitude', 'Longitude', 'Altitude']] = pd.DataFrame( dataExtract.totalRentalDf['Point'].tolist(), index= dataExtract.totalRentalDf.index)"""

    # outputs all rental results
    DataExtract.totalRentalDf.to_csv("Total Rental Results.csv", index=False)

# runs data extraction for all FIPS codes in US
def main():

    # sets up time tracking of program
    startTime = time.time()
    cpuStartTime = time.process_time()

    # gets a new dataset or updates existing dataset file
    #updateDataset()




    # outputs program time taken in total time and time CPU time
    elapsedTime = time.time() - startTime
    executionTime = time.process_time() - cpuStartTime
    print("\n\n***Program Performance***")
    print("Elapsed time: " + str(elapsedTime) + " seconds")
    print("CPU execution time: " + str(executionTime) + " seconds (without I/O or resource waiting time)")

# runs all main code
if __name__ == "__main__":
    main()
