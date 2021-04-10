#!/usr/bin/env python3
# Graphic generation for ATMO 324 by Sam Gardner <stgardner4@tamu.edu>, Drew Davis <acdavis01@tamu.edu>, Ashley Palm <ashleyp0301@tamu.edu>

import pandas as pd
pd.options.mode.chained_assignment = None
from metpy import calc as mpcalc
from metpy.units import units
import datetime as dt
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from cartopy import crs as ccrs
from cartopy import feature as cfeat
import numpy as np
import time
import xarray as xr
from math import floor
from math import ceil

def generatePlot(filename, startDate, endDate):
    # read and parse csv
    pandasData = pd.read_csv(filename, na_values="null")
    pandasData = pandasData.dropna(how="any", subset=["station", "valid", "lon", "lat", "tmpc", "drct", "sknt"])
    # convert valid column to datetime objects
    pandasData["valid"] = pd.to_datetime(pandasData["valid"], dayfirst=True)
    # filter to just the dates we want
    targetData = pandasData.loc[(pandasData["valid"] >= startDate) & (pandasData["valid"] <= endDate)]
    # get average temperature over the whole state throughout the event
    tempsOnTargetDates = targetData["tmpc"].tolist()
    averageTempAllStations = sum(tempsOnTargetDates) / len(tempsOnTargetDates)
    # Get deviations of every report from the average
    deviations = pandasData["tmpc"] - averageTempAllStations
    targetData["deviation"] = deviations
    # convert wind dir and speed into u and v components
    windDir = targetData["drct"].values * units.degrees
    windSpd = targetData["sknt"].values * units.knots
    uWind, vWind = mpcalc.wind_components(windSpd, windDir)
    targetData["uWind"] = uWind
    targetData["vWind"] = vWind
    # Average the temperature reports for each station
    averagedData = pd.DataFrame(columns=["station", "lon", "lat", "uWind", "vWind", "avgT", "lowT", "highT", "avgD"])
    for stationID in targetData["station"].values:
        if stationID in averagedData["station"].values:
            continue
        stationData = targetData.loc[targetData["station"] == stationID]
        avgTempAtStation = sum(stationData["tmpc"].values) / len(stationData["tmpc"].values)
        # Also collect the max and min temperature reported by each station
        highT = max(stationData["tmpc"].values)
        lowT = min(stationData["tmpc"].values)
        avgDeviationAtStation = sum(stationData["deviation"].values) / len(stationData["deviation"].values)
        avgUWindAtStation = sum(stationData["uWind"].values) / len(stationData["uWind"].values)
        avgVWindAtStation = sum(stationData["vWind"].values) / len(stationData["vWind"].values)
        stationAvgData = pd.DataFrame([[stationID, stationData["lon"].values[0], stationData["lat"].values[0], avgUWindAtStation, avgVWindAtStation, avgTempAtStation, highT, lowT, avgDeviationAtStation]], columns=["station", "lon", "lat", "uWind", "vWind", "avgT", "lowT", "highT", "avgD"])
        averagedData = pd.concat([averagedData, stationAvgData])
    # average the min and max temperatures to generate deviations for each station
    avgHigh = sum(averagedData["highT"].values) / len(averagedData["highT"].values)
    highDeviations = averagedData["highT"] - avgHigh
    averagedData["highD"] = highDeviations
    avgLow = sum(averagedData["lowT"].values) / len(averagedData["lowT"].values)
    lowDeviations = averagedData["lowT"] - avgLow
    averagedData["lowD"] = lowDeviations
    # Interpolate temp data where we have no stations
    latmin = 25.837377
    latmax = 36.600704
    lonmin = -106.745646
    lonmax = -93.508292
    # create mesh
    lonGrid = np.linspace(lonmin, lonmax, 1000)
    latGrid = np.linspace(latmin, latmax, 1000)
    lonGrid, latGrid = np.meshgrid(lonGrid, latGrid)
    # get interpolated data
    avgTempGrid = griddata((averagedData["lon"].values, averagedData["lat"].values), averagedData["avgD"].values, (lonGrid, latGrid), method="linear")
    highTempGrid = griddata((averagedData["lon"].values, averagedData["lat"].values), averagedData["highD"].values, (lonGrid, latGrid), method="linear")
    lowTempGrid = griddata((averagedData["lon"].values, averagedData["lat"].values), averagedData["lowD"].values, (lonGrid, latGrid), method="linear")

    # Create plot figure
    avgFig = plt.figure()
    plt.axis("off")
    plt.title("Average T deviations from statewide avg. T\n" + startDate.strftime("%Y-%m-%d") + " through " + endDate.strftime("%Y-%m-%d"))
    highFig = plt.figure()
    plt.axis("off")
    plt.title("High T deviations from statewide avg. high T\n" + startDate.strftime("%Y-%m-%d") + " through " + endDate.strftime("%Y-%m-%d"))
    lowFig = plt.figure()
    plt.axis("off")
    plt.title("Low T deviations from statewide avg. low T\n" + startDate.strftime("%Y-%m-%d") + " through " + endDate.strftime("%Y-%m-%d"))
    avgAx = avgFig.add_subplot(1, 1, 1, projection=ccrs.LambertCylindrical())
    avgAx.add_feature(cfeat.LAND)
    avgAx.add_feature(cfeat.OCEAN)
    avgAx.add_feature(cfeat.COASTLINE)
    avgAx.add_feature(cfeat.BORDERS, linestyle="-")
    avgAx.add_feature(cfeat.LAKES, alpha=0.5)
    avgAx.add_feature(cfeat.RIVERS, alpha=0.5)
    avgAx.add_feature(cfeat.STATES, linestyle=":")
    avgAx.set_extent((lonmin, lonmax, latmin, latmax))
    highAx = highFig.add_subplot(1, 1, 1, projection=ccrs.LambertCylindrical())
    highAx.add_feature(cfeat.LAND)
    highAx.add_feature(cfeat.OCEAN)
    highAx.add_feature(cfeat.COASTLINE)
    highAx.add_feature(cfeat.BORDERS, linestyle="-")
    highAx.add_feature(cfeat.LAKES, alpha=0.5)
    highAx.add_feature(cfeat.RIVERS, alpha=0.5)
    highAx.add_feature(cfeat.STATES, linestyle=":")
    highAx.set_extent((lonmin, lonmax, latmin, latmax))
    lowAx = lowFig.add_subplot(1, 1, 1, projection=ccrs.LambertCylindrical())
    lowAx.add_feature(cfeat.LAND)
    lowAx.add_feature(cfeat.OCEAN)
    lowAx.add_feature(cfeat.COASTLINE)
    lowAx.add_feature(cfeat.BORDERS, linestyle="-")
    lowAx.add_feature(cfeat.LAKES, alpha=0.5)
    lowAx.add_feature(cfeat.RIVERS, alpha=0.5)
    lowAx.add_feature(cfeat.STATES, linestyle=":")
    lowAx.set_extent((lonmin, lonmax, latmin, latmax))

    # Plot temp contour field
    avgContour = avgAx.contourf(lonGrid, latGrid, avgTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(floor(averagedData.avgD.min()), ceil(averagedData.avgD.max()), 0.5), cmap="coolwarm")
    highContour = highAx.contourf(lonGrid, latGrid, highTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(floor(averagedData.highD.min()), ceil(averagedData.highD.max()), 0.5), cmap="coolwarm")
    lowContour =lowAx.contourf(lonGrid, latGrid, lowTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(floor(averagedData.lowD.min()), ceil(averagedData.lowD.max()), 0.5), cmap="coolwarm")    
        
    # Add color bars
    avgFig.colorbar(avgContour)
    highFig.colorbar(highContour)
    lowFig.colorbar(lowContour)


    # Plot cities
    hou_lat = 29.7604
    hou_lon = -95.3698
    sa_lat = 29.4241
    sa_lon = -98.4936
    dal_lat = 32.7767
    dal_lon = -96.7970
    pas_lat = 31.7619
    pas_lon = -106.4850
    au_lat = 30.2672
    au_lon = -97.7431
    avgAx.scatter(hou_lon, hou_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    avgAx.text(hou_lon + .2, hou_lat - .4, 'Houston', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    avgAx.scatter(sa_lon, sa_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    avgAx.text(sa_lon + .2, sa_lat - .4, 'San Antonio', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    avgAx.scatter(dal_lon, dal_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    avgAx.text(dal_lon + .2, dal_lat - .4, 'Dallas', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    avgAx.scatter(pas_lon, pas_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    avgAx.text(pas_lon + .2, pas_lat - .4, 'El Paso', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    avgAx.scatter(au_lon, au_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    avgAx.text(au_lon + .2, au_lat - .4, 'Austin', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")

    highAx.scatter(hou_lon, hou_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    highAx.text(hou_lon + .2, hou_lat - .4, 'Houston', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    highAx.scatter(sa_lon, sa_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    highAx.text(sa_lon + .2, sa_lat - .4, 'San Antonio', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    highAx.scatter(dal_lon, dal_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    highAx.text(dal_lon + .2, dal_lat - .4, 'Dallas', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    highAx.scatter(pas_lon, pas_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    highAx.text(pas_lon + .2, pas_lat - .4, 'El Paso', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    highAx.scatter(au_lon, au_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    highAx.text(au_lon + .2, au_lat - .4, 'Austin', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")

    lowAx.scatter(hou_lon, hou_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    lowAx.text(hou_lon + .2, hou_lat - .4, 'Houston', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    lowAx.scatter(sa_lon, sa_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    lowAx.text(sa_lon + .2, sa_lat - .4, 'San Antonio', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    lowAx.scatter(dal_lon, dal_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    lowAx.text(dal_lon + .2, dal_lat - .4, 'Dallas', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    lowAx.scatter(pas_lon, pas_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    lowAx.text(pas_lon + .2, pas_lat - .4, 'El Paso', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")
    lowAx.scatter(au_lon, au_lat, transform=ccrs.PlateCarree(), marker=".", s=10, c="g")
    lowAx.text(au_lon + .2, au_lat - .4, 'Austin', horizontalalignment='right', transform=ccrs.PlateCarree(), c="g")

    # Save plots
    avgFig.savefig("output-avg-"+ startDate.strftime("%Y-%m-%d") +".png")
    highFig.savefig("output-high-"+ startDate.strftime("%Y-%m-%d") +".png")
    lowFig.savefig("output-low-"+ startDate.strftime("%Y-%m-%d") +".png")
    
    # Plot wind barbs
    windLonList = averagedData["lon"].values
    windLatList = averagedData["lat"].values
    uWindList = averagedData["uWind"].values
    vWindList = averagedData["vWind"].values
    for idx in range(0, len(windLatList)):
        avgAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=4, transform=ccrs.PlateCarree())
        highAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=4, transform=ccrs.PlateCarree())
        lowAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=4, transform=ccrs.PlateCarree())
    
    avgFig.savefig("output-avg-"+ startDate.strftime("%Y-%m-%d") +"-WINDS.png")
    highFig.savefig("output-high-"+ startDate.strftime("%Y-%m-%d") +"-WINDS.png")
    lowFig.savefig("output-low-"+ startDate.strftime("%Y-%m-%d") +"-WINDS.png")
    plt.close("all")

if __name__ == "__main__":
    start_exec = time.time()
    print("Generating pre-event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 1, 28, 0, 0, 0), dt.datetime(2011, 1, 31, 23, 59, 59))
    print("11% Generating during event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 2, 1, 0, 0, 0), dt.datetime(2011, 2, 4, 23, 59, 59))
    print("22% Generating post-event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 2, 5, 0, 0, 0), dt.datetime(2011, 2, 8, 23, 59, 59))
    print("33% Generating pre-event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 2, 0, 0, 0), dt.datetime(2017, 12, 5, 23, 59, 59))
    print("44% Generating during event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 6, 0, 0, 0), dt.datetime(2017, 12, 8, 23, 59, 59))
    print("55% Generating post-event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 9, 0, 0, 0), dt.datetime(2017, 12, 12, 23, 59, 59))
    print("66% Generating pre-event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 10, 0, 0, 0), dt.datetime(2021, 2, 13, 23, 59, 59))
    print("77% Generating during event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 14, 0, 0, 0), dt.datetime(2021, 2, 18, 23, 59, 59))
    print("88% Generating post-event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 19, 0, 0, 0), dt.datetime(2021, 2, 22, 23, 59, 59))
    print("Done! (Took %s seconds)" % (time.time() - start_exec))
