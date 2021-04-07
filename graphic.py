#!/usr/bin/env python3
# Graphic generation for ATMO 324 by Sam Gardner <stgardner4@tamu.edu>, Drew Davis <acdavis01@tamu.edu>, Ashley Palm <ashleyp0301@tamu.edu>

import csv
import datetime as dt
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from cartopy import crs as ccrs
from cartopy import feature as cfeat
import numpy as np
import time

def generatePlot(filename, startDate, endDate):
    # Read csv data
    tempValidReports = list()
    windValidReports = list()
    targetDates = [startDate + dt.timedelta(days=x) for x in range((endDate - startDate).days + 1)]

    with open(filename) as asosCSVFile:
        csvRead = csv.DictReader(asosCSVFile, delimiter=",")
        for row in csvRead:
            if (row["tmpc"] != "null"):
                tempValidReports.append(row)
            if (row["drct"] != "null"):
                if (row["sknt"] != "null"):
                    windValidReports.append(row)
    # make dates in csv use datetime objects
    tempReportsOnTargetDates = list()
    tempsOnTargetDates = list()
    windReportsOnTargetDates = list()
    windDirsOnTargetDates = list()
    windKtsOnTargetDates = list()
    for targetDT in targetDates:
        targetDateStr = targetDT.strftime("%Y-%m-%d")
        for report in tempValidReports:
            if targetDateStr in report["valid"]:
                tempsOnTargetDates.append(float(report["tmpc"]))
                tempReportsOnTargetDates.append(report)
        for report in windValidReports:
            if targetDateStr in report["valid"]:
                windDirsOnTargetDates.append(float(report["drct"]))
                windKtsOnTargetDates.append(float(report["sknt"]))
                windReportsOnTargetDates.append(report)

    # Get overall average temperature during event
    averageTempAllStations = sum(tempsOnTargetDates) / len(tempsOnTargetDates)

    # Create list of temp dicts for all reports we're interested in
    deviationRpts = list()
    highRpts = dict()
    lowRpts = dict()
    tempDictOfStations = dict()
    for targetedReport in tempReportsOnTargetDates:
        deviationReport = { "Station ID" : targetedReport["station"],
                        "Date/Time" : targetedReport["valid"],
                        "Latitude" : targetedReport["lat"],
                        "Longitude" : targetedReport["lon"],
                        "Temperature" : float(targetedReport["tmpc"]),
                        "Temp deviation" : float(float(targetedReport["tmpc"]) - averageTempAllStations)
                        }
        deviationRpts.append(deviationReport)
        if targetedReport["station"] in highRpts.keys():
            if targetedReport["tmpc"] > highRpts[targetedReport["station"]]["tmpc"]:
                highRpts[targetedReport["station"]] = targetedReport
        else:
            highRpts[targetedReport["station"]] = targetedReport
        if targetedReport["station"] in lowRpts.keys():
            if targetedReport["tmpc"] < lowRpts[targetedReport["station"]]["tmpc"]:
                lowRpts[targetedReport["station"]] = targetedReport
        else:
            lowRpts[targetedReport["station"]] = targetedReport
            
        if targetedReport["station"] not in tempDictOfStations.keys():
            stationDict = { "lat" : targetedReport["lat"], 
                        "lon" : targetedReport["lon"] }
            tempDictOfStations[targetedReport["station"]] = stationDict
    # Create list of wind dicts for all reports we're interested in
    windRpts = list()
    windDictOfStations = dict()
    for targetedReport in windReportsOnTargetDates:
        windReport = { "Station ID" : targetedReport["station"],
                        "Date/Time" : targetedReport["valid"],
                        "Latitude" : targetedReport["lat"],
                        "Longitude" : targetedReport["lon"],
                        "u wind" : 
                        (float(targetedReport["sknt"])*-1*np.sin(np.deg2rad(float(targetedReport["drct"])))), 
                        "v wind" : 
                        (float(targetedReport["sknt"])*-1*np.cos(np.deg2rad(float(targetedReport["drct"]))))
                        }
        windRpts.append(windReport)
        
        if targetedReport["station"] not in windDictOfStations.keys():
            stationDict = { "lat" : targetedReport["lat"], 
                        "lon" : targetedReport["lon"] }
            windDictOfStations[targetedReport["station"]] = stationDict
            

    # Get average deviation for each station across the event 
    # maybe don't do this in the future
    # ...could use an animated GIF instead of single frame to show change over time
    tempStationList = tempDictOfStations.keys()
    tempLatList = list()
    tempLonList = list()
    deviationList = list()
    for station in tempStationList:
        tempLatList.append(tempDictOfStations[station]["lat"])
        tempLonList.append(tempDictOfStations[station]["lon"])
        rollingSum = 0.0
        numOfRpts = 0
        for targetReport in deviationRpts:
            if targetReport["Station ID"] == station:
                rollingSum = rollingSum + targetReport["Temp deviation"]
                numOfRpts = numOfRpts + 1
        deviationAvg = (rollingSum / numOfRpts)
        deviationList.append(deviationAvg)
    highLatList = list()
    highLonList = list()
    highDevList = list()
    # Get average high temperature
    rollingHigh = 0.0
    numHighRpts = 0
    for station in highRpts.keys():
        rollingHigh = rollingHigh + float(highRpts[station]["tmpc"])
        numHighRpts = numHighRpts + 1
    highAvg = (rollingHigh / numHighRpts)
    for station in highRpts.keys():
        highLatList.append(highRpts[station]["lat"])
        highLonList.append(highRpts[station]["lon"])
        highDevList.append(float(highRpts[station]["tmpc"]) - highAvg)
    lowLatList = list()
    lowLonList = list()
    lowDevList = list()
    rollingLow = 0.0
    numLowRpts = 0
    for station in lowRpts.keys():
        rollingLow = rollingLow + float(lowRpts[station]["tmpc"])
        numLowRpts = numLowRpts + 1
    lowAvg = (rollingLow / numLowRpts)
    for station in lowRpts.keys():
        lowLatList.append(lowRpts[station]["lat"])
        lowLonList.append(lowRpts[station]["lon"])
        lowDevList.append(float(lowRpts[station]["tmpc"]) - lowAvg)
    # Get average wind for each station across the event
    windStationList = windDictOfStations.keys()
    windLatList = list()
    windLonList = list()
    uWindList = list()
    vWindList = list()
    for station in windStationList:
        windLatList.append(float(windDictOfStations[station]["lat"]))
        windLonList.append(float(windDictOfStations[station]["lon"]))
        rollingSumU = 0.0
        rollingSumV = 0.0
        numOfRpts = 0
        for targetReport in windRpts:
            if targetReport["Station ID"] == station:
                rollingSumU = rollingSumU + targetReport["u wind"]
                rollingSumV = rollingSumU + targetReport["v wind"]
                numOfRpts = numOfRpts + 1
        uWindAvg = (rollingSumU / numOfRpts)
        uWindList.append(uWindAvg)
        vWindAvg = (rollingSumV / numOfRpts)
        vWindList.append(vWindAvg)

    ## interpolate individual stations into mesh covering the whole area
    # get min/max bounds
    latmin = 25.837377
    latmax = 36.600704
    lonmin = -106.745646
    lonmax = -93.508292
    # create mesh
    lonGrid = np.linspace(lonmin, lonmax, 1000)
    latGrid = np.linspace(latmin, latmax, 1000)
    lonGrid, latGrid = np.meshgrid(lonGrid, latGrid)
    # get interpolated data
    avgTempGrid = griddata((tempLonList, tempLatList), deviationList, (lonGrid, latGrid), method="linear")
    highTempGrid = griddata((highLonList, highLatList), highDevList, (lonGrid, latGrid), method="linear")
    lowTempGrid = griddata((lowLonList, lowLatList), lowDevList, (lonGrid, latGrid), method="linear")


    # Create plot figure
    avgFig = plt.figure()
    plt.title("Average T deviations from statewide avg. T\n" + targetDates[0].strftime("%Y-%m-%d") + " through " + targetDates[-1].strftime("%Y-%m-%d"))
    highFig = plt.figure()
    plt.title("High T deviations from statewide avg. high T\n" + targetDates[0].strftime("%Y-%m-%d") + " through " + targetDates[-1].strftime("%Y-%m-%d"))
    lowFig = plt.figure()
    plt.title("Low T deviations from statewide avg. low T\n" + targetDates[0].strftime("%Y-%m-%d") + " through " + targetDates[-1].strftime("%Y-%m-%d"))
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
    avgAx.contourf(lonGrid, latGrid, avgTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(-15, 10, 1), cmap="coolwarm")
    highAx.contourf(lonGrid, latGrid, highTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(-15, 10, 1), cmap="coolwarm")
    lowAx.contourf(lonGrid, latGrid, lowTempGrid, transform=ccrs.PlateCarree(), levels=np.arange(-15, 10, 1), cmap="coolwarm")    
        
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
    avgFig.savefig("output-avg-"+ targetDates[0].strftime("%Y-%m-%d") +".png")
    highFig.savefig("output-high-"+ targetDates[0].strftime("%Y-%m-%d") +".png")
    lowFig.savefig("output-low-"+ targetDates[0].strftime("%Y-%m-%d") +".png")
    
    # Plot wind barbs
    for idx in range(0, len(windLatList)):
        avgAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=5, transform=ccrs.PlateCarree())
        highAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=5, transform=ccrs.PlateCarree())
        lowAx.barbs(np.array([windLonList[idx]]), np.array([windLatList[idx]]), np.array([uWindList[idx]]), np.array([vWindList[idx]]), length=5, transform=ccrs.PlateCarree())
        
    avgFig.savefig("output-avg-"+ targetDates[0].strftime("%Y-%m-%d") +"-WINDS.png")
    highFig.savefig("output-high-"+ targetDates[0].strftime("%Y-%m-%d") +"-WINDS.png")
    lowFig.savefig("output-low-"+ targetDates[0].strftime("%Y-%m-%d") +"-WINDS.png")
    plt.close("all")


if __name__ == "__main__":
    start_exec = time.time()
    print("Generating pre-event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 1, 28, 0, 0, 0), dt.datetime(2011, 1, 31, 0, 0, 0))
    print("11% Generating during event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 2, 1, 0, 0, 0), dt.datetime(2011, 2, 4, 0, 0, 0))
    print("22% Generating post-event 2011...")
    generatePlot("asos-2011-full.csv", dt.datetime(2011, 2, 5, 0, 0, 0), dt.datetime(2011, 2, 8, 0, 0, 0))
    print("33% Generating pre-event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 2, 0, 0, 0), dt.datetime(2017, 12, 5, 0, 0, 0))
    print("44% Generating during event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 6, 0, 0, 0), dt.datetime(2017, 12, 8, 0, 0, 0))
    print("55% Generating post-event 2017...")
    generatePlot("asos-2017-full.csv", dt.datetime(2017, 12, 9, 0, 0, 0), dt.datetime(2017, 12, 12, 0, 0, 0))
    print("66% Generating pre-event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 10, 0, 0, 0), dt.datetime(2021, 2, 13, 0, 0, 0))
    print("77% Generating during event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 14, 0, 0, 0), dt.datetime(2021, 2, 18, 0, 0, 0))
    print("88% Generating post-event 2021...")
    generatePlot("asos-2021-full.csv", dt.datetime(2021, 2, 19, 0, 0, 0), dt.datetime(2021, 2, 22, 0, 0, 0))
    print("Done! (Took %s seconds)" % (time.time() - start_exec))

    
    
