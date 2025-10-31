import pickle
import numpy as np

# from analyzePCR import pcrData_varThresh


def getDataFromPickleFN(dataFN: str):
    """
    This function parses the pickled file containing the PCR data.
    The pickled file is an object of class pcrData_varThresh which has the data organized into attributes

    :param dataFN:
    :return:
        bias - list of unique bias currents in same order as data was collected
        counts - The (light-dark) counts using the optimal thresholds previously determined. List corresponds to the bias current
        counts_err - poisson error on counts (propagated through the light counts and dark counts)
        dark counts - The (dark) counts using the optimal thresholds previously determined. List corresponds to the bias current
        dark counts_err - poisson error on dark counts
        times -
    """
    data = pickle.load(open(dataFN, "rb"))
    _, x_inds = np.unique(data.biasCurrent, return_index=True)
    x = data.biasCurrent[np.sort(x_inds)]  # Keep same order as we collected the data
    counts_optimal = np.zeros(len(x))
    counts_optimal_expTime = np.zeros(len(x))
    darkCounts_optimal = np.zeros(len(x))
    times = np.zeros(len(x))

    def findOptimalCounts(i):
        # nonlocal data, x, counts_optimal, counts_optimal_expTime, darkCounts_optimal
        nonlocal counts_optimal, counts_optimal_expTime, darkCounts_optimal, times
        mask_optimal = (
            (data.biasCurrent == x[i])
            * (data.thresh >= data.bestThreshes[i, 0])
            * (data.thresh <= data.bestThreshes[i, 1])
        )
        counts_optimal[i] = np.sum(data.light_counts[mask_optimal])
        counts_optimal_expTime[i] = (
            np.sum(mask_optimal) * data.expTime * data.light_counts.shape[2]
        )
        darkCounts_optimal[i] = np.sum(data.dark_counts[mask_optimal])
        try:
            times[i] = np.mean(data.times[mask_optimal])
        except AttributeError:
            pass

    for i in range(len(x)):
        findOptimalCounts(i)

    # maxDCR = 2.E3
    expTimeRatio = data.light_counts.shape[-1] / data.dark_counts.shape[-1]

    # plotPCR(x, counts_optimal, counts_optimal_expTime, darkCounts_optimal,
    #        expTimeRatio=data.light_counts.shape[-1] / data.dark_counts.shape[-1])

    y = (counts_optimal - darkCounts_optimal * expTimeRatio) / counts_optimal_expTime
    yerr = (
        np.sqrt(counts_optimal + darkCounts_optimal * expTimeRatio**2)
        / counts_optimal_expTime
    )
    dark = darkCounts_optimal * expTimeRatio / counts_optimal_expTime
    dark_err = np.sqrt(darkCounts_optimal) * expTimeRatio / counts_optimal_expTime

    # Make bias current positive and in units of uA
    if np.all(x <= 10**-9):
        x = np.abs(x)
    # inds=np.argsort(x)
    x *= 10**6

    return np.transpose([x, y, yerr, dark, dark_err, times])
    # return np.transpose([x[inds], y[inds], yerr[inds], dark[inds], dark_err[inds]])
