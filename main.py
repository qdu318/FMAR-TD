import h5py as h5py
import numpy as np
from FMAR import FMAR
from FMAR.util.utility import get_index

if __name__ == "__main__":

    filename = "data/DATASET NAME"
    f = h5py.File(filename)
    data = f["data"][:700, 1, :, :]
    data = np.array(data)
    ori_ts = np.moveaxis(data, 0, -1)

    print("shape of data: {}".format(ori_ts.shape))
    print("This dataset have {} series, and each serie have {} time step".format(
        ori_ts.shape[0], ori_ts.shape[1]
    ))

    ts = ori_ts[..., :-1]
    label = ori_ts[..., -1]
    p = 3 # p-order
    Rs = [5,5] # tucker decomposition ranks
    k =  100
    tol = 0.001 # can ADJUST
    Us_mode = 4

    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = FMAR(ts, p, Rs, k, tol, verbose=0, Us_mode=Us_mode)
    result, _ = model.run()
    pred = result[-1]

    # print extracted forecasting result and evaluation indexes
    print("forecast result(first 10 series):\n", pred[:10])
    print("real result(first 10 series):\n", label[:10])
    print("Evaluation index: \n{}".format(get_index(pred, label)))



