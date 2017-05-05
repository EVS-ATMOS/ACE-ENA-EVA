"""
=====================================
core (:mod:`ACE-ENA-EVA.code.save_domain`)
=====================================
.. currentmodule:: ACE-ENA-EVA.code

test code for saving a subset via Python/Siphon

Tools
========================
.. autosummary::
    :toctree: generated/
    pblplot
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from datetime import datetime, timedelta
from netCDF4 import num2date, Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def copy_to_local(dsin, dsout):
    #Copy dimensions
    for dname in dsin.dimensions.keys():
        the_dim = dsin.dimensions[dname]
        print(dname, len(the_dim))
        dsout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)


    # Copy variables
    for v_name in dsin.variables.keys():
        varin = dsin.variables[v_name]
        outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
        print(varin.datatype)

        # Copy variable attributes
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        outVar[:] = varin[:]


def return_gfs():
    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p5deg/Best')
    best_gfs.datasets
    best_ds = list(best_gfs.datasets.values())[0]
    best_ds.access_urls
    return NCSS(best_ds.access_urls['NetcdfSubset'])




if __name__=="__main__":
    print('Hello world')
    ncss = return_gfs()
    query = ncss.query()
    query.lonlat_box(north=43, south=35, east=-20, west=-31).time(datetime.utcnow())
    query.accept('netcdf4')
    query.variables('Planetary_Boundary_Layer_Height_surface', 'MSLP_Eta_model_reduction_msl')
    data = ncss.get_data(query)

    dsout = Dataset('/Users/scollis/test.nc', "w", format="NETCDF4")
    copy_to_local(data, dsout)
    dsout.close()
    data.close()



