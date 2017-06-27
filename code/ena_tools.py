import urllib3
from matplotlib import use
use('agg')
#fetch GFS data and plot at ENA
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
import tempfile
import boto3
from siphon.catalog import get_latest_access_url
import pygrib
import tempfile
import xarray
import os
import ftplib


def gen_s3_key(fig_datetime, pref, sfx=''):
    fmt_str = '%Y/%m/%d/' + pref + '%Y%m%d_%H%M' + sfx + '.png'
    o_str = datetime.strftime(fig_datetime, fmt_str)
    return o_str


def format_minnis(my_datetime):
    base = 'https://www-pm.larc.nasa.gov/cgi-bin/site/'
    useless_text = '/showdoc?docid=22&c=binary-download&domain=amf_azores&type=P&'
    fmt_str = 'year=%Y&month=%m&day=%d&fn=MT10V03.0.AMFAZRS.%Y%j.%H00.PX.03K.NC'
    last_str = datetime.strftime(my_datetime, fmt_str)
    fqdn = base + useless_text + last_str
    return fqdn


def fetch_minnis(url):
    localfile = tempfile.NamedTemporaryFile()
    connection_pool = urllib3.PoolManager()
    resp = connection_pool.request('GET',url )
    f = open(localfile.name, 'wb')
    f.write(resp.data)
    f.close()
    resp.release_conn()
    mydata = Dataset(localfile.name)
    return mydata

def fetch_latest_minnis():
    now = datetime.utcnow()
    try:
        url = format_minnis(now)
        data = fetch_minnis(url)
    except OSError: #get last hours
        print('Fetching last hours')
        url = format_minnis(now - timedelta(minutes=60))
        data =  fetch_minnis(url)

    return data

def ir_plot(mydata, ofile):
    lat_2d = mydata.variables['latitude'][:]
    lon_2d = mydata.variables['longitude'][:]
    temp_vals = mydata.variables['temperature_ir'][:] -273.15
    time_val = num2date(mydata.variables['time_offset'][:], mydata.variables['time_offset'].units)
    ter_lat = 38.7216
    ter_lon = -27.2206
    gra_lat = 39.0525
    gra_lon = -28.0069


    # Create a new figure
    fig = plt.figure(figsize=(15, 12))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_xticks([-23, -24, -26, -28 ], crs=ccrs.PlateCarree())
    ax.set_yticks([37,39,41], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_extent([-23., -31., 35, 43])

    # Add state boundaries to plot
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')
    _ = ax.add_feature(coast, edgecolor='black')

    #ax.add_feature(cfeature.NaturalEarthFeature, edgecolor='black', linewidth=2)

    # Contour temperature at each lat/long
    cf = ax.pcolormesh(lon_2d, lat_2d, temp_vals, transform=ccrs.PlateCarree(), zorder=0,
                     cmap='coolwarm', vmin = -55, vmax = 25)

    # Plot a colorbar to show temperature and reduce the size of it
    plt.colorbar(cf, ax=ax, fraction=0.032)


    # Make a title with the time value
    ax.set_title(u'Minnis IR Brightness Temperature (\u00b0C) for {0:%d %B %Y %H:%MZ}'.format(time_val),
                 fontsize=20)


    ax.plot([ter_lon, gra_lon], [ter_lat, gra_lat],
           'ro', transform=ccrs.PlateCarree())

    ax.text(ter_lon+.2, ter_lat+.2,
            'Tericia', transform=ccrs.PlateCarree(), fontsize = 16)

    ax.text(gra_lon+.2, gra_lat+.2,
            'Graciosa', transform=ccrs.PlateCarree(), fontsize = 16)

    plt.savefig(ofile)
    plt.close(fig)
    return time_val


def save_latest_minnis_png_s3():
    my_data = fetch_latest_minnis()
    local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    fn = local_fig.name
    fig_datetime = ir_plot(my_data, fn)
    my_data.close()
    s3_key = 'minnis_ir/' + gen_s3_key(fig_datetime, 'minnis_ir_')
    s3 = boto3.resource('s3')
    data = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key=s3_key, Body=data, ACL='public-read')
    data.close()
    data2 = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key='latest_minnis_ir.png', Body=data2, ACL='public-read')
    data2.close()
    return s3_key, fn

def give_me_latest_gfs():
    best_gfs = 'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/catalog.xml'
    latest_gfs = get_latest_access_url(best_gfs, "NetcdfSubset")
    ncss = NCSS(latest_gfs)
    return ncss

def get_me_grw_ts(ncss, varss):
    query = ncss.query()
    now = datetime.utcnow()
    gra_lat = 39.0525
    gra_lon = -28.0069
    query.lonlat_point(gra_lon, gra_lat).vertical_level(0).all_times()
    query.variables(*varss).accept('netcdf')
    data = ncss.get_data(query)
    return data


def plot_ttd_gra(my_ts, ofile):
    temp = my_ts.variables['Temperature_height_above_ground']
    dp = my_ts.variables['Dewpoint_temperature_height_above_ground']
    time = my_ts.variables['time']
    datetimes = num2date(time[:].squeeze(), time.units)
    fig = plt.figure(figsize = [15,5])
    plt.plot(datetimes, temp[:].squeeze()-273.15, 'r-', label='Temperatue (c)')
    plt.plot(datetimes, dp[:].squeeze()-273.15, 'b-', label=r't$_{dp}$ (c)')
    plt.legend()
    plt.title('Time Series for Graciosa')
    plt.savefig(ofile)
    plt.close(fig)
    return datetimes[0]


def save_latest_gfs_grw_ttd():
    my_ncss = give_me_latest_gfs()
    my_ts = get_me_grw_ts(my_ncss, ['Temperature_height_above_ground',
               'Dewpoint_temperature_height_above_ground'])
    local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    fn = local_fig.name
    fig_datetime = plot_ttd_gra(my_ts, fn)
    s3_key = 'time_series_gfs/' + gen_s3_key(fig_datetime, 'T_ts_timeseries_GFS_GRW')
    s3 = boto3.resource('s3')
    data = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key=s3_key, Body=data, ACL='public-read')
    data.close()
    data2 = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key='latest_T_ts_timeseries_GFS_GRW.png',
            Body=data2, ACL='public-read')
    data2.close()

def get_sfc_gfs(ncss, varrs, bbox):
    north = bbox[3]
    south = bbox[2]
    west = bbox[1]
    east = bbox[0]
    query = ncss.query()
    query.lonlat_box(north=north, south=south,
                     east=east, west=west).vertical_level(0).all_times()
    query.accept('netcdf4')
    query.variables(*varrs)
    data = ncss.get_data(query)
    return data

def add_panel(data_set,time_step,
              times, fig, ax, bbox,
             background_var = 'Temperature_surface',
             oset = -273.15, scale = 1.,
             pref = u'SFC T forecast (\u00b0C)',
             vmin = 15, vmax = 25):
    gra_lat = 39.0525
    gra_lon = -28.0069
    ter_lat = 38.7216
    ter_lon = -27.2206
    north = bbox[3]
    south = bbox[2]
    west = bbox[1]
    east = bbox[0]
    lat_vals = data_set.variables['lat'][:].squeeze()
    lon_vals = data_set.variables['lon'][:].squeeze()
    lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)

    # Add the map and set the extent
    ax.set_xticks(np.arange(west, east, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(south, north, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    print(bbox)


    # Add state boundaries to plot
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')
    _ = ax.add_feature(coast, edgecolor='black')


    # Contour temperature at each lat/long
    cf = ax.pcolormesh(lon_2d, lat_2d, scale * data_set[background_var][time_step] + oset,
                                       transform=ccrs.PlateCarree(),
                                       zorder=0,
                                       cmap='coolwarm', vmin=vmin, vmax=vmax)

    cn = ax.contour(lon_2d, lat_2d,
                    data_set['MSLP_Eta_model_reduction_msl'][time_step]/100.0,
                    np.arange(980, 1040, 2), colors='black')

    plt.clabel(cn, inline=1, fontsize=10, fmt='%1.0f')
    ax.set_extent(bbox)


    # Plot a colorbar to show temperature and reduce the size of it
    plt.colorbar(cf, ax=ax, fraction=0.032)

    # Make a title with the time value
    time_val = times[time_step]
    ax.set_title(pref + ' for {0:%d %B %Y %H:%MZ}'.format(time_val),
                 fontsize=10)


    ax.plot([ter_lon, gra_lon], [ter_lat, gra_lat],
           'ro', transform=ccrs.PlateCarree())

    ax.text(ter_lon+.2, ter_lat+.2,
            'Tericia', transform=ccrs.PlateCarree(), fontsize = 10)

    ax.text(gra_lon+.2, gra_lat+.2,
            'Graciosa', transform=ccrs.PlateCarree(), fontsize = 10)



def nine_panel(my_data, bbox, time_steps, bgv = 'Temperature_surface',
             oset = -273.15, scale = 1.,
             pref = u'SFC T forecast (\u00b0C)',
             vmin = 15, vmax = 25):
    f, ((ax1, ax2, ax3),
        (ax4, ax5, ax6),
        (ax7, ax8, ax9)) = plt.subplots(3,3, figsize = [15,10],
                                        sharex=True, sharey=True,
                                        subplot_kw={'projection': ccrs.PlateCarree()})

    time = my_data.variables['time1']
    datetimes = num2date(time[:].squeeze(), time.units)
    axxx = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    for i in range(len(axxx)):
        add_panel(my_data,time_steps[i], datetimes, f, axxx[i], bbox,
                 background_var=bgv,
                 vmin=vmin, vmax=vmax, oset=oset, pref=pref, scale=scale)
    return datetimes[0]

def save_one_9pan(s3name, figname, fig_datetime):
    s3_key = s3name + '/' + gen_s3_key(fig_datetime, s3name)
    s3 = boto3.resource('s3')
    data = open(figname, 'rb')
    s3.Bucket('aceena').put_object(Key=s3_key, Body=data, ACL='public-read')
    data.close()
    data2 = open(figname, 'rb')
    s3.Bucket('aceena').put_object(Key='latest_' + s3name + '.png',
                                   Body=data2, ACL='public-read')
    data2.close()


def nine_panel_s3():
    bbox = [-5, -35, 30, 50]
    varrs = ['Temperature_surface',
           'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
            'Planetary_Boundary_Layer_Height_surface',
            'Precipitation_rate_surface_Mixed_intervals_Average']
    names = ['temp_nine_panel_gfs', 'BL_clouds_nine_panel_gfs', 'PBL_H', 'precip']
    mins = [15, 0, 0, 0, 0]
    maxs = [25, 100, 2000, 5]
    osets = [-273.15, 0, 0, 0]
    scales = [1. , 1., 1., 60.*60.]
    prefs = [ u'SFC T forecast (\u00b0C)', 'BL cloud frac', 'PBL Ht', 'Prec rate (mm/h)']
    varrs.append('MSLP_Eta_model_reduction_msl')
    print(varrs)
    my_ncss = give_me_latest_gfs()
    my_data = get_sfc_gfs(my_ncss, varrs, bbox)
    time_steps = [0,2,4,6,8,10,12,14,16]
    local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    fn = local_fig.name
    for i in range(len(names)):
        fig_datetime = nine_panel(my_data, bbox, time_steps,
                                        bgv=varrs[i],
                                        vmin=mins[i], vmax=maxs[i], oset=osets[i],
                                        scale=scales[i], pref=prefs[i])
        plt.savefig(fn)
        save_one_9pan(names[i], fn, fig_datetime)

def get_me_grw_ths(ncss, varss):
    query = ncss.query()
    now = datetime.utcnow()
    gra_lat = 39.0525
    gra_lon = -28.0069
    query.lonlat_point(gra_lon, gra_lat).all_times()
    query.variables(*varss).accept('netcdf')
    data = ncss.get_data(query)
    return data

def ths(data_set, vert_var_bg = 'isobaric', vert_var_c = 'isobaric',
        pref_bg = u'T forecast (\u00b0C)',
        pref_c = 'Humdity (\%)',
             background_var = 'Temperature_isobaric',
              cont_var = 'Relative_humidity_isobaric',
             oset_1 = -273.15, scale_1 = 1.,
             vmin_1 = -80, vmax_1 = 25,
       clevs = [50,90,99], colors = ['r', 'b', 'k'],
       fmt = '%1.0f', con_size = 10):
    fig = plt.figure(figsize = [15,5])
    vert_bg = data_set.variables[vert_var_bg][:].squeeze()[0,:]/100
    vert_c = data_set.variables[vert_var_c][:].squeeze()[0,:]/100
    time = data_set.variables['time']
    datetimes = num2date(time[:].squeeze(), time.units)
    mapb = plt.pcolormesh(datetimes, vert_bg,
                   scale_1 * data_set.variables[background_var][:].squeeze().transpose() + oset_1,
                         vmin = vmin_1,
                         vmax = vmax_1)
    con = plt.contour(datetimes, vert_c,
               data_set.variables[cont_var][:].squeeze().transpose(),
               levels = clevs, colors = colors)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Pressure Level (hPa)')
    plt.clabel(con, inline=1, fontsize=con_size, fmt=fmt)
    plt.ylim([1000,0])
    time_val = datetimes[0]
    pref = pref_bg + ' and ' + pref_c
    plt.gca().set_title(pref + ' for {0:%d %B %Y %H:%MZ}'.format(time_val),
                 fontsize=10)
    plt.colorbar(mappable = mapb)
    return datetimes[0]

def th_plots():
    my_ncss = give_me_latest_gfs()
    my_var = ['Temperature_isobaric', 'Relative_humidity_isobaric',
              'Vertical_velocity_pressure_isobaric',
             'Cloud_mixing_ratio_isobaric']
    my_data = get_me_grw_ths(my_ncss, my_var)
    fig_datetime = ths(my_data, vert_var_bg = 'isobaric3',
                                 background_var = 'Vertical_velocity_pressure_isobaric',
                                 oset_1 = 0., scale_1 = 1.0,
                                 vmin_1 = -0.5, vmax_1 = 1,
                                pref_bg = u'Omega (hPa/s)')
    local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    fn = local_fig.name
    plt.savefig(fn)
    save_one_9pan('time_height_omega_rh', fn, fig_datetime)

    fig_datetime = ths(my_data, vert_var_bg = 'isobaric3',
                                 background_var = 'Cloud_mixing_ratio_isobaric',
                                 oset_1 = 0., scale_1 = 1000.0,
                                 vmin_1 = 0, vmax_1 = 0.5,
                                pref_bg = u'Omega (hPa/s)')
    #local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    #fn = local_fig.name
    plt.savefig(fn)
    save_one_9pan('time_height_cld_rh', fn, fig_datetime)

def get_ecmwf_137():
    levels_loc = os.path.join(os.path.expanduser('~'), 'projects/ACE-ENA-EVA/ecmwf_137_levels.txt')
    levels = np.genfromtxt(levels_loc, missing_values='-')
    ht = levels[1::, 6]
    pres = levels[1::, 3]
    return ht, pres

def extract_3d_grib(grb_obj, search_term, skip_last=False):
    grb_list = grb_obj.select(name=search_term)
    level_nums = [this_grb['level'] for this_grb in grb_list]
    order =  np.array(level_nums).argsort()
    lats, lons = grb_list[0].latlons()
    shp = grb_list[order[0]].values.shape
    transfer_array = np.empty([len(order), shp[0], shp[1]])
    if skip_last:
        llen = len(order) -1
    else:
        llen = len(order)

    for i in range(llen):
        transfer_array[i,:,:] = grb_list[order[i]].values

    return transfer_array, lats, lons

def ecmwf_name_to_date(ename):
    start_time = datetime.strptime('2017'+ename[3:11], '%Y%m%d%H%M')
    valid_time = datetime.strptime('2017'+ename[11:19], '%Y%m%d%H%M')
    return start_time, valid_time

def file_list_to_date(file_list):
    start_times = [ecmwf_name_to_date(this_name)[0] for this_name in file_list]
    end_times = [ecmwf_name_to_date(this_name)[1] for this_name in file_list]
    return start_times, end_times

def get_run_hours(file_list):
    gen_t, val_t = file_list_to_date(file_list)
    gen_hour = np.array([dt.hour for dt in gen_t])
    return np.unique(gen_hour)

def get_time_for_run(file_list, gen_hour):
    gen_t, val_t = file_list_to_date(file_list)
    gen_hours = np.array([dt.hour for dt in gen_t])
    good_files = []
    good_times_gen = []
    good_times_val = []
    for i in range(len(gen_hours)):
        if gen_hours[i] == gen_hour:
            good_files.append(file_list[i])
            good_times_gen.append(gen_t[i])
            good_times_val.append(val_t[i])

    return good_files, good_times_gen, good_times_val

def create_bundle_latest(var_list, n=None, skippy = None):
    username_file = os.path.join(os.path.expanduser('~'), 'ecmwf_username')
    password_file = os.path.join(os.path.expanduser('~'), 'ecmwf_passwd')
    uname_fh = open(username_file, 'r')
    uname = uname_fh.readline()[0:-1]
    uname_fh.close()
    passwd_fh = open(password_file, 'r')
    passwd = passwd_fh.readline()[0:-1]
    passwd_fh.close()
    host = 'dissemination.ecmwf.int'

    #get ECMWF vert coord
    ht, pres = get_ecmwf_137()
    ftp = ftplib.FTP(host)
    ftp.login(user=uname, passwd = passwd )
    closest_now = datetime.utcnow().strftime('%Y%m%d')
    ftp.cwd(closest_now)
    lst = ftp.nlst()
    lst.sort()

    run_hours = get_run_hours(lst)
    target_files, generated_times, valid_times = get_time_for_run(lst, run_hours[-1])
    if skippy is None:
        skippy = len(var_list)*[False]

    if n is None:
        these_target_files = target_files[1::]
        these_run_times = generated_times[1::]
        these_valid_times = valid_times[1::]
    else:
        these_target_files = target_files[1:n]
        these_run_times = generated_times[1:n]
        these_valid_times = valid_times[1:n]

    bundle = {}

    for var_name in var_list:
        bundle.update({var_name: []})

    for i in range(len(these_valid_times)):
        print(these_target_files[i])
        fh = tempfile.NamedTemporaryFile()
        ftp.retrbinary('RETR ' + these_target_files[i], fh.write)
        grbs = pygrib.open(fh.name)
        grbs.seek(0)
        for i in range(len(var_list)):
            var_name = var_list[i]
            this_step, lats, lons = extract_3d_grib(grbs, var_name, skip_last = skippy[i])
            bundle[var_name].append(this_step)
        grbs.close()

    return bundle, these_valid_times, these_run_times, lats, lons


def concat_bundle(bundle):
    varss = list(bundle.keys())
    n_times = len(bundle[varss[0]])
    hlatlon = bundle[varss[0]][0].shape
    unwound = {}
    for this_var in varss:
        transfer = np.empty([n_times, 136, hlatlon[1], hlatlon[2]])
        for time_step in range(n_times):
            transfer[time_step, 0:136, :, :] = bundle[this_var][time_step][0:136, :, :]
        unwound.update({this_var: transfer})
    return unwound

def unwind_to_xarray(unwound, valid_times, lats, lons):
    ds = xarray.Dataset()
    for vvar in list(unwound.keys()):
        my_data = xarray.DataArray(unwound[vvar],
                                   dims = ('time', 'z', 'y', 'x'),
                                  coords = {'time' : (['time'], valid_times),
                                           'z' : (['z'], get_ecmwf_137()[0][0:136]),
                                           'lat' :(['y','x'], lats),
                                           'lon' : (['y','x'],lons)})
        ds[vvar.replace(' ', '_')] = my_data

    ds.lon.attrs = [('long_name', 'longitude of grid cell center'),
             ('units', 'degrees_east'),
             ('bounds', 'xv')]
    ds.lat.attrs = [('long_name', 'latitude of grid cell center'),
             ('units', 'degrees_north'),
             ('bounds', 'yv')]
    return ds

def save_one_ecmwf_clouds(dataset, gen_datetime):
    start_str = gen_datetime.strftime('%Y%m%d_%H%M')
    s3name = 'ecmwf_time_height_clwc_ciwc'
    plt.figure(figsize=(17,5))
    my_levels = [0.01, 0.05, 0.09, 0.13]
    my_colors = ['white', 'yellow', 'cyan', 'pink']
    (dataset.Specific_cloud_liquid_water_content*1000.0).mean(dim=('y','x')).plot.pcolormesh(x='time', y='z')
    cs = (dataset.Specific_cloud_ice_water_content*1000.0).max(dim=('y','x')).plot.contour(x='time', y='z',
                                                                                         levels=my_levels,
                                                                                         colors=my_colors)
    plt.clabel(cs, inline=1, fontsize=10, fmt='%0.3f')
    plt.ylim([0,12000])
    str1 = start_str + ' ECMWF Domain max cloud ice and domain mean cloud liquid water content (g/kg) \n'
    str2 = 'ACE-ENA forecast guidence. ARM Climate Research Facility. scollis@anl.gov'
    plt.title(str1+str2)
    local_fig =  tempfile.NamedTemporaryFile(suffix='.png')
    fn = local_fig.name
    plt.savefig(fn)

    s3_key = s3name + '/' + gen_s3_key(gen_datetime, s3name)
    s3 = boto3.resource('s3')
    data = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key=s3_key, Body=data, ACL='public-read')
    data.close()
    data2 = open(fn, 'rb')
    s3.Bucket('aceena').put_object(Key='latest_' + s3name + '.png',
                                   Body=data2, ACL='public-read')
    data2.close()

