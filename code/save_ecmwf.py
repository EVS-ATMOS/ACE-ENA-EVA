import os
import imp
lib_loc = os.path.join(os.path.expanduser('~'), 'projects/ACE-ENA-EVA/code/ena_tools.py')
ena_tools = imp.load_source('ena_tools', lib_loc)

if __name__ == '__main__':
    var_list = ['Specific cloud liquid water content', 'Specific cloud ice water content',
           'Specific rain water content', 'Specific snow water content',
           'Temperature', 'V component of wind', 'U component of wind', 'Specific humidity', 'Vertical velocity']
    skip_me = [False, False, False, False, False, True, True, False, False]
    tpl = ena_tools.create_bundle_latest(var_list, skippy=skip_me)
    my_bundle, my_these_valid_times, my_these_run_times, my_lats, my_lons, metad = tpl
    my_unwind = ena_tools.concat_bundle(my_bundle)
    trans = {'cfName': 'standard_name'}
    my_dataset = ena_tools.unwind_to_xarray(my_unwind, my_these_valid_times, my_lats, my_lons, metad, trans=trans)
    my_dataset.attrs['Conventions'] = 'CF-1.6'
    my_dataset.attrs['source'] = 'ECMWF 137 level 0.1 degree model'
    my_dataset.attrs['conatact'] = 'Scott Collis, scollis@anl.gov'
    st1 = "European Center for Medium range Weather Forecasting, "
    st2 = "Atmospheric Climate Research Facility, A DoE User facility, "
    st3 = "Scott Collis, Argonne National Laboratory"
    my_dataset.attrs['attribution'] = st1 + st2 + st3
    my_dataset.attrs['experiment'] ='ACE-ENA'
    my_dataset.encoding['unlimited_dims'] = ['time']
    my_dataset.z.encoding['_FillValue'] = -9999
    my_dataset.lat.encoding['_FillValue'] = -9999
    my_dataset.lon.encoding['_FillValue'] = -9999

    my_dataset.Specific_cloud_ice_water_content.attrs['standard_name'] = 'cloud_ice_mixing_ratio'
    my_dataset.Specific_cloud_liquid_water_content.attrs['standard_name'] = 'cloud_liquid_water_mixing_ratio'
    my_dataset.Specific_rain_water_content.attrs['standard_name'] = 'rain_water_content'
    my_dataset.Specific_snow_water_content.attrs['standard_name'] = 'snow_water_content'

    pressure_4d = ena_tools.get_pres4d(my_dataset.Specific_humidity)
    my_dataset['Relative_humidity'] = ena_tools.calc_rh(my_dataset.Specific_humidity,
                                                    pressure_4d, my_dataset.Temperature)
    vap_pres = ena_tools.calc_vap_pressure(pressure_4d, my_dataset.Specific_humidity)
    dewpoint = ena_tools.calc_dewpoint(vap_pres)

    ena_tools.save_one_ecmwf_clouds(my_dataset, my_these_run_times[0])
    ena_tools.save_one_ecmwf_cloud9(my_dataset, my_these_run_times[0])
    ena_tools.save_skews_s3(my_dataset.Temperature, dewpoint, pressure_4d,
          my_dataset.U_component_of_wind, my_dataset.V_component_of_wind,
          my_dataset.time, gen_datetime= my_these_run_times[0])


    sstting = '/lcrc/group/earthscience/ecmwf/%Y%m%d/'
    fst = 'ecmwf_%Y%m%d_%H%M.nc'
    local_dir = my_these_run_times[0].strftime(sstting)
    local_file = my_these_run_times[0].strftime(fst)
    print(local_file)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    my_dataset.to_netcdf(local_dir + local_file)


