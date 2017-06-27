import os
import imp
lib_loc = os.path.join(os.path.expanduser('~'), 'projects/ACE-ENA-EVA/code/ena_tools.py')
ena_tools = imp.load_source('ena_tools', lib_loc)

if __name__ == '__main__':
    var_list_1 = ['Specific cloud liquid water content', 'Specific cloud ice water content',
           'Specific rain water content', 'Specific snow water content',
           'Temperature', 'V component of wind', 'U component of wind', 'Specific humidity', 'Vertical velocity']
    skip_me = [False, False, False, False, False, True, True, False, False]
    tpl = ena_tools.create_bundle_latest(var_list, skippy=skip_me)
    my_bundle, my_these_valid_times, my_these_run_times, my_lats, my_lons = tpl
    my_unwind = ena_tools.concat_bundle(my_bundle)
    my_dataset = ena_tools.unwind_to_xarray(my_unwind, my_these_valid_times, my_lats, my_lons)
    save_one_ecmwf_clouds(my_dataset, my_these_run_times[0])
    sstting = '/lcrc/group/earthscience/ecmwf/%Y%m%d/'
    fst = 'ecmwf_%Y%m%d_%H%M.nc'
    local_dir = my_these_run_times[0].strftime(sstting)
    local_file = my_these_run_times[0].strftime(fst)
    print(local_file)
    if not os.path.exists(local_dir):
        os.mkdirs(local_dir)
    my_dataset.to_netcdf(local_dir+)


