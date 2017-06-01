import os
import imp
lib_loc = os.path.join(os.path.expanduser('~'), 'projects/ACE-ENA-EVA/code/ena_tools.py')
ena_tools = imp.load_source('ena_tools', lib_loc)

if __name__ == '__main__':
    ena_tools.save_latest_minnis_png_s3()

