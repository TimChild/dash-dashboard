import os
import sys
if (path := os.path.dirname(os.path.realpath(__file__))) not in sys.path:  # Not sure if this is necessary
    sys.path.append(path)  # This is necessary when running from outside of dash_dashboard
import component_defaults
import util
import my_dash_labs_plugins as my_plugins