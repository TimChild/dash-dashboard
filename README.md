# dash_dashboard
Convenient python layouts and wrappers for making a Dash app for data visualization and analysis


# How to use:
Clone this repo somewhere (best to have a repo instance for every distinct app that will use it in case things are 
changed).

## Option 1
Open main project in PyCharm in a `New Window`, then open this project and choose to `Attach`. 

## Option 2
Add directory of top level 'dash_dashboard' to interpreter path or sys.path. 


## Finally:
Now from the main project, all `dash_dashboard` things are accessible like it is a package... e.g. `from dash_dashboard
import base_classes`

### Testing a single Page
Use the `TemplatePage.py` as a starting point for creating new dashboard pages. For testing a single page, pass
`layout` and `callbacks` to `test_page()` which can be found in `dash_dashboard.app`. 

### Running a Multi page app
For running a full multipage app. Import `get_app` from `dash_dashboard.app` and then run that with a list of modules 
which follow the `TemplatePage.py` format. 

e.g.  
```
from dash_dashboard.app import get_app
from path_to_pages import page1, page2, page3  # Where each page is a whole python .py file

app = get_app([page1, page2, page3])
app.run_server()  # Might want to use debug=True here
```