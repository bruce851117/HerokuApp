from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, Spacer
from bokeh.models import ColumnDataSource, Slider, Select,Panel,RangeSlider,Tabs, TableColumn,\
    DataTable,Arrow,OpenHead,Div, HoverTool
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import json
from value_transform import value_transfrom
from app_matrix import Persistence_EM_Matrix
from app_map import Market_map
matrix_tab = Persistence_EM_Matrix()
map_tab = Market_map()


curdoc().add_root(Tabs(tabs=[matrix_tab, map_tab]))
curdoc().title = "Persistence & EM App"
