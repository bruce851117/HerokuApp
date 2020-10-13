from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, Spacer
from bokeh.models import ColumnDataSource, Slider, Select,Panel,RangeSlider,Tabs, TableColumn,\
    DataTable,Arrow,OpenHead,Div, HoverTool, HTMLTemplateFormatter, Button, CustomJS
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import json
def Market_map():
    df = pd.read_csv('./data/App_data/AQ_EM_tse_final_profibility_1.csv',encoding='utf-8')
    df['yyyymm'] = df['yyyymm'].astype(str)
    df['company_name'] = df.company.astype(int).astype(str)+' '+df.company_abbreviation

    year = Select(title='年份：',value='201812',options=df.yyyymm.drop_duplicates().sort_values().astype(str).to_list(), width = 100)
    col_type = Select(title='分類方式：',value='產業',options=['產業','新產業','市值排名'], width = 100)
    sort_type = Select(title='排序方式：',value='市值',options=['市值','eps','負債比','working capital/TA'], width = 100)
    color_type = Select(title='上色方式(前10%)', value='市值', options=['市值','資產','eps'], width = 100)
    input_dict = {'產業':'tse_industry_name',
                  '市值':'market_value',
                  '資產':'asset',
                  '新產業':'new_industry_code',
                  '市值排名':'value_rank',
                  '負債比':'debt ratio',
                  'working capital/TA':'working capital/TA',
                  'eps':'eps'
                  
                 }

    def get_data():
        select_year = year.value
        col = input_dict[col_type.value]
        sort = input_dict[sort_type.value]
        color = input_dict[color_type.value]
        data = df.query('yyyymm==@select_year')
        if col=='value_rank':
            data = data.dropna(subset=['market_value'])
            data['rank_h'] = (data['market_value'].rank(ascending=False)/100).astype(int)
            rmax = data['rank_h'].max()
            def cal_vrank(x):
                if x==0:
                    return '0~100'
                elif x==rmax:
                    return f'{rmax*100}名以上'
                else :
                    return f'{x*100}名~{(x+1)*100}名'
            data['value_rank'] = data['rank_h'].apply(cal_vrank)
            del data['rank_h']
        else :
            data = data.dropna(subset=[col,sort,color])
        sorting = data.sort_values(sort, ascending=False).groupby([col]).agg({'company_name':list})
        if col!='value_rank':
            sorting['co_num'] = sorting['company_name'].apply(lambda x:len(x))
            sorting = sorting.sort_values('co_num',ascending=False)[['company_name']]
        table = pd.DataFrame.from_dict(sorting.T.to_dict('records')[0],'index').T.fillna('')
        source_table = ColumnDataSource(table)
        top_co = data[ data[color] > data[color].quantile(0.9)]['company_name'].to_list()
        source_color = ColumnDataSource(data = {'top_co':top_co})
       
        return source_table, source_color
        
    source_table, source_color = get_data()

    template='<div style="background:<%=(function colorfromint(){'+\
            f'if({str(source_color.data["top_co"])}.indexOf(value) > -1)'+\
            '{return("#ffb3b3")}else{return("white")}}()) %>; color: black"> <%= value %></div>'
    formatter =  HTMLTemplateFormatter(template=template)
    columns = []
    for colnames in source_table.data.keys():
        if colnames !='index':
            columns.append(TableColumn(field=colnames, title=colnames, formatter=formatter))
        
    table = DataTable(source=source_table, columns=columns, width = 2500, height = 1000, editable=True)


    def update(attr,old,new):
        new_source_table, new_source_color = get_data()

        new_template = '<div style="background:<%=(function colorfromint(){'+\
                                f'if({str(new_source_color.data["top_co"])}.indexOf(value) > -1)'+\
                                '{return("#ffb3b3")}else{return("white")}}()) %>; color: black"> <%= value %></div>'
        new_formatter =  HTMLTemplateFormatter(template=new_template)
        new_columns = []
        for colnames in new_source_table.data.keys():
            if colnames !='index':
                new_columns.append(TableColumn(field=colnames, title=colnames, formatter=new_formatter))
        table.source = new_source_table
        table.columns = new_columns
        
        
    year.on_change('value', update)
    col_type.on_change('value', update)
    sort_type.on_change('value', update)
    color_type.on_change('value', update)
        

    hspace = Spacer(width = 20)
    input = row(year, hspace,
                col_type, hspace,
                sort_type, hspace, 
                color_type)
    final_layout = column(input,table)

    return Panel(child = column(Spacer(height = 35), final_layout), title = '資本市場地圖')










