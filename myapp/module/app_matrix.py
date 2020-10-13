from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, Spacer
from bokeh.models import ColumnDataSource, Slider, Select,Panel,RangeSlider,Tabs, TableColumn,\
    DataTable,Arrow,OpenHead,Div, HoverTool, MultiChoice, CustomJS

from bokeh.plotting import figure
import pandas as pd
import numpy as np
import json
from value_transform import value_transfrom

def Persistence_EM_Matrix():
    ### ------------ read files ------------
    
    final = pd.read_csv('./data/App_data/AQ_EM_tse_final.csv',encoding='utf-8')
    profit = pd.read_csv('./data/App_data/AQ_EM_tse_final_profibility_1.csv',encoding='utf-8')
    profit['company_name'] = profit.company.astype(str)+" "+profit.company_abbreviation
    #此檔案中放的為dict, key為用途及名稱，value為list
    with open('app_lists.json','r')as f: 
        app_lists = json.load(f)
    with open('smallco.json','r')as f: 
        smallco_index = json.load(f)
        
    ### ------------ 左側的選項 ------------

    year = Select(title='Year：',value='200812',
                        options=list(final.yyyymm.drop_duplicates().sort_values().astype(str)))

    industry = Select(title='Industry:',value='水泥工業',
                          options = list(profit.tse_industry_name.drop_duplicates())+['All Sectors'])

    index_factor = Select(title='Compared Market Index:',value='TWN50',options=["TWN50", "TM100","TF001"])

    company_list = list((profit.query("yyyymm==200812 & tse_industry_name=='水泥工業'").company_name.astype(str)))

    company_code = Select(title='Company Code: ',value='',options=['']+company_list)

    persistence = Select(title='Persistence:',value='ebit_slope_standard',
                         options= app_lists['options_persistence'])

    EM = Select(title='EM:',value='Jones_model_measure',options=app_lists['options_EM'])

    profit_measure = Select(title='Profit Measure:',value='ROA_ebi',
                            options=app_lists['options_profit_measure'])

    Persistence_percent = RangeSlider(start=0, end=100,value=(20,40), step=1, title="Persistence % :")
    EM_percent = RangeSlider(start=0, end=100,value=(20,40), step=1, title="EM %:")

    #根據選擇日期、產業更新公司列表選項
    ###############################################################################
    def update_company_list(attr,old,new):
        selected_year = year.value
        selected_industry = industry.value
        if selected_industry !='All Sectors':
            company_list = list((profit.query("yyyymm==@selected_year & tse_industry_name==@selected_industry").\
                                 company_name.sort_values().astype(str)))
            #前面加入空値，代表沒有選公司
            company_code.options = ['']+company_list
            #default 為空値
            company_code.value = ''
        else:
            company_list = list((profit.query("yyyymm==@selected_year").\
                            company_name.sort_values().astype(str)))
            #前面加入空値，代表沒有選公司
            company_code.options = ['']+company_list
            #default 為空値
            company_code.value = ''

    #選出畫圖的資料
    ###############################################################################
    def get_plot_data():
        selected_year=year.value
        selected_industry = industry.value
        selected_index_factor = index_factor.value
        selected_Persistence = persistence.value
        selected_EM = EM.value
        selected_profit_measure = profit_measure.value
        if selected_industry !='All Sectors':
            data = profit.query('yyyymm == @selected_year & tse_industry_name  == @selected_industry')
        else :
            data = profit.query('yyyymm == @selected_year')
        #依據日期、產業選擇
        data = data[(data[selected_Persistence].notna()) & (data[selected_EM].notna())]
        #因為有可能選擇的資料也是之後要保留的資料，因此先備份，以免在rename後找不到資料
        origin_data = [selected_Persistence,selected_EM,selected_profit_measure]
        origin = data[origin_data]
        #重新命名，在ColumnDataSource中較好使用
        data.rename(columns={selected_Persistence:'Persistence',selected_EM:'EM',selected_index_factor:'index_factor',
                             selected_profit_measure:'profit_measure'} , inplace=True)
        for i in origin_data:
            data[i] = origin[i]
        data['Persistence'] = data.Persistence.apply(lambda x:value_transfrom(x,selected_Persistence))
        data['EM'] = data.EM.apply(lambda x:value_transfrom(x,selected_EM))
        data['color'] = data['index_factor'].apply(lambda x:'green' if x=='Y' else 'blue')
        data['color'] = data.apply(lambda x:'red' if str(x.company) in smallco_index['2020'] else x.color, 1)

        profit_min = data['profit_measure'].min()
        profit_range = data['profit_measure'].max()-data['profit_measure'].min()
        data['profit_score'] = data['profit_measure'].apply(lambda x:((x-profit_min)/profit_range)*25+5\
                                                            if profit_range!=0 else 30 if x==1 else 5)
        table_data = data[app_lists['select_stock_picking_table_column']]
        data_for_source = data.fillna('--')
        if company_code.value!='':
            data_for_source['text'] = data_for_source['company'].apply(lambda x:'.Here' if x==int(company_code.value[:4])else '')
        else :
            data_for_source['text']=''
        data_for_source = data_for_source[~data_for_source.isin([np.nan, np.inf, -np.inf]).any(1)]
        if company_code.value!='':
            select_co = int(company_code.value[:4])
            data_for_source['select_p'] = data.query('company==@select_co')['Persistence'].to_list()[0]
            data_for_source['select_e'] = data.query('company==@select_co')['EM'].to_list()[0]
        else :
            data_for_source['select_p'] = np.nan
            data_for_source['select_e'] = np.nan

        plot_source = ColumnDataSource(data_for_source)
        return (plot_source,table_data)
    def get_stock_picking_table_data(table_data):
        df = table_data
        Persistence_top = df.Persistence.quantile(Persistence_percent.value[1]/100)
        Persistence_low = df.Persistence.quantile(Persistence_percent.value[0]/100)
        EM_top = df.EM.quantile(EM_percent.value[1]/100)
        EM_low = df.EM.quantile(EM_percent.value[0]/100)
        df = df.query('Persistence <= @Persistence_top & Persistence >= @Persistence_low & EM <= @EM_top & EM >= @EM_low')
        df = df.applymap(lambda x:round(x,2) if type(x)==float else x)
        stock_picking_table_co_choice.options = (df.company.astype(str)+' '+df.company_abbreviation).sort_values().to_list()
        stock_picking_table_co_num.text = f'Total: {df.shape[0]} company'
        return ColumnDataSource(df)

    def get_stock_return_table_2_data():
        selected_year=year.value
        selected_index_factor = index_factor.value
        df = profit.rename(columns={selected_index_factor:'index_factor'})
        df = df.query('yyyymm == @selected_year & index_factor=="Y"')
        return ColumnDataSource(df)

    def get_stock_return_table_3_data(stock_picking_table_source,stock_return_table_2_source):
        if stock_picking_table_source.data['yearly_return'].size ==0 :
            stock_average = [' ']
        else : stock_average = [round(np.nanmean(stock_picking_table_source.data['yearly_return']),4)]

        if stock_return_table_2_source.data['yearly_return'].size ==0 :
            etf_average = [' ']
        else : etf_average = [round(np.nanmean(stock_return_table_2_source.data['yearly_return']),4)]

        return ColumnDataSource(data={'Stock Picking Return (Equally Weighted)':stock_average,
                                      "ETF Return (Equally Weighted)" :etf_average 
                                      })
    def get_matrix_plot_data():
        selected_year=year.value
        df = profit.query('yyyymm == @selected_year')
        df = df[app_lists['options_persistence']+app_lists['options_EM']].corr()
        df = df.apply(lambda x:round(x,2))
        return ColumnDataSource(df)
    ###################################################
    # 製作圖、表  
    def make_scatter_plot(plot_source):
        hover = HoverTool( names=['circle'],
                            tooltips=[('Company Abbreviation :','@company_abbreviation'),
                                        ('Company Code :','@company'),
                                        ('Persistence','@Persistence'),
                                        ('EM :','@EM'),('ROA (EBI) :','@ROA_ebi'),
                                        ('EPS :','@eps'),('ROE_b :','@ROE_b'),
                                        ('Diluted EPS :','@eps_diluted'),('Yearly Return','@yearly_return')]
                         )
        plot = figure(plot_height=500, plot_width=800,
                          tools = ['box_zoom','reset',hover],
                          x_axis_label='Persistence (Log Transformed)',
                          y_axis_label='EM (Log Transformed)', 
                          toolbar_location="right"
                     )
        plot.circle(x="Persistence", y="EM", source=plot_source,color= 'color',size='profit_score', name='circle',
                    line_color=None,alpha=0.5)
#         plot.text('Persistence','EM','text',source=plot_source,color='red',text_font_style='bold',text_font_size='20pt')
        plot.asterisk('select_p','select_e',source=plot_source,color='red',size=20)
        plot.toolbar.active_drag = None
        return plot
    def make_stock_picking_table(stock_picking_table_source):
        columns = []
        for colnames in stock_picking_table_source.data.keys():
            if colnames !='index':
                columns.append(TableColumn(field=colnames, title=colnames, width=6*len(colnames)))
        stock_picking_table = DataTable(source=stock_picking_table_source, columns=columns, width=4000, height = 500)
        return (stock_picking_table)

    def make_stock_return_table_1(stock_picking_table_source):
        columns = []
        for colnames in ['tse_industry_name','company','company_abbreviation','index_factor','yearly_return']:
            columns.append(TableColumn(field=colnames, title=colnames, width=6*len(colnames)))
        stock_return_table_1 = DataTable(source=stock_picking_table_source, columns=columns, height = 500)
        return (stock_return_table_1)
    def make_stock_return_table_2(stock_return_table_2_source):
        columns = []
        for colnames in ['tse_industry_name','company','company_abbreviation','index_factor','yearly_return']:
            columns.append(TableColumn(field=colnames, title=colnames, width=6*len(colnames)))
        stock_return_table_2 = DataTable(source=stock_return_table_2_source, columns=columns, height = 500)
        return (stock_return_table_2)

    def make_stock_return_table_3(stock_return_table_3_source): 
        columns = []
        for colnames in stock_return_table_3_source.data.keys():
            if colnames !='index':
                columns.append(TableColumn(field=colnames, title=colnames, width=6*len(colnames)))
        stock_return_table_3 = DataTable(source=stock_return_table_3_source, columns=columns)
        return (stock_return_table_3)

    def make_matrix_plot(matrix_plot_source):
        columns = []
        for colnames in matrix_plot_source.data.keys():
            if colnames =='index':
                columns.append(TableColumn(field=colnames, title=' ', width=200))
            else:
                columns.append(TableColumn(field=colnames, title=colnames, width=6*len(colnames)))
        matrix_plot = DataTable(source=matrix_plot_source, columns=columns, index_position=None, width = 2500, height=300)
        return (matrix_plot)
    ###################################################
    # 更新 
    def update(attr,old,new):
        stock_picking_table_co_choice.value = []
        new_plot_source,new_table_data=get_plot_data()
        plot_source.data.update(new_plot_source.data)


        new_stock_picking_table_source = get_stock_picking_table_data(new_table_data)
        stock_picking_table_source.data.update(new_stock_picking_table_source.data)

        new_stock_return_table_2_source = get_stock_return_table_2_data()
        stock_return_table_2_source.data.update(new_stock_return_table_2_source.data)

        new_stock_return_table_3_source = get_stock_return_table_3_data(new_stock_picking_table_source,new_stock_return_table_2_source)
        stock_return_table_3_source.data.update(new_stock_return_table_3_source.data)

        new_matrix_plot_source = get_matrix_plot_data()
        matrix_plot_source.data.update(new_matrix_plot_source.data)
    def update_stock_picking(attr,old,new):
        
        pick_list = list(map(lambda x:x[:4],stock_picking_table_co_choice.value))
        new_plot_source,new_table_data=get_plot_data()
        new_stock_picking_table_source = get_stock_picking_table_data(new_table_data)
        df = pd.DataFrame(new_stock_picking_table_source.data).iloc[:,1:]
        if len(pick_list)==0:
            df = df
        else:
            df = df.query('company in @pick_list')
        stock_picking_table_source.data.update(ColumnDataSource(df).data)
        stock_picking_table_co_num.text = f'Total: {df.shape[0]} company'
            

    ###################################################
    # initial 

    plot_source,table_data = get_plot_data()
    plot = make_scatter_plot(plot_source)
    plot_explain = Div(text =
                       '''
                       <span style="padding-left:20px">顏色(綠色): 該公司在該年，有被列在所選的Compared Market Index中 <br/>
                       <span style="padding-left:20px">顏色(紅色): 該公司在該年，有被列在中小型成分股中 <br/>
                       <span style="padding-left:20px">大小: 圈圈越大，代表該公司Profit Measure越大
                       ''')
    tab1 = Panel(child=column(Spacer(height=35), plot, Spacer(height=20), plot_explain), title='Persistence EM Matrix')

    
    stock_picking_table_co_choice = MultiChoice(title = 'select_company:', value=[], options=[], placeholder = '選擇想看的公司')
    stock_picking_table_co_choice.js_on_change("value", CustomJS(code="""
        console.log('multi_choice: value=' + this.value, this.toString())
    """))
    stock_picking_table_co_num = Div(text ='Total:   company')
    stock_picking_table_source = get_stock_picking_table_data(table_data)
    stock_picking_table = make_stock_picking_table(stock_picking_table_source)
    tab2 = Panel(child=column(stock_picking_table_co_num, stock_picking_table, stock_picking_table_co_choice), title='Stock Picking Table')

    div1 = Div(text ='Table 1: The next year return of stocks from the matrix')
    stock_return_table_1 = make_stock_return_table_1(stock_picking_table_source)
    div2 = Div(text ='Table 2: The next year return of stocks in ETF')
    stock_return_table_2_source = get_stock_return_table_2_data()
    stock_return_table_2 = make_stock_return_table_2(stock_return_table_2_source)
    div3 = Div(text ='Table 3: The next year return of equally weighted portfolios')
    stock_return_table_3_source = get_stock_return_table_3_data(stock_picking_table_source,stock_return_table_2_source)
    stock_return_table_3 = make_stock_return_table_3(stock_return_table_3_source)
    tab3 = Panel(child=row([column(div1,stock_return_table_1),
                            column(div2,stock_return_table_2),
                            column(div3,stock_return_table_3)]),
                 title='Stock Return Table')

    matrix_plot_source = get_matrix_plot_data()
    matrix_plot = make_matrix_plot(matrix_plot_source)
    matrix_plot_explain = Div(text = 
        '''
        Persistence: <br/>
        <span style="padding-left:50px">ebit_slope_standard <br/>
        <span style="padding-left:50px">operating_slope_standard <br/>
        <span style="padding-left:50px">yoy_ebit_standard <br/>
        <span style="padding-left:50px">yoy_operating_standard <br/>
        <br/><br/>
        EM: <br/>
        <span style="padding-left:50px">Jones_model_measure <br/>
        <span style="padding-left:50px">Modified_Jones_model_measure <br/>
        <span style="padding-left:50px">Performance_matching_measure <br/>
        <span style="padding-left:50px">opacity_Jones_model_measure <br/>
        <span style="padding-left:50px">opacity_modified_Jones_model_measure <br/>
        <span style="padding-left:50px">opacity_performance_matching <br/>
        ''')
    tab4 = Panel(child=column(matrix_plot,row(Spacer(width=20), matrix_plot_explain)), title='Correlation Matrix of Persistence & EM')

    tabs = Tabs(tabs=[tab1,tab2,tab3,tab4])

    ###################################################
    # input change
    year.on_change('value', update, update_company_list)
    industry.on_change('value', update, update_company_list)
    index_factor.on_change('value', update)
    company_code.on_change('value', update)
    persistence.on_change('value', update)
    EM.on_change('value', update)
    profit_measure.on_change('value', update)
    Persistence_percent.on_change('value', update)
    EM_percent.on_change('value', update)
    stock_picking_table_co_choice.on_change('value', update_stock_picking)

    ###################################################
    # layout
    div_title = Div(text ='Persistence & EM Matrix',style={'font-size': '200%', 'color': 'blue'})
    inputs = column(div_title,year, industry, index_factor, company_code,persistence,EM,profit_measure,
                             Persistence_percent,EM_percent, background='gainsboro')
    final_layout = row(inputs, tabs, width=1200)
    return Panel(child = column(Spacer(height = 35), final_layout), title = 'Persistence & EM 概況')

