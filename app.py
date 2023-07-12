# import modules
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
from datetime import datetime, timedelta, date
import pathlib
import re

# from app import app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css?family=Questrial', '/Users/maddywang/tgc-dash/assets/styles.css'],  suppress_callback_exceptions=True)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

app.title = "Hoomi Analytics Dashboard"

server = app.server

app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data (sample data before connecting API data fetch)
revenue_78_weeks_df = pd.read_csv("data/vend-total_revenue-for-supplier-by-week (2022-01-03 to 2023-06-25).csv")
sale_count_product_78_weeks = pd.read_csv("data/vend-sale_count-for-product-by-week (2022-01-03 to 2023-06-27).csv")
item_count_product_78_weeks = pd.read_csv("data/vend-item_count-for-product-by-week (2022-01-03 to 2023-06-27).csv")
products = pd.read_csv("data/product-export.csv")

#Data Cleaning

def change_to_datetime(dataframe):
    df = dataframe.copy()
    num_index = df.shape[1] - 6
    relevant_df = df.iloc[:, 1:num_index]
    # Convert dates to datetime format
    dates_to_convert = relevant_df.columns.tolist()[1:(len(relevant_df)-6)]
    datetime_dates = []
    for date in dates_to_convert:
        try:
            datetime_dates.append(datetime.strptime(remove_ordinal(date), '%d %b %Y'))
        except ValueError:
            pass  # Skip invalid date column
    df.rename(columns=dict(zip(dates_to_convert, datetime_dates)), inplace=True)
    return df.iloc[:-5]

# Function to remove ordinal indicators from a date string
def remove_ordinal(date):
    pattern = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
    return re.sub(pattern, r'\1', date)

def special_adjust(dataframe):
    df = dataframe.copy()
    num_index = df.shape[1] - 6
    relevant_df = df.iloc[:, 1:num_index]
    # Convert dates to datetime format
    dates_to_convert = relevant_df.columns.tolist()[6:85]
    datetime_dates = []
    for date in dates_to_convert:
        try:
            datetime_dates.append(datetime.strptime(remove_ordinal(date), '%d %b %Y'))
        except TypeError:
            pass  # Skip invalid date column
    df.rename(columns=dict(zip(dates_to_convert, datetime_dates)), inplace=True)
    return df.iloc[:-6]

def rev_special_adjust(dataframe):
    df = dataframe.copy()
    num_index = df.shape[1] - 5
    relevant_df = df.iloc[:, :num_index]
    # Convert dates to datetime format
    dates_to_convert = relevant_df.columns.tolist()[1:78]
    datetime_dates = [datetime.strptime(remove_ordinal(date), '%d %b %Y') for date in dates_to_convert]
    df.rename(columns=dict(zip(dates_to_convert, datetime_dates)), inplace=True)
    return df.iloc[:-6]

updated_revenue_78_weeks=rev_special_adjust(revenue_78_weeks_df)
updated_78_sale_count = special_adjust(sale_count_product_78_weeks)
updated_78_item_count_products = special_adjust(item_count_product_78_weeks)

suppliers_list = updated_revenue_78_weeks['Supplier'].sort_values()

def restrict_date_range(date, df, earliest_date=True):
    """
    date should be in datetime
    """
    datetime_columns = [col for col in df.columns if isinstance(col, datetime)]
    non_datetime_columns = [col for col in df.columns if col not in datetime_columns]
    if earliest_date:
        restricted_dates = [col for col in datetime_columns if col >= date]
    else:
        restricted_dates = [col for col in datetime_columns if col <= date]
    return df[restricted_dates + non_datetime_columns]

# To get current quarter helper function
def current_quarter(date):
    quarter_month = (date.month - 1) // 3 * 3 + 1
    beginning_of_quarter = datetime(date.year, quarter_month, 1)
    return beginning_of_quarter.strftime('%Y-%m-%d')

# More helper functions for revenue purposes

def length_previously(dates_range):
    if bool(re.search(r'month$', dates_range)):
        return "Last Month"
    elif bool(re.search(r'quarter$', dates_range)):
        return "Last Quarter"
    elif bool(re.search(r'year$', dates_range)):
        return "Last Year"
    return ""

def determine_delta(revenue_df, brand_name, start_date, dates_range):
    """
    Return total revenue as last month/quarter/year based on given dates, helps determine delta value for indicator below.
    """
    start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    e_date = start_date_datetime - timedelta(days = 1)
    dates = []
    if dates_range == 'select':
        return 0 # Don't know what to put, so keep it at 0
    if bool(re.search(r'month$', dates_range)):
        s_date = e_date- timedelta(days=30)
        revenue_df = restrict_date_range(s_date, revenue_df)
        revenue_df = restrict_date_range(e_date, revenue_df, earliest_date = False)
        dates = [col for col in revenue_df if isinstance(col, datetime)]
    elif bool(re.search(r'quarter$', dates_range)):
        s_date = e_date- timedelta(days=91)
        revenue_df = restrict_date_range(s_date, revenue_df)
        revenue_df = restrict_date_range(e_date, revenue_df, earliest_date = False)
        dates = [col for col in revenue_df if isinstance(col, datetime)]
    elif bool(re.search(r'year$', dates_range)):
        s_date = e_date- timedelta(days=365)
        revenue_df = restrict_date_range(s_date, revenue_df)
        revenue_df = restrict_date_range(e_date, revenue_df, earliest_date = False)
        dates = [col for col in revenue_df if isinstance(col, datetime)]
    delta_total_revenue = np.sum(revenue_df[revenue_df['Supplier'] == brand_name][dates].values)
    return delta_total_revenue

#Plotly.Dash components
pyramid_graph = html.Div([dcc.Graph(id="pyramid-chart", config={'displayModeBar': False})])
pie_chart = dcc.Graph(id="pie-chart")
no_sale_table = dcc.Graph(id="no-sale-table")

all_input_cards = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H2('Pick your brand:', className="input-card-title"),
                html.Div(
                dcc.Dropdown(className="input-card", id='supplier-dropdown', 
                             options=[{'label': company, 'value': company} for company in suppliers_list], 
                             placeholder='Select a company', optionHeight=20)
                             ),

                html.Div([
                    html.H3("Select Range:", className="input-card-title"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        className="input-card",
                        min_date_allowed= date(2022, 1, 3),
                        max_date_allowed=date(2023, 6, 25),
                        start_date = date(2022, 1, 3),
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        display_format='YYYY-MM-DD',
                        day_size=30
                    ),
                    dcc.DatePickerSingle(
                        className="input-card",
                        id='date-picker-single',
                        min_date_allowed= date(2022, 1, 3),
                        max_date_allowed=date(2023, 6, 25),
                        initial_visible_month = date(2023, 6, 1),
                        date = date(2023, 6, 23),
                        display_format='YYYY-MM-DD',
                        day_size=30
                    )
            ])
                ,

                dcc.RadioItems(
                        id='timeframe-radio',
                        className = 'btn-group',
                        inputClassName='btn-check',
                        labelClassName="btn btn-outline-primary",
                        options=[
                            {'label': 'Select Dates', 'value': 'select'},
                            {'label': 'Last Month', 'value': 'last-month'},
                            {'label': 'Last Quarter', 'value': 'last-quarter'},
                            {'label': 'Last Year', 'value': 'last-year'},
                            {'label': 'Current Month', 'value': 'curr-month'},
                            {'label': 'Current Quarter', 'value': 'curr-quarter'},
                            {'label': 'Current Year', 'value': 'curr-year'}
                        ],
                        value='select',
                        labelStyle={'display': 'block'},
                ),
            ],
        style = {'height':'225px'}),
    ],
    color = 'success', outline = True
)

total_rev = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H2("Total Revenue:", className="visual"),
                dcc.Graph(id="total-revenue", figure={}),
            ],
        )
    ],
    color='info', outline=True
)

items_sold = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Items Sold:", className="visual"),
                dcc.Graph(id="items-sold", figure={}),
            ],
        )
    ],
    color='info', outline=True
)
greenscore = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H2("Greenscore", className="visual"),
                dcc.Graph(id='greenscore', figure={})
            ],
        )
    ],
    color='success', outline = True
)

sale_count = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Sale Count:", className="visual"),
                dcc.Graph(id="sale-count", figure={}),
            ],
        )
    ],
    color='info', outline=True
)

def get_gauge_color(value):
    if value < 50:
        return 'red'
    elif value < 70:
        return 'orange'
    else:
        return 'green'


app.layout = html.Div([
    dbc.Container(
        [
            html.Img(src="assets/tgc-logo.png", className="logo", style = {'vertical-align': 'middle', 'display': 'inline-block'}),
            dbc.Row(
                dbc.Col(
                    html.H1("Sales Dashboard by TGC", className = 'logo-text', 
                            style = {'font-family': 'Questrial', 'color': '#10890c', 'vertical-align': 'middle'}),
                    width=12
                )
            )
        ], className = 'banner'
    ),
    dbc.Container(
        [
            dbc.Card(all_input_cards, style= {'height':'225px'}),
            html.Br(),
            dbc.Row(children = [
                dbc.Col(greenscore, width = 3),
                dbc.Col(total_rev, width = 3),
                dbc.Col(sale_count, width = 3),
                dbc.Col(items_sold, width = 3),
                ], className = 'row', style={"height": "200px"} ),
            html.Br(),
            pyramid_graph,
            dbc.Row(children = [
                dbc.Col(pie_chart, width = 8),
                dbc.Col(no_sale_table, width = 4),
                ], className = 'row'),
        ], 
    )
])

# Callback to update the dashboard title based on the selected brand
@app.callback(
    Output('dashboard-title', 'children'),
    Input('supplier-dropdown', 'value')
)

def update_dashboard_title(selected_brand):
    if selected_brand is None:
        return 'Sales Dashboard by TGC'
    return f'{selected_brand}'

# Callback to update dates when 'any of the Radio Items are selected'
@app.callback(
        Output('date-picker-range','style'),
        Output('date-picker-single', 'style'),
        [Input('timeframe-radio', 'value')]
)

def toggle_date_picker(value):
    if value == 'select':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'block'}, {'display': 'block'}
    
@app.callback(
        Output('pyramid-chart', 'style'),
        Output('pie-chart', 'style'),
        Output("total-revenue", "style"),
        Output("no-sale-table", "style"),
        Output("sale-count", "style"),
        Output("items-sold", "style"),
        Output("greenscore", "style"),
        Input('supplier-dropdown', 'value'))

def no_brand_show(brand_name):
    if brand_name is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    [Input('timeframe-radio', 'value'),
     Input('date-picker-single', 'date')]    
)

def update_date_range(value, date):
    if date:
        selected_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        selected_date = datetime(2023, 6, 25) #Change for next versions with real data (like today)
    if value == 'last-month':
        starting_datetime = selected_date - timedelta(days=30)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, date
    elif value == 'last-quarter':
        starting_datetime = selected_date - timedelta(days=91)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, date
    elif value == 'last-year':
        starting_datetime = selected_date - timedelta(days=365)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, date
    elif value == 'curr-month':
        starting_datetime = datetime(selected_date.year, selected_date.month, 1)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, date
    elif value == 'curr-quarter':
        start_date = current_quarter(selected_date)
        return start_date, date
    elif value == 'curr-year':
        starting_datetime = datetime(selected_date.year, 1, 1)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, date
    else:
        starting_datetime = datetime(2022, 1, 3)
        start_date = starting_datetime.strftime('%Y-%m-%d')
        return start_date, None

# Callback to update the graphs based on the selected product
@app.callback(
    [Output('pyramid-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output("total-revenue", "figure"),
    Output("no-sale-table", "figure"),
    Output("sale-count", "figure"),
    Output("items-sold", "figure"),
    Output("greenscore", "figure")],
    [Input('supplier-dropdown', 'value'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range', 'end_date'),
    Input('timeframe-radio', 'value')
    ]
)

def update_graphs(brand_name, start_date, end_date, dates_range):
    if brand_name is None: #No brand has said anything
        return {}, {}, {}, {}, {}, {}, {}
    revenue = updated_revenue_78_weeks.copy()
    sale_count = updated_78_sale_count.copy()
    item_count = updated_78_item_count_products.copy()
    if start_date is not None:
        s_date = datetime.strptime(start_date, '%Y-%m-%d')
        revenue = restrict_date_range(s_date, revenue)
        sale_count = restrict_date_range(s_date, sale_count)
        item_count = restrict_date_range(s_date, item_count)
    if end_date is not None:
        e_date = datetime.strptime(end_date, '%Y-%m-%d')
        revenue = restrict_date_range(e_date, revenue, earliest_date = False)
        sale_count = restrict_date_range(e_date, sale_count, earliest_date= False)
        item_count = restrict_date_range(e_date, item_count, earliest_date=False)
    dates = [col for col in revenue if isinstance(col, datetime)]

    # Filter the product data based on the selected brand name
    sku_of_products_item_df = item_count[item_count['Supplier'] == brand_name]['SKU']
    brands_products_item_df = item_count[item_count['SKU'].isin(sku_of_products_item_df)]
    brands_products_item_df['Sum of Items Sold'] = item_count[dates].sum(axis=1)
    # zero_items_sold = brands_products_item_df[brands_products_item_df['Sum of Items Sold'] == 0]['Product']
    brands_products_item_df= brands_products_item_df[brands_products_item_df['Sum of Items Sold'] != 0]
    top_10 = brands_products_item_df.nlargest(10, 'Sum of Items Sold').sort_values('Sum of Items Sold', ascending=True)
    fig1 = go.Figure(
        data=[
            go.Bar(
                y=top_10['Product'],
                x=top_10['Sum of Items Sold'],
                orientation='h',
                marker=dict(color='steelblue')
            )
        ],
        layout=go.Layout(
            title='Best Sellers',
            xaxis=dict(title='Items Sold'),
            yaxis=dict(title='Products'),
            height=500,
            bargap=0.1
        )
    )
    
    sku_of_products_sales_df = sale_count[sale_count['Supplier'] == brand_name]['SKU']
    brands_products_sales = sale_count[sale_count['SKU'].isin(sku_of_products_sales_df)]
    brands_products_sales['Sum Sale Count'] = sale_count[dates].sum(axis=1)
    zero_sales_products = brands_products_sales[brands_products_sales['Sum Sale Count'] == 0]['Product'].values
    brands_products_sales = brands_products_sales[brands_products_sales['Sum Sale Count'] > 0]
    if len(brands_products_sales) <= 19:
        # Compute market share directly for all products
        sales_count = brands_products_sales['Sum Sale Count'].values
        product_names = brands_products_sales['Product'].values
    else:
        # Compute market share for the top 19 products and group the rest as "Other"
        top_19 = brands_products_sales.nlargest(19, 'Sum Sale Count').sort_values('Sum Sale Count', ascending=True)
        other_products = brands_products_sales[~brands_products_sales['Product'].isin(top_19['Product'])]
        other_sales_count = np.sum(other_products['Sum Sale Count'].values)

        top_19_names = top_19['Product'].values.tolist()
        top_19_sales = top_19['Sum Sale Count'].values.tolist()
        top_19_percentage = [(sale_count / np.sum(top_19_sales)) * 100 for sale_count in top_19_sales]

        top_19_names.append('Other')
        top_19_sales.append(other_sales_count)
        top_19_percentage.append((other_sales_count / np.sum(top_19_sales)) * 100)

        sales_count = top_19_sales
        product_names = top_19_names

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=product_names,
                values=sales_count,
                hole=0.25,
                title="Market Share",
                textinfo='percent'
            )
        ]
    )
    
    total_revenue = np.sum(revenue[revenue['Supplier'] == brand_name][dates].values)
    tr = "{:,.2f}".format(total_revenue)
    tr = tr.replace(',', '')
    float_tr = float(tr)
    fig3 = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value= float_tr,
            number = {"prefix":"$"},
            delta={"position": "bottom", "reference": determine_delta(updated_revenue_78_weeks.copy(), brand_name, start_date, dates_range),
                    'relative': True, "valueformat": ".1f"},
            title = {"text":length_previously(dates_range)},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig3.update_layout(
        height=160,
        width=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    zero_sales_products_df = pd.DataFrame(zero_sales_products, columns = ['No Sale Products'])
    fig4 = dict(
        data =[
            dict(
                type='table',
                header=dict(values=list(zero_sales_products_df.columns)),
                cells=dict(values=[zero_sales_products_df[col] for col in zero_sales_products_df.columns])
            )
        ],
        layout=dict(height=500)
    )
    sale_count_num = np.sum(sale_count[sale_count['Supplier'] == brand_name][dates].values)
    fig5 = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value= int(sale_count_num),
            delta={"reference": determine_delta(updated_78_sale_count.copy(), brand_name, start_date, dates_range),
                    'relative': True, "valueformat": ".1f"},
            title = {"text":length_previously(dates_range)},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig5.update_layout(
        height=160,
        width=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    item_count_num = np.sum(item_count[item_count['Supplier'] == brand_name][dates].values)
    fig6= go.Figure(
        go.Indicator(
            mode = "number+delta",
            value= int(item_count_num),
            delta={"reference": determine_delta(updated_78_item_count_products.copy(), brand_name, start_date, dates_range),
                    'relative': True, "valueformat": ".1f"},
            title = {"text":length_previously(dates_range)},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig6.update_layout(
        height=160,
        width=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    value = 85
    fig7 = go.Figure(
        go.Indicator(
        mode="gauge+number", 
        value=value, 
        # title={'text': "Green Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        ))
    fig7.update_traces(gauge = {'axis': {'range': [None, 100]},  # Customize the gauge range
                                 'bar': {'color': get_gauge_color(value)}  # Dynamically set the gauge bar color based on value
                                })
    fig7.update_layout(
        width=200,                                # Customize the plot width
        height=150,                               # Customize the plot height
        margin=dict(l=50, r=50, t=50, b=50))     # Customize the plot margins
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


if __name__ == '__main__':
    app.run_server(debug=True)


