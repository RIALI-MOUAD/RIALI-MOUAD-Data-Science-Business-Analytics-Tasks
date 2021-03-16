
#Importing libraries

import dash
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from flask import Flask
import dash_table
import numpy as np





#Dash

server = Flask(__name__)
app = dash.Dash(__name__ , meta_tags=[{"name": "viewport", "content": "width=device-width"}],external_stylesheets=[dbc.themes.BOOTSTRAP],server=server)
#app.config.supress_callback_exceptions = True

#dataframe
def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }
minProfit0 =  np.array(["Ohio","Colorado","Tennessee","North Calorina","Illinois","Texas","Pennsylvania","Arizona","Florida","Oregon"])
minProfit = np.array(["Ohio","Colorado"])
min1Profit=np.array(["Tennessee","North Calorina"])
min2Profit=np.array(["Illinois","Texas","Pennsylvania"])
min3Profit=np.array(["Arizona"])
min4Profit=np.array(["Florida","Oregon"])
maxProfit=np.array(["New York","Washington","California"])
# Connect to database or data source here
df = pd.read_csv('https://raw.githubusercontent.com/RIALI-MOUAD/Dash-Retail/main/SampleSuperstore.csv')
dfcorr = df.corr()
dfPerStateMean=df.groupby(['State']).mean().drop("Postal Code",axis=1)
dfPerState=df.groupby(['State']).sum().drop("Postal Code",axis=1)
StatesNegProfit=dfPerState.where(dfPerState["Profit"]<0).dropna().index.values
dfPerStateNeg = df.loc[df.State.isin(StatesNegProfit)].groupby(['State']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfminProfit = df.loc[df.State.isin(minProfit)]
dfminProfit0 = df.loc[df.State.isin(minProfit0)]
dfPerCitymin = dfminProfit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipMode = dfminProfit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfmin1Profit = df.loc[df.State.isin(min1Profit)]
dfPerCitymin1 = dfmin1Profit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipMode1 = dfmin1Profit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfmin2Profit = df.loc[df.State.isin(min2Profit)]
dfPerCitymin2 = dfmin2Profit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipMode2 = dfmin2Profit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfmin3Profit = df.loc[df.State.isin(min3Profit)]
dfPerCitymin3 = dfmin3Profit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipMode3 = dfmin3Profit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfmin4Profit = df.loc[df.State.isin(min4Profit)]
dfPerCitymin4 = dfmin4Profit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipMode4 = dfmin4Profit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfmaxProfit = df.loc[df.State.isin(maxProfit)]
dfPerCitymax = dfmaxProfit.groupby(['City']).mean().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerShipModemax = dfmaxProfit.groupby(['Ship Mode']).sum().drop("Postal Code",axis=1).sort_values(by="Discount")
dfPerSubCat0=df.groupby(['Sub-Category']).mean().sort_values(by="Discount")
dfPerSubCatmin=dfminProfit0.groupby(['Sub-Category']).mean().sort_values(by="Discount")
dfPerSubCatmax=dfmaxProfit.groupby(['Sub-Category']).mean().sort_values(by="Discount")


#Define figures here
Fig = make_subplots(rows=2, cols=2, shared_xaxes=False,row_heights=[3.5,0.6],
                    specs=[[{'type': 'xy'},    {'type': 'scene'}],
                           [{'type': 'scene'}, {'type': 'ternary'}]])

figCorr = go.Figure(data=go.Heatmap(df_to_plotly(dfcorr)))
#Fig.add_trace(px.scatter_matrix(df, dimensions=["Quantity","Discount","Sales","Category","Profit"], color="Category"))
#Fig.add_trace(px.scatter_3d(df, y="Discount",z="Profit", x="Category",color="Sub-Category"))
fig = px.scatter_matrix(df, dimensions=["Quantity","Discount","Sales","Category","Profit"], color="Category")
fig2 = px.sunburst(df, path=['Region', 'State','City'], values='Sales',color='Discount', hover_data=['Profit'])
fig3d = px.scatter_3d(df, y="Discount",z="Profit", x="Category",color="Sub-Category")
fig2d = px.scatter(df, x="Discount",y="Profit",marginal_x="box",marginal_y="box")
figSalesDis = px.scatter(df, y="Sales",x="Discount", color="Category",marginal_x="box",title="Sales & Discount for each Category")
figStateProf = px.bar(df, y="State",x="Profit", color="Sub-Category")
figPerState = px.bar(dfPerState, y=dfPerState.index,x="Profit",title="Profit per State")
figPerStateNeg = px.bar(dfPerStateNeg, y=dfPerStateNeg.index.values,x="Discount",color="Profit",title="Discount per State (Profit<0)")
figmin0Profit = px.scatter(dfminProfit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : Ohio, Colorado")
figmin0ProfitGlobal = px.scatter_matrix(dfminProfit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: Ohio, Colorado")
figmin0ProfitCSP = px.sunburst(dfminProfit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: Ohio & Colorado")
figmin0ProfitSP = px.sunburst(dfminProfit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: Ohio & Colorado")
figBarmin0 = px.bar(dfPerCitymin, y=dfPerCitymin.index.values,x="Discount",color="Profit",title="Discount Per City in: Ohio & Colorado")
fig1BarDP0 = px.bar(dfPerShipMode, x=dfPerShipMode.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: Ohio & Colorado")
figmin1Profit = px.scatter(dfmin1Profit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : Tennessee, North Calorina")
figmin1ProfitGlobal = px.scatter_matrix(dfmin1Profit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: Tennessee, North Calorina")
figmin1ProfitCSP = px.sunburst(dfmin1Profit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: Tennessee & North Calorina")
figmin1ProfitSP = px.sunburst(dfmin1Profit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: Tennessee & North Calorina")
figBarmin1 = px.bar(dfPerCitymin1, y=dfPerCitymin1.index.values,x="Discount",color="Profit",title="Discount Per City in: Tennessee & North Calorina")
fig1BarDP1 = px.bar(dfPerShipMode1, x=dfPerShipMode1.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: Tennessee & North Calorina")
figmin2Profit = px.scatter(dfmin2Profit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : Illinois, Texas & Pennsylvania")
figmin2ProfitGlobal = px.scatter_matrix(dfmin2Profit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: Illinois, Texas & Pennsylvania")
figmin2ProfitCSP = px.sunburst(dfmin2Profit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: Illinois, Texas & Pennsylvania")
figmin2ProfitSP = px.sunburst(dfmin2Profit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: Illinois, Texas & Pennsylvania")
figBarmin2 = px.bar(dfPerCitymin2, y=dfPerCitymin2.index.values,x="Discount",color="Profit",title="Discount Per City in: Illinois, Texas & Pennsylvania")
fig1BarDP2 = px.bar(dfPerShipMode2, x=dfPerShipMode2.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: Illinois, Texas & Pennsylvania")
figmin3Profit = px.scatter(dfmin3Profit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : Arizona")
figmin3ProfitGlobal = px.scatter_matrix(dfmin3Profit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: Arizona")
figmin3ProfitCSP = px.sunburst(dfmin3Profit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: Arizona")
figmin3ProfitSP = px.sunburst(dfmin3Profit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: Arizona")
figBarmin3 = px.bar(dfPerCitymin3, y=dfPerCitymin3.index.values,x="Discount",color="Profit",title="Discount Per City in: Arizona")
fig1BarDP3 = px.bar(dfPerShipMode3, x=dfPerShipMode3.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: Arizona")
figmin4Profit = px.scatter(dfmin4Profit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : Florida & Oregon")
figmin4ProfitGlobal = px.scatter_matrix(dfmin4Profit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: Florida & Oregon")
figmin4ProfitCSP = px.sunburst(dfmin4Profit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: Florida & Oregon")
figmin4ProfitSP = px.sunburst(dfmin4Profit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: Florida & Oregon")
figBarmin4 = px.bar(dfPerCitymin4, y=dfPerCitymin4.index.values,x="Discount",color="Profit",title="Discount Per City in: Florida & Oregon")
fig1BarDP4 = px.bar(dfPerShipMode4, x=dfPerShipMode4.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: Florida & Oregon")
figmaxProfit = px.scatter(dfmaxProfit, y="Sales",x="Profit", color="Category",marginal_x="box",marginal_y="box",title="Sales & Profit for : New York, Washington & California")
figmaxProfitGlobal = px.scatter_matrix(dfmaxProfit, dimensions=["Ship Mode","City","Quantity","Sales","Category","Profit"], color="Category",title="Global Data for: New York, Washington & California")
figmaxProfitCSP = px.sunburst(dfmaxProfit, path=['Category', 'Sub-Category'], values='Sales',color='Profit', hover_data=['Profit'],title="Sales & Profit for each Category and Sub-Cataegory for: New York, Washington & California")
figmaxProfitSP = px.sunburst(dfmaxProfit, path=['City', 'Ship Mode'], values='Sales',color='Profit', hover_data=['Profit'],title="Discount per Ship Mode for cities in: New York, Washington & California")
figBarmax = px.bar(dfPerCitymax, y=dfPerCitymax.index.values,x="Discount",color="Profit",title="Discount Per City in: New York, Washington & California")
fig1BarDPmax = px.bar(dfPerShipModemax, x=dfPerShipModemax.index.values,y="Discount",color="Profit",title="Discount Per Ship Mode for: New York, Washington & California")
fig0 = px.pie(dfPerSubCat0, values='Discount', names=dfPerSubCat0.index.values, title='mean discount of each sub-cat for all states')
figmin = px.pie(dfPerSubCatmin, values='Discount', names=dfPerSubCatmin.index.values, title='mean discount of each sub-cat where the profit is minimal')
figmax = px.pie(dfPerSubCatmax, values='Discount', names=dfPerSubCatmax.index.values, title='mean discount of each sub-cat where the profit is maximal')

#Layout
app.title = 'Analytics Dashboard'
app.layout = html.Div(children=[
		html.Div(children = [
		dbc.NavbarSimple(
		    children=[
			dbc.NavItem(dbc.NavLink("Web Portal", href="#")),
		    ],
		    brand="Analytics Dashboard",
		    brand_href="#",
		    color="#001524",
		    dark=True,)],style={'width': '100%'}
		    ),

            html.Div([
    	        dash_table.DataTable(id='table-multicol-sorting',
        		columns=[{"name": i, "id": i} for i in df.columns],
            	    page_current=0,
            	    page_size=10,
            	    page_action='custom',
            	    sort_action='custom',
            	    sort_mode='multi',
            	    sort_by=[]
            	    )],style={'width': '100%','margin': '1% 5% 10px 5%'}),
		html.Div(children=[dcc.Graph(id='subplot', figure=fig),],style={'margin': '5px 1% 0px 1%', 'width':'100%'}),
    	html.Div(children=[dcc.Graph(id='pop', figure=figCorr),],style={'margin': '1% 1% 0px 10%'}),
		html.Div(children=[dcc.Graph(id='Sub-cat-Dis-profit', figure=fig3d),],style={'margin': '1% 1% 0px 10%','width':'100%'}),
	    html.Div(children=[dcc.Graph(id='Profit-Dis', figure=fig2d),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='Sales-Dis', figure=figSalesDis),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Sub-Cat', figure=figStateProf),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit', figure=figPerState),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg', figure=figPerStateNeg),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0', figure=figmin0Profit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0-Global', figure=figmin0ProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0-CSP', figure=figmin0ProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0-SP', figure=figmin0ProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0-B', figure=figBarmin0),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-0-DP', figure=fig1BarDP0),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1', figure=figmin1Profit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1-Global', figure=figmin1ProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1-CSP', figure=figmin1ProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1-SP', figure=figmin1ProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1-B', figure=figBarmin1),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-1-DP', figure=fig1BarDP1),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2', figure=figmin2Profit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2-Global', figure=figmin2ProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2-CSP', figure=figmin2ProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2-SP', figure=figmin2ProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2-B', figure=figBarmin2),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-2-DP', figure=fig1BarDP2),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3', figure=figmin3Profit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3-Global', figure=figmin3ProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3-CSP', figure=figmin3ProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3-SP', figure=figmin3ProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3-B', figure=figBarmin3),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-3-DP', figure=fig1BarDP3),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4', figure=figmin4Profit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4-Global', figure=figmin4ProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4-CSP', figure=figmin4ProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4-SP', figure=figmin4ProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4-B', figure=figBarmin4),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-4-DP', figure=fig1BarDP4),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max', figure=figmaxProfit),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max-Global', figure=figmaxProfitGlobal),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max-CSP', figure=figmaxProfitCSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max-SP', figure=figmaxProfitSP),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max-B', figure=figBarmax),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Profit-Neg-batch-max-DP', figure=fig1BarDPmax),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='Profit-Discount-State', figure=fig2),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Sub-Cat', figure=fig0),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Sub-Cat-min', figure=figmin),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        html.Div(children=[dcc.Graph(id='State-Sub-Cat-max', figure=figmax),],style={'margin': '5px 1% 0px 1%','width':'100%'}),
        #html.Div(children=[dcc.Graph(id='Sub-cat-Dis-profit', figure=fig3d),],style={'margin': '1% 1% 0px 10%','width':'100%'}),
],style={'display': 'flex','flex-direction': 'row','flex-wrap': 'wrap','overflow': 'hidden'})


#Callbacks
#Dash Table
@app.callback(
    Output('table-multicol-sorting', "data"),
    [Input('table-multicol-sorting', "page_current"),
     Input('table-multicol-sorting', "page_size"),
     Input('table-multicol-sorting', "sort_by")])
def update_table(page_current, page_size, sort_by):
    print(sort_by)
    if len(sort_by):
        dff = df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = df

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)
# By : Mouad Riali 
