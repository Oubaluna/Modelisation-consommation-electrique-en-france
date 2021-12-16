from bokeh.plotting import Figure
from bokeh.models.layouts import Row
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
import pydeck as pdk
from bokeh.io import curdoc
from bokeh.models.layouts import Row
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import ColumnDataSource
from pynergized_variables import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image 
from io import BytesIO
from datetime import timedelta
from bokeh.models.widgets import Div


def create_array(array, n_timesteps): 
    '''
    Fonction qui prend en argument un array et un nombre de pas de temps à utiliser pour une prédiction de serie temporelle et renvoie :
    - Les données (échantillons de n pas de temps de la variable température stockée dans la première colonne de l'array.)
    - Et les labels (pour chaque échantillon la valeur prise par la température à l'instant suivant)
    '''
    dataX, dataY = [], []
    for i in range(len(array)-n_timesteps):
        x = array[i:i+n_timesteps] # Température entre t-n et t-1
        y = array[i + n_timesteps] # Température à l'instant t
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


def bokeh_link_model(title, y_pred_pls, y_pred_knn, y_pred_rf, y_pred_sarima, y_test, df_test,
                     title2, y_pred_pls2, y_pred_knn2, y_pred_rf2, y_pred_sarima2, y_test2,
                     title3, y_pred_pls3, y_pred_knn3, y_pred_rf3, y_pred_sarima3, y_test3,
                     t, modeles):
    if modeles == 'Mois':
        t_correction = timedelta(days=t*30)
    elif modeles == 'Jours':
        t_correction = timedelta(days=t)
    elif modeles == '3 heures':
        t_correction = timedelta(hours=t*3)

    p1 = Figure(
            width = 500,
            plot_height=400,
            x_axis_type="datetime",  
            title = 'Prediction de la Consommation française' + title,
            tools=['xwheel_zoom','reset', 'pan'], 
            active_scroll='xwheel_zoom',)
    p1.line(df_test.index+t_correction, y_test, 
           color='#21D19F', 
           legend_label='Données Réelles',
           line_width=1.5,
           line_dash='dashed',)
    p1.line(df_test.index+t_correction, y_pred_pls.ravel(), 
           color='#EEF1BD', 
           legend_label='PLS Regressor',
           line_width=1.5)
    p1.line(df_test.index+t_correction, y_pred_knn.ravel(), 
           color='#B2675E', 
           legend_label='KNN',
           line_width=1.5)
    p1.line(df_test.index+t_correction, y_pred_rf.ravel(), 
           color = '#644536', 
           legend_label='RF',
           line_width=1.5)
    p1.line(df_test.index+t_correction, y_pred_sarima.predicted_mean,
           color = '#D8DDEF',
           legend_label='Sarima',
           line_width=1.5)
    p1.legend.click_policy = 'hide'

    p2 = Figure(
            x_range = p1.x_range, 
            y_range = p1.y_range,
            width = 500,
            plot_height=400,
            x_axis_type="datetime",  
            title = 'Prediction de la Consommation française' + title2,
            tools=['xwheel_zoom','reset', 'pan'], 
            active_scroll='xwheel_zoom',)
    p2.line(df_test.index, y_test2, 
           color='#21D19F', 
           legend_label='Données Réelles',
           line_width=1.5,
           line_dash='dashed',)
    p2.line(df_test.index, y_pred_pls2.ravel(), 
           color='#EEF1BD', 
           legend_label='PLS Regressor',
           line_width=1.5)
    p2.line(df_test.index, y_pred_knn2.ravel(), 
           color='#B2675E', 
           legend_label='KNN',
           line_width=1.5)
    p2.line(df_test.index, y_pred_rf2.ravel(), 
           color = '#644536', 
           legend_label='RF',
           line_width=1.5)
    p2.line(df_test.index, y_pred_sarima2.predicted_mean,
           color = '#D8DDEF',
           legend_label='Sarima',
           line_width=1.5)
    p2.legend.click_policy = 'hide'

    p3 = Figure(
            x_range = p1.x_range, 
            y_range = p1.y_range,
            width = 500,
            plot_height=400,
            x_axis_type="datetime",  
            title = 'Prediction de la Consommation française' + title3,
            tools=['xwheel_zoom','reset', 'pan'], 
            active_scroll='xwheel_zoom',)
    p3.line(df_test.index, y_test3, 
           color='#21D19F', 
           legend_label='Données Réelles',
           line_width=1.5,
           line_dash='dashed',)
    p3.line(df_test.index, y_pred_pls3.ravel(), 
           color='#EEF1BD', 
           legend_label='PLS Regressor',
           line_width=1.5)
    p3.line(df_test.index, y_pred_knn3.ravel(), 
           color='#B2675E', 
           legend_label='KNN',
           line_width=1.5)
    p3.line(df_test.index, y_pred_rf3.ravel(), 
           color = '#644536', 
           legend_label='RF',
           line_width=1.5)
    p3.line(df_test.index, y_pred_sarima3.predicted_mean,
           color = '#D8DDEF',
           legend_label='Sarima',
           line_width=1.5)
    p3.legend.click_policy = 'hide'
    
    p = Row(p1, p2, p3)
    curdoc().add_root(p)
    curdoc().theme = 'dark_minimal'

    return p


def print_MAPE_score(y_pred_pls, y_pred_knn, y_pred_rf, y_pred_sarima, y_test,
                     y_pred_pls2, y_pred_knn2, y_pred_rf2, y_pred_sarima2, y_test2,
                     y_pred_pls3, y_pred_knn3, y_pred_rf3, y_pred_sarima3, y_test3,):
    #CONSO
    train_score_sarima_1 = mean_absolute_percentage_error(y_true=y_test, y_pred= y_pred_sarima.predicted_mean)
    train_score_rf_1 = mean_absolute_percentage_error(y_true=y_test, y_pred= y_pred_rf)
    train_score_knn_1 = mean_absolute_percentage_error(y_true=y_test, y_pred= y_pred_knn)
    train_score_pls_1 = mean_absolute_percentage_error(y_true=y_test, y_pred= y_pred_pls)
    #METEO
    train_score_sarima_2 = mean_absolute_percentage_error(y_true=y_test2, y_pred= y_pred_sarima2.predicted_mean)
    train_score_rf_2 = mean_absolute_percentage_error(y_true=y_test2, y_pred= y_pred_rf2)
    train_score_knn_2 = mean_absolute_percentage_error(y_true=y_test2, y_pred= y_pred_knn2)
    train_score_pls_2 = mean_absolute_percentage_error(y_true=y_test2, y_pred= y_pred_pls2)
    #ACP
    train_score_sarima_3 = mean_absolute_percentage_error(y_true=y_test3, y_pred= y_pred_sarima3.predicted_mean)
    train_score_rf_3 = mean_absolute_percentage_error(y_true=y_test3, y_pred= y_pred_rf3)
    train_score_knn_3 = mean_absolute_percentage_error(y_true=y_test3, y_pred= y_pred_knn3)
    train_score_pls_3 = mean_absolute_percentage_error(y_true=y_test3, y_pred= y_pred_pls3)

    conso = [train_score_sarima_1/2, train_score_rf_1/2, train_score_knn_1/2, train_score_pls_1/2]
    meteo = [train_score_sarima_2/2, train_score_rf_2/2, train_score_knn_2/2, train_score_pls_2/2]
    acp = [train_score_sarima_3/2, train_score_rf_3/2, train_score_knn_3/2, train_score_pls_3/2]
    modele = ['SARIMA', 'RF', 'KNN', 'PLS']

    df = pd.DataFrame({'Consommation': conso,
                   'Meteo': meteo,
                   'ACP': acp,
                   'modeles' : modele})
    df = df.set_index('modeles')

    return df.style.format("{:.2%}")\
                .applymap(lambda x: 'background-color : #274e13' if x<0.02 else '')\
                .applymap(lambda x: 'background-color : #660000' if x>0.03 else '')\
                .applymap(lambda x: 'background-color : #7f6000' if x<0.03 and x>0.02 else '')\


def plotting_map_pydeck():
    # view (location, zoom level, etc.)
    view = pdk.ViewState(latitude=47.059167, longitude=0.727333, pitch=50, zoom=5)

    # layers
    prod_totale = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['longitude', 'latitude'],
                         get_elevation='prod_totale',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[234, 233, 232, 200],
                         pickable=True,
                         auto_highlight=True)

    nucleaire_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo2', 'la2'],
                         get_elevation='nucleaire_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[252, 141, 89, 255],
                         pickable=True,
                         auto_highlight=True)

    thermique_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo3', 'la3'],
                         get_elevation='thermique_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[213, 62, 79, 255],
                         pickable=True,
                         auto_highlight=True)

    bioenergies_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo4', 'la4'],
                         get_elevation='bioenergies_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[152, 213, 149, 255],
                         pickable=True,
                         auto_highlight=True)

    eolien_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo5', 'la5'],
                         get_elevation='eolien_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[230, 244, 153, 255],
                         pickable=True,
                         auto_highlight=True)

    solaire_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo6', 'la6'],
                         get_elevation='solaire_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[255, 255, 139, 255],
                         pickable=True,
                         auto_highlight=True)

    hydraulique_mw = pdk.Layer('ColumnLayer',
                         data=geo_df,
                         get_position=['lo7', 'la7'],
                         get_elevation='hydraulique_mw',
                         elevation_scale=0.0001,
                         radius=5000,
                         get_fill_color=[50, 136, 189, 255],
                         pickable=True,
                         auto_highlight=True)
    
    

    # render map
    # with no map_style, map goes to default
    main = pdk.Deck(layers= [prod_totale, nucleaire_mw,
                                     thermique_mw, bioenergies_mw,
                                     eolien_mw, solaire_mw, hydraulique_mw],
                            initial_view_state=view)
    return main


    labels = ['Poduction totale', 'Nucleaire', 'Thermique', 'Bioenergies',
              'Eolienne', 'Solaire', 'Hydraulique']
    colors = [(0.84, 0.84, 0.84), (0.85, 0.87, 0), (0.73, 0, 0,), (0.30, 0.78, 0.30),
              (0.67, 1, 0.96), (1, 0.82, 0),(0.27, 0.57, 1,)]
    fig = plt.figure(figsize=(2, 1.5),
                     facecolor=(0.15, 0.15, 0.19))
    patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in zip(labels, colors)]
    fig.legend(patches, labels, loc='center', labelcolor='white' ,frameon=False)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf


def plot_bokeh_histo():
    filieres = ['hydraulique', 'bioenergies', 'eolien', 'solaire', 'nucleaire', 'thermique']

    p1 = Figure(x_range=list(filieres), width=700, height=400,title = '2013', toolbar_location=None)
    p2 = Figure(x_range=list(filieres), width=700, height=400,title = '2014', toolbar_location=None)
    p3 = Figure(x_range=list(filieres), width=700, height=400,title = '2015', toolbar_location=None)
    p4 = Figure(x_range=list(filieres), width=700, height=400,title = '2016', toolbar_location=None)
    p5 = Figure(x_range=list(filieres), width=700, height=400,title = '2017', toolbar_location=None)
    p6 = Figure(x_range=list(filieres), width=700, height=400,title = '2018', toolbar_location=None)
    p7 = Figure(x_range=list(filieres), width=700, height=400,title = '2019', toolbar_location=None)

    index = [0, 1, 2, 3, 4, 5, 6]
    figures = [p1, p2, p3, p4, p5, p6, p7]

    for i, p in zip(index, figures):
        filieres = ['hydraulique', 'bioenergies', 'eolien', 'solaire', 'nucleaire', 'thermique']
        mw = df_histo.iloc[i,:]
        source = ColumnDataSource(data=dict(filieres=filieres, mw=mw))
    
        p.vbar(x='filieres', 
           top='mw', 
           width=0.9,
           source=source,
           legend_field='filieres',
           line_color='white', 
           fill_color=factor_cmap('filieres', palette=Spectral6, factors=filieres))

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.y_range.end = 1
        p.legend.location = "top_center"
        

    tab1 = Panel(child = p1, title = '2013')
    tab2 = Panel(child = p2, title = '2014')
    tab3 = Panel(child = p3, title = '2015')
    tab4 = Panel(child = p4, title = '2016')
    tab5 = Panel(child = p5, title = '2017')
    tab6 = Panel(child = p6, title = '2018')
    tab7 = Panel(child = p7, title = '2019')

    l_tabs = [tab1, tab2, tab3, tab4, tab5, tab6, tab7]    

    tabs = Tabs (tabs = l_tabs)
    curdoc().add_root(tabs)
    curdoc().theme = 'dark_minimal'

    return tabs


def plot_heatmap():
    sns.set(font_scale=0.5)
    df = df_meteo_prod.drop(['latitude', 'longitude', 'altitude'], axis=1)
    g = sns.heatmap(df.corr(), annot = True, center=0, cmap = 'RdBu_r', fmt='.2f')
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right', )
    fig = g.get_figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(6, 5)
    fig.set_facecolor('#272731')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    return fig



def plot_scatter_bokeh_conso():
    source = ColumnDataSource(df_meteo_prod)

    p1 = Figure(width=500, plot_height=400,
                title = 'Température / Consommation_mw')

    p1.circle('consommation_mw', 'temperature_c', source = source, color="lightskyblue", alpha=0.7)

    p2 = Figure(width=500, plot_height=400,
                title ='Point de Rosée / Consommation_mw')

    p2.circle('consommation_mw', 'point_de_rosee', source = source, color="deepskyblue", alpha=0.7)

    p = Row(p1, p2)
    curdoc().add_root(p)
    curdoc().theme = 'dark_minimal'
    return p


def plot_scatter_bokeh_autre():
    source = ColumnDataSource(df_meteo_prod)

    p1 = Figure(width=500, plot_height=400,
                title = 'Humidité / Production solaire ')

    p1.circle('solaire_mw', 'humidite', source = source, color="darkorange", alpha=0.7)

    p2 = Figure(width=500, plot_height=400,
                title ='Vitesse du Vent Moyen / Production Eolienne')

    p2.circle('eolien_mw', 'vitesse_du_vent_moyen', source = source, color="deepskyblue", alpha=0.7)

    p = Row(p1, p2)
    curdoc().add_root(p)
    curdoc().theme = 'dark_minimal'
    return p


def ACP_graph1():
     conso = pd.read_csv('ener_conso_full.csv',parse_dates=[0], index_col=0, squeeze=True)
     meteo = pd.read_csv('clean_data_meteo_3h.csv')
     conso = conso.resample('3H').sum()
     meteo['date'] = pd.to_datetime(meteo['date'])
     meteo = meteo.set_index('date')
     meteo = meteo['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
     conso = conso['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
     df=pd.merge(conso, meteo, right_index = True,left_index = True)
     df = df.drop(['regioncode', 'latitude', 'longitude', 'altitude'], axis=1)
     df = df.drop('consommation_mw', axis=1)
     sc = StandardScaler()
     Z = sc.fit_transform(df)
     acp = PCA()
     acp.fit_transform(Z)
     v =  acp.explained_variance_
     fig, ax = plt.subplots()
     plt.plot(np.arange(1, len(v)+1), v)
     plt.xlabel('Nombre de facteur')
     plt.ylabel('Valeurs propres')
     return fig


def ACP_graph2():
     conso = pd.read_csv('ener_conso_full.csv',parse_dates=[0], index_col=0, squeeze=True)
     meteo = pd.read_csv('clean_data_meteo_3h.csv')
     conso = conso.resample('3H').sum()
     meteo['date'] = pd.to_datetime(meteo['date'])
     meteo = meteo.set_index('date')
     meteo = meteo['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
     conso = conso['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
     df=pd.merge(conso, meteo, right_index = True,left_index = True)
     df = df.drop(['regioncode', 'latitude', 'longitude', 'altitude'], axis=1)
     df = df.drop('consommation_mw', axis=1)
     sc = StandardScaler()
     Z = sc.fit_transform(df)
     acp = PCA()
     acp.fit_transform(Z)
     r = acp.explained_variance_ratio_
     fig, ax = plt.subplots()
     plt.plot(np.arange(1, len(r)+1), np.cumsum(r))
     plt.xlabel('Nombre de facteur')
     plt.ylabel('Valeurs propres')
     return fig


def visu_ACP():
    conso = pd.read_csv('ener_conso_full.csv',parse_dates=[0], index_col=0, squeeze=True)
    meteo = pd.read_csv('clean_data_meteo_3h.csv')
    conso = conso.resample('3H').sum()
    meteo['date'] = pd.to_datetime(meteo['date'])
    meteo = meteo.set_index('date')
    meteo = meteo['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
    conso = conso['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
    df=pd.merge(conso, meteo, right_index = True,left_index = True)
    df = df.drop(['regioncode', 'latitude', 'longitude', 'altitude'], axis=1)
    target = df.consommation_mw
    df = df.drop('consommation_mw', axis=1)
    sc = StandardScaler()
    Z = sc.fit_transform(df)
    acp = PCA()
    coord = acp.fit_transform(Z)
    PCA_mat = pd.DataFrame({'AXE 1': coord[:, 0], 'AXE 2': coord[:, 1], 'target': target})
    sqrt_eigval = np.sqrt(acp.explained_variance_)
    corvar = np.zeros((10, 10))
    for k in range(10):
        corvar[:, k] = acp.components_[:, k] * sqrt_eigval[k]
    fig, axes = plt.subplots(figsize=(3, 3))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    for j in range(10):
        plt.annotate(df.columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
        plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.6, alpha=0.4, head_width=0.03, color='b')
    plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
    cercle = plt.Circle((0, 0), 0.99, color='#16E4CA', fill=False)
    axes.add_artist(cercle)
    plt.xlabel('AXE 1')
    plt.ylabel('AXE 2')
    plt.show()


def plot_heatmap2():
    conso = pd.read_csv('ener_conso_full.csv',parse_dates=[0], index_col=0, squeeze=True)
    meteo = pd.read_csv('clean_data_meteo_3h.csv')
    conso = conso.resample('3H').sum()
    meteo['date'] = pd.to_datetime(meteo['date'])
    meteo = meteo.set_index('date')
    meteo = meteo['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
    conso = conso['2012-12-31 21:00:00+00:00':'2019-12-31 21:00:00+00:00']
    df=pd.merge(conso, meteo, right_index = True,left_index = True)
    df = df.drop(['regioncode', 'latitude', 'longitude', 'altitude'], axis=1)
    target = df.consommation_mw
    df = df.drop('consommation_mw', axis=1)
    sc = StandardScaler()
    Z = sc.fit_transform(df)
    acp = PCA()
    coord = acp.fit_transform(Z)
    PCA_mat = pd.DataFrame({'AXE 1': acp.components_[:, 0], 'AXE 2': acp.components_[:, 1]})
    sns.set(font_scale=0.5)
    g = sns.heatmap(PCA_mat, annot=True, cmap='Blues',fmt='.2f')
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right', )
    g.set_yticklabels(['rafales_sur_une_periode', 'variation_de_pression_en_3_heures',
    'precipitations_dans_les_3_dernieres_heures', 'pression_station',
    'pression_au_niveau_mer', 'direction_du_vent_moyen',
    'vitesse_du_vent_moyen', 'humidite', 'point_de_rosee', 'temperature_c'], rotation= 360)

    fig = g.get_figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(2, 2)
    fig.set_facecolor('#272731')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    return fig

########################################

def set_introduction():
    st.title('Introduction')
    st.write(introduction_1)
    col1, col2, col3, col4 = st.columns([1,3,3,1])   
    with col1:
        st.write("")

    with col2:
        st.image(['Image/image1.jpg'], width=420) 

    with col3:
        st.image(['Image/image2.jpg'], width=370) 
    
    with col4:
        st.write("")
    st.write(introduction_2)

    link = '[Données Consos, Prods, Brutes](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure)'
    link2 = '[Données Météos Brutes](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/?sort=date)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(link2, unsafe_allow_html=True)


def set_visualisation():
    st.title('Heatmap : Coefficient de Corrélation')
    st.write('''Après avoir merged les datasets des données météorologique 
    & des données de production et de consommation,
    nous avons réalisé une Heatmap pour nous rendre compte des corrélations entre les variables :''')
    st.pyplot(plot_heatmap())

    st.write('''Nous pouvons d'ores et déjà remarqué une forte corrélation de la consommation avec certaines variables météorologiques 
    particulièrement les points de rosée et la température.
    Il existe aussi des corrélations intéressantes qui n'ont pas été utilisé pour nos modélisations mais qui peuvent servir d'ouverture à 
    notre projet. Notamment la corrélation entre **la vitesse du vent moyen et la production électrique via les éolliennes**, ou encore, entre
    **l'humidité et la production solaire**.''')
    
    st.subheader('Corrélation température & point de rosée avec la consommation :')

    st.bokeh_chart(plot_scatter_bokeh_conso())

    st.subheader('Corrélation production solaire & humidté / production éolienne & vitesse du vent moyen :')

    st.bokeh_chart(plot_scatter_bokeh_autre())

    st.title('Présentation du Mix Energétique')
    st.write('''Ci-dessous figure un histogramme interactif permettant de se rendre compte de l'évolution du mix énergétique en france :''')

    st.bokeh_chart(plot_bokeh_histo())


    st.text('''Cette carte représente la production par filière : \n
    ''')

    st.pydeck_chart(plotting_map_pydeck())


def set_modelisation_conso():
    
    #### TITRE & INTRO ####
    st.title('Modelisation - Machine Learning')

    st.write(modelisation_text1)

    #### Choix Granularité 
    modeles = st.selectbox('Choisissez la Granularité à afficher :', 
                            ('Mois', 'Jours', '3 heures'))
    
    #### Nombre de temps précédents à prendre en compte :
    t = st.slider("Temps précédents  Knn, RF & PLS pour la modélisation avec la consommation) :", 2, 5, 3, 1)
    t1 = st.slider("Temps précédents pour SARIMA :", 0, 4, 2, 1)


    #### RAJOUTER UN SELECTEUR POUR CHOISIR LE DATA SET ET AFFICHER 9 GRAPHS ####

    if modeles == 'Mois':
        df_conso = conso_mois
        df_meteo = df_meteo_mois
        df_acp = df_acp_mois
        param_pls = param_modele_conso_mois[t]['PLS']
        param_knn = param_modele_conso_mois[t]['KNN']
        param_rf = param_modele_conso_mois[t]['RF']
        
    elif modeles == 'Jours':
        df_conso = conso_jours['2018-01-01':'2019-12-31']
        df_meteo = df_meteo_jours['2018-01-01':'2019-12-31']
        df_acp = df_acp_jours['2018-01-01':'2019-12-31']
        param_pls = param_modele_conso_jours[t]['PLS']
        param_knn = param_modele_conso_jours[t]['KNN']
        param_rf = param_modele_conso_jours[t]['RF']
        
    elif modeles == '3 heures':
        df_conso = conso_3h['2019-01-01':'2019-12-31']
        df_meteo = df_meteo_3h['2019-01-01':'2019-12-31']
        df_acp = df_acp_3h['2019-01-01':'2019-12-31']
        param_pls = param_modele_conso_mois[t]['PLS']
        param_knn = param_modele_conso_mois[t]['KNN']
        param_rf = param_modele_conso_mois[t]['RF']
        
    

    df_train, df_test = train_test_split(df_conso,test_size = 0.2, shuffle = False)
    X_train, y_train = create_array(df_train, t)
    X_test, y_test = create_array(df_test, t)


    pls = PLSRegression(**param_pls).fit(X_train, y_train)
    knn = KNeighborsRegressor(**param_knn).fit(X_train, y_train)
    rf = RandomForestRegressor(**param_rf).fit(X_train, y_train)
    sarima = sm.tsa.SARIMAX(df_conso, order= param_modele_SARIMA_Conso[t1][modeles]['order'], 
                                seasonal_order=param_modele_SARIMA_Conso[t1][modeles]['seasonal_order']).fit()

    y_pred_pls = pls.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_sarima = sarima.get_prediction(start=df_test.index[t], dynamic=False)
    
    #st.header('Modelisation à partir de la consommation des temps précédents')    
    
    #### PARTIE 2 : Modelisation avec Météo Uniquement

    X_train2, X_test2, y_train2, y_test2 = train_test_split(df_meteo, df_conso, test_size=0.2, shuffle = False)
    scaler = StandardScaler()
    
    X_train_scaled2 = scaler.fit_transform(X_train2)
    X_test_scaled2 = scaler.transform(X_test2)

    pls = PLSRegression(**param_modele_meteo[modeles]['PLS']).fit(X_train_scaled2, y_train2)
    knn = KNeighborsRegressor(**param_modele_meteo[modeles]['KNN']).fit(X_train_scaled2, y_train2)
    rf = RandomForestRegressor(**param_modele_meteo[modeles]['RF']).fit(X_train_scaled2, y_train2)
    sarima = sm.tsa.SARIMAX(df_conso, exog=df_meteo, 
                                order= param_modele_SARIMA_meteo[t1][modeles]['order'], 
                                seasonal_order=param_modele_SARIMA_meteo[t1][modeles]['seasonal_order']).fit()
    
    y_pred_pls2 = pls.predict(X_test_scaled2)
    y_pred_rf2 = rf.predict(X_test_scaled2)
    y_pred_knn2 = knn.predict(X_test_scaled2)
    y_pred_sarima2 = sarima.get_prediction(start=df_test.index[0], dynamic=False)

    
    #st.header('Modelisation avec uniquement la météo')
    
    #### PARTIE 3 : Modelisation avec Météo Uniquement

    X_train3, X_test3, y_train3, y_test3 = train_test_split(df_acp, df_conso, test_size=0.2, shuffle = False)
    scaler = StandardScaler()
    
    X_train_scaled3 = scaler.fit_transform(X_train3)
    X_test_scaled3 = scaler.transform(X_test3)

    pls = PLSRegression(**param_modele_meteoacp[modeles]['PLS']).fit(X_train_scaled3, y_train3)
    knn = KNeighborsRegressor(**param_modele_meteoacp[modeles]['KNN']).fit(X_train_scaled3, y_train3)
    rf = RandomForestRegressor(**param_modele_meteoacp[modeles]['RF']).fit(X_train_scaled3, y_train3)
    sarima = sm.tsa.SARIMAX(df_conso, exog=df_acp, 
                                order= param_modele_SARIMA_meteoacp[t1][modeles]['order'], 
                                seasonal_order=param_modele_SARIMA_meteoacp[t1][modeles]['seasonal_order']).fit()
    
    y_pred_pls3 = pls.predict(X_test_scaled3)
    y_pred_rf3 = rf.predict(X_test_scaled3)
    y_pred_knn3 = knn.predict(X_test_scaled3)
    y_pred_sarima3 = sarima.get_prediction(start=df_test.index[0], dynamic=False)

    #st.header("Modelisation avec météo corrigé de l'analyse ACP")
    
    
    st.bokeh_chart(bokeh_link_model('- consommation', y_pred_pls, y_pred_knn, y_pred_rf, y_pred_sarima, y_test, df_test,
                     ' - meteo', y_pred_pls2, y_pred_knn2, y_pred_rf2, y_pred_sarima2, y_test2,
                     ' - ACP', y_pred_pls3, y_pred_knn3, y_pred_rf3, y_pred_sarima3, y_test3, t, modeles), use_container_width=False)

    st.subheader('MAPE des modèles :')
    st.write('''**MAPE** :  mean absolute percentage error, c'est à dire la moyenne des écarts en valeur absolue par rapport aux valeurs observées. 
    C’est donc un pourcentage et par conséquent un indicateur pratique de comparaison. **Les valeurs sont en pourcentage**.''')

    st.dataframe(print_MAPE_score(y_pred_pls, y_pred_knn, y_pred_rf, y_pred_sarima, y_test,
                      y_pred_pls2, y_pred_knn2, y_pred_rf2, y_pred_sarima2, y_test2,
                      y_pred_pls3, y_pred_knn3, y_pred_rf3, y_pred_sarima3, y_test3,))
    

def set_conclusion():
    st.title('Conclusion')
    st.write(conclusion_1)
    col1, col2, col3 = st.columns((1,6,1))
    with col2:
        st.image('Image/image3.jpg', width=600)
    

def set_exploration():
    #### TITRE & INTRO ####
    st.title(Explication_données)
    st.write(Explication_données2)
    st.subheader(Explication_données3)
    st.write(Explication_données4)
    st.subheader(Explication_données5)
    st.write(Explication_données6)
    st.write(Dataclean)
    
    image = Image.open('Image/Représentation_des_NaN.png')
    st.image(image, caption='Représentation_des_NaN')
    st.write(Dataclean3)
    st.write(Dataclean4)
    st.subheader(Dataclean5)
    
    st.write('''Dataset clean par année (Aperçu)''')
    data_clean_by_year=pd.read_csv('data_clean_by_year.csv')
    prod1=data_clean_by_year.head(5)
    st.dataframe(prod1.style.format("{:.0f}"))
    
    st.write('''Dataset clean par mois (Aperçu)''')
    data_clean_by_month=pd.read_csv('data_clean_by_month.csv')
    prod2=data_clean_by_month.head(5)
    st.dataframe(prod2.style.format("{:.0f}"))

    st.write('''Dataset clean par jours (Aperçu)''')
    data_clean_by_days=pd.read_csv('data_clean_by_days.csv')
    prod3=data_clean_by_days.head(5)
    st.dataframe(prod3.style.format("{:.0f}"))

    st.subheader(Dataclean6)
    st.write(Dataclean7)
    st.write('Pour effectuer la concaténation nous regroupons les données énergétiques par jour et par région:')

    st.code(code,language='python')

    st.write('Nous procédons à la même opération sur le dataset clean de la météo')
    st.write(Dataclean7)
    
    st.subheader(Dataclean8)
    st.write(Dataclean9)
    st.dataframe(data_meteo_prod_region.style.format("{:.9}"))
    st.write(Dataclean10)

    image2 = Image.open('Image/Explications_variables2.png')
    st.image(image2, caption='Explications des variables')


def set_ACP():
    st.title(acp)
    st.write(acp2)
    st.write(acp3)
    st.code(code2,language='python')
    st.subheader(acp4)
    st.write(acp5)
    st.write(acp6)
    col1, col2 = st.columns([1, 1])
    col1.subheader("Graphique des valeurs propres")
    col1.pyplot(ACP_graph1())
    col2.subheader("Cumul de variance restituée")
    col2.pyplot(ACP_graph2())
    st.write(acp7)
    st.subheader(acp8)
    st.pyplot(visu_ACP())
    st.write(acp9)
    st.write(acp10)
    st.pyplot(plot_heatmap2())
    st.write(acp11)
    st.write(acp12)
    st.dataframe(df_acp.style.format("{:.9}"))
