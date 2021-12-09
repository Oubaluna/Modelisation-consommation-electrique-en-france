import pandas as pd
import streamlit as st
from pynergized_fonctions import *

### MAIN FILE APPLICATION STREAMLIT ###
st.set_page_config(layout="wide")
### Suppression de l'affichage des warning :
st.set_option('deprecation.showPyplotGlobalUse', False)
## Instanciation d'un menu ##
st.sidebar.header('Pynergized')
st.sidebar.markdown('Consommation Electrique Française | Données : 2013 à 2019')
menu = st.sidebar.radio("Pynergized :", ("Introduction", 
                                         "Exploration des données",
                                         "Visualisations",
                                         "Etude via ACP",
                                         "Modelisation", 
                                         "Conclusion"))


if menu == "Introduction":
    set_introduction()
if menu == "Exploration des données":
    set_exploration()
elif menu == "Visualisations":
    set_visualisation()
elif menu == "Etude via ACP":
    set_ACP()
elif menu == "Modelisation":
    set_modelisation_conso()
elif menu == "Conclusion":
    set_conclusion()


