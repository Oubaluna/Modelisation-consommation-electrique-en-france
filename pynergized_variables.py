import pandas as pd
import numpy as np

###############################
###############################
## CHARGEMENT JEU DE DONNEES ##
###############################
###############################

# => CONSO
conso = pd.read_csv('ener_conso_full.csv', 
                 parse_dates=[0], 
                 index_col=0, 
                 squeeze=True)

# => METEO
df_meteo = pd.read_csv('clean_data\clean_data_meteo_3h.csv',index_col=0)
to_drop = ['regioncode','latitude', 'longitude', 'altitude']
df_meteo = df_meteo.drop(to_drop, axis=1)
df_meteo = df_meteo['2012-12-31':'2020-01-01']
df_meteo.index = pd.to_datetime(df_meteo.index)

# => ACP
df_acp = pd.read_csv('clean_data/df_acp.csv',index_col=0)
df_acp.index = pd.to_datetime(df_acp.index)


## CREATION DE 3 GRANULARITES ##

# => CONSO
conso_mois= conso.resample('M').sum()
conso_jours = conso.resample('D').sum()
conso_3h = conso.resample('3h').sum()


# => METEO
function_to_apply = {
    'rafales_sur_une_periode' : 'max',
    'variation_de_pression_en_3_heures' : 'sum',
    'precipitations_dans_les_3_dernieres_heures' : 'sum',
    'pression_station' : 'mean',
    'pression_au_niveau_mer' : 'mean',
    'direction_du_vent_moyen' : 'median', #médiane de la direction sur la journée
    'vitesse_du_vent_moyen' : 'mean',
    'humidite' : 'mean',
    'point_de_rosee' : 'mean',
    'temperature_c' : 'mean',
}


df_meteo_mois= df_meteo.resample('M').agg(function_to_apply)
df_meteo_jours = df_meteo.resample('D').agg(function_to_apply)
df_meteo_3h = df_meteo.resample('3h').agg(function_to_apply)

# => ACP
function_to_apply = {
    'temperature_c' : 'mean',
    'humidite' : 'mean',
    'pression_station' : 'mean',
    'pression_au_niveau_mer' : 'mean',
    'precipitations_dans_les_3_dernieres_heures' : 'sum',
}


df_acp_mois= df_acp.resample('M').agg(function_to_apply)
df_acp_jours = df_acp.resample('D').agg(function_to_apply)
df_acp_3h = df_acp.resample('3h').agg(function_to_apply)


# => Pour Pydeck
geo_df = pd.read_csv('clean_data\geo_df.csv')

# => Pour histo
df_histo = pd.read_csv('clean_data/filiere_histo.csv')

# => Pour heatmap
df_meteo_prod = pd.read_csv('clean_data/data_meteo_prod.csv',index_col=0)

###########################################################################
###########################################################################
## DICTIONNAIRES DE PARAMETRE - OPTIMISATION SUR NOTEBOOK VIA GRIDSEARCH ##
## Pour éviter un trop long temps de chargement report des meilleurs paramètres ici ##
## Pour PLS RF KNN : optimisation via GridSearch
## Pour Sarima : Différenciation pour stationnariser / test via AIC

param_modele_conso_mois = {
     2 : 
            {'PLS': {'n_components': 2, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 50}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 3, 'weights': 'distance'}},

     3 : 
             {'PLS': {'n_components': 3, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 100}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 4, 'weights': 'distance'}},

     4 : 
             {'PLS': {'n_components': 3, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 250}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}},
     
     5 : 
             {'PLS': {'n_components': 3, 'scale': False}, 
             'RF': {'max_features': 'log2', 'min_samples_leaf': 1, 'n_estimators': 100}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}},

}

param_modele_conso_jours = {
     2 : 
            {'PLS': {'n_components': 2, 'scale': False}, 
             'RF': {'max_features': 'log2', 'min_samples_leaf': 3, 'n_estimators': 500}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 14, 'weights': 'distance'}},

     3 : 
             {'PLS': {'n_components': 3, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 100}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 6, 'weights': 'distance'}},
     4 : 
             {'PLS': {'n_components': 4, 'scale': False}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 1000}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 6, 'weights': 'distance'}},
     5 : 
             {'PLS': {'n_components': 5, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 250}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 5, 'weights': 'distance'}},
}

param_modele_conso_3h = {
     2 : 
            {'PLS': {'n_components': 2, 'scale': False}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 250}, 
             'KNN': {'metric': 'chebyshev', 'n_neighbors': 25, 'weights': 'uniform'}},

     3 : 
             {'PLS': {'n_components': 3, 'scale': True}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 1000}, 
             'KNN': {'metric': 'chebyshev', 'n_neighbors': 10, 'weights': 'distance'}},

     4 : 
             {'PLS': {'n_components': 4, 'scale': False}, 
             'RF': {'max_features': 'log2', 'min_samples_leaf': 1, 'n_estimators': 1000}, 
             'KNN': {'metric': 'chebyshev', 'n_neighbors': 6, 'weights': 'distance'}},
     
     5 : 
             {'PLS': {'n_components': 5, 'scale': True}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 1000}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 4, 'weights': 'distance'}},
}

param_modele_SARIMA_Conso = {
        0 :    {'Mois': {'order':(0,1,0) ,'seasonal_order' : (0, 1, 1, 12)}, 
                'Jours': {'order':(0,1,0) ,'seasonal_order' : (1, 1, 1, 7)},  
                '3 heures': {'order':(0,1,0) ,'seasonal_order' : (1, 0, 1, 8)},},
        1 :    {'Mois': {'order':(1,1,0) ,'seasonal_order' : (0, 1, 1, 12)}, 
                'Jours': {'order':(1,1,0) ,'seasonal_order' : (1, 1, 1, 7)},  
                '3 heures': {'order':(1,1,0) ,'seasonal_order' : (1, 0, 1, 8)},},
        2 :    {'Mois': {'order':(2,1,0) ,'seasonal_order' : (0, 1, 1, 12)}, 
                'Jours': {'order':(2,1,0) ,'seasonal_order' : (1, 1, 1, 7)},  
                '3 heures': {'order':(2,1,0) ,'seasonal_order' : (1, 0, 1, 8)},}, 
        3 :    {'Mois': {'order':(3,1,0) ,'seasonal_order' : (0, 1, 1, 12)}, 
                'Jours': {'order':(3,1,0) ,'seasonal_order' : (1, 1, 1, 7)},  
                '3 heures': {'order':(3,1,0) ,'seasonal_order' : (1, 0, 1, 8)},}, 
        4 :    {'Mois': {'order':(4,1,0) ,'seasonal_order' : (0, 1, 1, 12)}, 
                'Jours': {'order':(4,1,0) ,'seasonal_order' : (1, 1, 1, 7)},  
                '3 heures': {'order':(4,1,0) ,'seasonal_order' : (1, 0, 1, 8)},},  
          }

##### METEO #####

param_modele_meteo = {
     'Mois' : 
            {'PLS': {'n_components': 1, 'scale': False}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 250}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}},

     'Jours' : 
             {'PLS': {'n_components': 9, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 5, 'n_estimators': 1000}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'distance'}},

     '3 heures' : 
             {'PLS': {'n_components': 2, 'scale': False}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 10}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 33, 'weights': 'distance'}},
     
}

param_modele_SARIMA_meteo = {
        0: {'Mois': {'order':(0, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(0, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(0, 1, 1),'seasonal_order' : (1, 1, 1, 8)},}, 
        1: {'Mois': {'order':(1, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(1, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(1, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        2: {'Mois': {'order':(2, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(2, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(2, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        3: {'Mois': {'order':(3, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(3, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(3, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        4: {'Mois': {'order':(4, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(4, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(4, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
          }

##### METEO ACP #####

param_modele_meteoacp = {
     'Mois' : 
            {'PLS': {'n_components': 1, 'scale': True}, 
             'RF': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 50}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}},

     'Jours' : 
             {'PLS': {'n_components': 3, 'scale': False}, 
             'RF': {'max_features': 'auto', 'min_samples_leaf': 5, 'n_estimators': 500}, 
             'KNN': {'metric': 'minkowski', 'n_neighbors': 18, 'weights': 'distance'}},

     '3 heures' : 
             {'PLS': {'n_components': 3, 'scale': True}, 
             'RF': {'max_features': 'log2', 'min_samples_leaf': 3, 'n_estimators': 100}, 
             'KNN': {'metric': 'manhattan', 'n_neighbors': 40, 'weights': 'uniform'}},
     
}

param_modele_SARIMA_meteoacp = {
        0 : {'Mois': {'order':(1, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(1, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(1, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        1 : {'Mois': {'order':(1, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(1, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(1, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        2 : {'Mois': {'order':(2, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(2, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(2, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        3 : {'Mois': {'order':(3, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(3, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(3, 1, 1),'seasonal_order' : (1, 1, 1, 8)},},
        4 : {'Mois': {'order':(4, 1, 0) ,'seasonal_order' : (0, 1, 1, 12)}, 
          'Jours': {'order':(4, 1, 0) ,'seasonal_order' : (1, 1, 1, 7)},  
          '3 heures': {'order':(4, 1, 1),'seasonal_order' : (1, 1, 1, 8)},}, 
          }

#########################################################
#########################################################
#####################VARIABLE TEXTE######################
#########################################################
#########################################################

modelisation_text1 = '''
Notre objectif était simple : prédire la consommation le plus précisément possible avec ou sans la consommation des temps précédents. 

En effet, nous estimons que pouvoir prédire la consommation électrique sans utiliser les données de consommation elle-même peut répondre à une
problématique métier intéressante en cas d'indisponibilité des données. 

Après plusieurs tests sur des modèles différents, nous nous sommes cantonné à comparer 4 modèles : 
- SARIMAX : Séries temporelles
- K-Nearest Neighbours
- Random Forest
- PLS Regressor 

- **SARIMAX : Séries temporelles**

Un modèle SARIMA s'écrit sous la forme  SARIMA(p,d,q)(P,D,Q)k
La termes en  d  et  D  correspondent aux degrés de différenciation utilisés pour stationnariser la série temporelle.
La seconde parenthèse contenant les termes en majuscule correspond à la partie saisonnière de la  SARIMA et  k  indique la saisonnalité utilisée.
Pour chaque granularité un test par différenciation a été effectué.

- **K-Nearest Neighbours**

L’algorithme de K-Nearest Neighbours est un algorithme d’apprentissage supervisé que l'on peut utiliser dans un problème de régression ou de classification.
Pour effectuer une prédiction, l’algorithme K-NN va estimer une nouvelle donnée en regardant quelles sont les k données voisines les plus proches 
(d’où le nom de l’algorithme). Le seul paramètre à fixer est k, le nombre de voisins à considérer.

- **Random Forest**

Comme chacun le sait,une forêt est constituée d’arbres. Il en va de même pour l’algorithme des forêts aléatoires.
Son fonctionnement consiste à apprendre en parallèle sur plusieurs arbres de décisions
construits aléatoirement et entrain´es sur des sous-ensembles contenant des données différentes.
Chaque arbre propose alors une pr´ediction et la production finale consiste à réaliser la
moyenne de toutes les prédictions.
La complexité dans l’utilisation de cet algorithme est de trouver le bon nombre d’arbres à utiliser pouvant aller jusqu’à plusieurs centaines.

- **PLS Regressor**

La régression des moindres carrés partiels a été inventée en 1983 par Svante Wold et son père Herman Wold ; on utilise fréquemment l'abréviation anglaise régression PLS 
(« Partial Least Squares regression » et/ou « Projection to Latent Structure »). La régression PLS maximise la variance des prédicteurs (Xi) = X et maximise la corrélation 
entre X et la variable à expliquer Y. Cet algorithme emprunte sa démarche à la fois à l'analyse en composantes principales (ACP) et à la régression. 
Plus précisément, la régression PLS cherche des composantes, appelées variables latentes, liées à X et à Y, servant à exprimer la régression de Y sur ces variables 
et finalement de Y sur X.

**Afin de comparer ces trois modèles nous avons construit les graphismes Bokeh Interactifs suivants** :

*Lors de changement de paramètre : être patient, les modèles calculent.*
'''

introduction_1 = '''
©2021, Pape, Léo, Timothée - Promotion Octobre 2021 - Cursus DA \n

L'objectif de ce projet est de réaliser une prédiction de la consommation électrique 
française à partir des deux jeux de données suivants. \n
Actuellement, il urge d’optimiser la consommation d’énergie et d’accélérer 
la transition énergétique. L’apprentissage automatique se positionne comme une 
alternative pour répondre à ces défis. 
Alors, grâce à l’analyse des données obtenues des productions antérieures, le machine 
learning est capable de répondre à des problématiques tels que : la prédiction de 
la consommation pour rationaliser la production, trouver un équilibre des charges de 
consommation d’énergie.\n
'''

introduction_2='''
- **Du point de vue technique** :\n
Une meilleure gestion des données énergétiques ouvre de nouvelles possibilités dans
la collecte et l’analyse de ces données, ainsi que dans l’obtention de prévisions plus
précises.\n
- **Du point de vue économique** :\n
Les entreprises et les particuliers peuvent convertir la quantité d’énergie consommée\n
en valeur monétaire, et donc estimer la facture énergétique et prendre des décisions à\n
partir de ces données.\n
- **Du point de vue scientifique** :\n
En sachant non seulement combien d’énergie nous allons consommer, mais aussi comment
et pourquoi, nous pouvons changer nos habitudes sans affecter notre productivité ou
notre qualité de vie.\n
les données sont en téléchargement libre aux adresses ci-dessous :
'''

conclusion_1 = '''
La découverte principale de ce projet réside en des résultats assez surprenants concernant
la prédiction de la consommation à partir des données météorologiques uniquement. \n
Sans surprise, lorsque l’on prend uniquement les consommations des instants précédents
le modèle SARIMA s’en sort le mieux.\n
Les possibilités d’ouverture de sujets supplémentaires à creuser que permettent ce projet
serait : \n
1. Définir une puissance de production maximale en mega watt et calculer un **taux
de charge maximal** cohérent pour faire une analyse du **risque de blackout**.\n
2. Effectuer un **voting regressor** \n
3. Prédiction de la **production hydraulique** à partir de la météo (certaines variables
sur la heatmap sont fortement corrélées à la production hydraulique) \n
4. **Utiliser du deep learning** (en dehors de notre cursus malheureusement) et comparer
les résultats avec les modèles de machine learning classique'''
#image2 = 'image2'

#######EXPLICATION DES DONNEES

Explication_données = '''Exploration des données'''

Explication_données2 = '''
Nous avons deux types de datasets : les données de la consommation et de la météo. Dans cette partie, nous allons expliquer les variables et le prepocessing effectué pour obtenir des données exploitables. Nous
verrons dans une première partie les données utilisées, puis les grandes étapes de notre datacleaning.'''

Explication_données3 = '''A\ Données énergétiques et de la météo'''

Explication_données4 = '''
La source des données de la consommation est celle de L’ODRE (Open Data Réseaux Energies). Nous avons accès à 
toutes les informations de consommation et de production par filière jour par jour (toutes les 1/2 heure) depuis 2013.
Ce jeu de données, rafraîchi une fois par jour, présente les données consolidées depuis janvier 2021 et définitives 
(de janvier 2012 à décembre 2019) issues de l’application éCO2mix. Elles sont élaborées à partir des comptages et 
complétées par des forfaits. Les données sont dites consolidées lorsqu’elles ont été vérifiées et complétées 
(livraison en milieu de M + 1). Elles deviennent définitives lorsque tous les partenaires ont transmis et 
vérifié l’ensemble de leurs comptages (livraison deuxième semestre A + 1).

Concernant la météo, les données proviennent des observations issues des messages internationaux d’observation
en surface (SYNOP) circulant sur le système mondial de télécommunication (SMT) de l’Organisation Météorologique
Mondiale (OMM). Les paramètres atmosphériques mesurés (température, humidité, direction et force du vent,
pression atmosphérique, hauteur de précipitations) ou observés (temps sensible, description des nuages,
visibilité) depuis la surface terrestre. Selon instrumentation et spécificités locales, d’autres paramètres
peuvent être disponibles (hauteur de neige,  état du sol, etc.)'''

Explication_données5 = ''' B\ Data cleanning : données énergétiques'''     
        
Explication_données6 = ''' 
Dans un premier temps nous avons exploré les données pour regarder les variables utilisables avec notre
problématique. '''

Dataclean= ''' 
Cependant, Nous avons observé que le Dataframe contient beaucoup de valeurs manquantes (cf.ci-dessous).'''

Dataclean2 = '''
Comme nous pouvons le constater, les parties blanches représentes les données manquantes. Nous avons 
des variables avec peu d'information. Nous décidons de conserver les données ayant moins de 90% de NaN '''

Dataclean3 = ''' 
Les données sont récoltées par région, ce qui explique le manque de données car toutes les régions ne
disposent pas des mêmes ressources en énergie.De ce fait nous effectuons un remplacement des NaN par 0.'''

Dataclean4 = ''' 
Enfin, pour faciliter la lecture nous avons créé une nouvelle colonne avec la porduction totale et
changer la type de la "date" en datetime.'''


Dataclean5 = ''' 
C\ Création de 3 dataframe par temporalité : Année, Mois, Jours
NB : suppression de 2021 pour travailler sur un bloc temporel sans 'trou' '''

data_clean_by_year=pd.read_csv('clean_data/data_clean_by_year.csv')
prod1=data_clean_by_year.head(5)

Dataclean6= ''' D\ Cleaning des données météo et création d'un dataset clean avec la consommation et la météo'''

Dataclean7=''' 
Nous avons procédé aux mêmes étapes sur le dataset de la météo pour le préprocessing:
uniformisation de colonnes et suppréssion des NaN. 

Nous Conservons uniquement les colonnes météo utiles pour notre étude :
- Température
- Pression
- Pluviométrie
- Vents (vitesse & direction & Rafales)'''

code ='''data_clean['date'] =  pd.to_datetime(data_clean['date'])
    data_clean['year'] = data_clean['date'].apply(lambda x : x.year)
    data_clean['month'] = data_clean['date'].apply(lambda x : x.month)
    data_clean['days'] = data_clean['date'].apply(lambda x : x.day)
    data_clean_jour = data_clean.groupby(['year', 'month', 'days', 'region'], as_index = False).sum()
    data_clean_jour = data_clean_jour.drop('code_insee_region', axis=1)
    data_clean_jour['date'] = data_clean_jour['year'].astype(str) + '-' + data_clean_jour['month'].astype(str) + '-' + data_clean_jour['days'].astype(str)
    data_clean_jour['date'] =  pd.to_datetime(data_clean_jour['date'])
    data_clean_jour = data_clean_jour.drop(['year', 'month', 'days'], axis=1)
    data_clean_jour'''

Dataclean8='''
E\ MERGE avec la méthode INNER pour obtenir un dataset clean de NaN en utilisant la journée et la région'''
Dataclean9=''' Pour finir nous fusionnons l'ensemble de nos data clean par région et par jours avec un pas de temps de 1 jour. '''
data_meteo_prod_region=pd.read_csv('clean_data/data_meteo_prod_region.csv')
dataframeclean=data_meteo_prod_region.head(10)

Dataclean10 = '''Ainsi nous avons un dataset clean avec l'ensemble des données énéergétiques et météorologiques pour
appréhender la suite avec comme variables principales:'''

data_meteo_prod=pd.read_csv('clean_data/data_meteo_prod.csv')

########ACP

acp = '''ACP : Réduction de dimension'''

acp2 = '''Dans le but d'étayer notre analyse, nous avons réalisé une analyse par composantes principales sur
 les données de la météo avec target la consommation.'''
acp3 = '''Avant de commencer deux étapes sont essentiel :
- Isoler la consommation en target 
- Centrer et réduire les variables pour réaliser une ACP normée avec PCA
'''
code2='''target = df.consommation_mw
df = df.drop('consommation_mw', axis=1)
#classe pour standardisation
from sklearn.preprocessing import StandardScaler
#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(df)'''

acp4 = '''Analyse en composantes principales avec PCA'''

acp5 = ''' La première étape est d'instancier la classe PCA et d'appliquer les données centrées réduites de la météo.'''
code3 = ''' from sklearn.decomposition import PCA 
acp = PCA() 
coord = acp.fit_transform(Z)'''

acp6 = '''Par la suite, nous allons étudier l'information de la variance expliquée pour construire l'ACP'''

acp7 = '''Les 2 premières composantes accaparent 50 %\ de l’information disponible.
Les 'cassures' dans les graphiques ci-dessus sont souvent évoquées (règle du coude)
pour identifier le nombre de facteurs à retenir. La solution (K = 2) semble s’imposer ici '''

acp8 = '''Visualisation de l'ACP'''
acp9 = '''Pour rappel :
- Plus une variable possède une qualité de représentation élevée dans l’ACP, plus sa flèche est longue.
- Plus une variable possède une qualité de représentation élevée dans l’ACP, plus sa flèche est longue.
- Plus deux variables sont corrélées, plus leurs flèches pointent dans la même direction.
- Plus une variable est proche d’un axe principal de l’ACP, plus elle est liée à lui.
Ainsi, nous avons décidé de retirer les variables suivantes : 'altitude', 'longitude', 'latitude',
'variation de pression en 3h', 'direction du vent moyenne'. '''
acp10 = '''Nous pouvons aussi vérifier notre graphique avec le heatmap suivante:'''

acp11 = '''Les résultats confirment notre analyse. Donc nous reduisons notre dataframe en gardant :
'humidite', 'pression_station', 'pression_au_niveau_mer', 'precipitations_dans_les_3_dernieres_heures'

Après analyse avec un dataset d'uniquement ces 4 variables. Les résultats n'étaient pas assez satisfaisants. 
Suite aux visualisations & à la corrélation forte entre la témpérature et la consommation, nous avons décidé de
l'intégrer au dataset réduit via l'ACP.'''

acp12 = '''Le nouveau dataframe avec l'ACP est le suivant'''

df_acp=pd.read_csv('clean_data/df_acp.csv')
dataframeclean=df_acp.head(10)
