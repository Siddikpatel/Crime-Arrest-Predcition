import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
sns.set_style("darkgrid")

def load_data(filename):
    return pd.read_csv(filename)

def primary_type_map():

    primary_type_map = {
        ('BURGLARY','MOTOR VEHICLE THEFT','THEFT','ROBBERY') : 'THEFT',
        ('BATTERY','ASSAULT','NON-CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)') : 'NON-CRIMINAL_ASSAULT',
        ('CRIM SEXUAL ASSAULT','SEX OFFENSE','STALKING','PROSTITUTION') : 'SEXUAL_OFFENSE',
        ('WEAPONS VIOLATION','CONCEALED CARRY LICENSE VIOLATION') :  'WEAPONS_OFFENSE',
        ('HOMICIDE','CRIMINAL DAMAGE','DECEPTIVE PRACTICE','CRIMINAL TRESPASS') : 'CRIMINAL_OFFENSE',
        ('KIDNAPPING','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN') : 'HUMAN_TRAFFICKING_OFFENSE',
        ('NARCOTICS','OTHER NARCOTIC VIOLATION') : 'NARCOTIC_OFFENSE',
        ('OTHER OFFENSE','ARSON','GAMBLING','PUBLIC PEACE VIOLATION','INTIMIDATION','INTERFERENCE WITH PUBLIC OFFICER','LIQUOR LAW VIOLATION','OBSCENITY','PUBLIC INDECENCY') : 'OTHER_OFFENSE'
    }
    primary_type_mapping = {}
    for keys, values in primary_type_map.items():
        for key in keys:
            primary_type_mapping[key] = values
    return primary_type_mapping

def season_map():
    season_map = {
        ('March','April','May') : 'Spring',
        ('June','July','August') : 'Summer',
        ('September','October','November') : 'Fall',
        ('December','January','February') : 'Winter'
    }
    season_mapping = {}
    for keys, values in season_map.items():
        for key in keys:
            season_mapping[key] = values
    return season_mapping

def loc_map():

    loc_map = {
        ('RESIDENCE', 'APARTMENT', 'CHA APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'RESIDENCE-GARAGE',
        'RESIDENTIAL YARD (FRONT/BACK)', 'DRIVEWAY - RESIDENTIAL', 'HOUSE') : 'RESIDENCE',
        
        ('BARBERSHOP', 'COMMERCIAL / BUSINESS OFFICE', 'CURRENCY EXCHANGE', 'DEPARTMENT STORE', 'RESTAURANT',
        'ATHLETIC CLUB', 'TAVERN/LIQUOR STORE', 'SMALL RETAIL STORE', 'HOTEL/MOTEL', 'GAS STATION',
        'AUTO / BOAT / RV DEALERSHIP', 'CONVENIENCE STORE', 'BANK', 'BAR OR TAVERN', 'DRUG STORE',
        'GROCERY FOOD STORE', 'CAR WASH', 'SPORTS ARENA/STADIUM', 'DAY CARE CENTER', 'MOVIE HOUSE/THEATER',
        'APPLIANCE STORE', 'CLEANING STORE', 'PAWN SHOP', 'FACTORY/MANUFACTURING BUILDING', 'ANIMAL HOSPITAL',
        'BOWLING ALLEY', 'SAVINGS AND LOAN', 'CREDIT UNION', 'KENNEL', 'GARAGE/AUTO REPAIR', 'LIQUOR STORE',
        'GAS STATION DRIVE/PROP.', 'OFFICE', 'BARBER SHOP/BEAUTY SALON') : 'BUSINESS',
        
        ('VEHICLE NON-COMMERCIAL', 'AUTO', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)', 'TAXICAB',
        'VEHICLE-COMMERCIAL', 'VEHICLE - DELIVERY TRUCK', 'VEHICLE-COMMERCIAL - TROLLEY BUS',
        'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS') : 'VEHICLE',
        
        ('AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'CTA PLATFORM', 'CTA STATION', 'CTA BUS STOP',
        'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA', 'CTA TRAIN', 'CTA BUS', 'CTA GARAGE / OTHER PROPERTY',
        'OTHER RAILROAD PROP / TRAIN DEPOT', 'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
        'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRCRAFT',
        'AIRPORT PARKING LOT', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION',
        'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
        'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',
        'CTA TRACKS - RIGHT OF WAY', 'AIRPORT/AIRCRAFT', 'BOAT/WATERCRAFT', 'CTA PROPERTY', 'CTA "L" PLATFORM',
        'RAILROAD PROPERTY') : 'PUBLIC_TRANSPORTATION',
        
        ('HOSPITAL BUILDING/GROUNDS', 'NURSING HOME/RETIREMENT HOME', 'SCHOOL, PUBLIC, BUILDING',
        'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PRIVATE, BUILDING',
        'MEDICAL/DENTAL OFFICE', 'LIBRARY', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'YMCA', 'HOSPITAL') : 'PUBLIC_BUILDING',
        
        ('STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'PARK PROPERTY', 'ALLEY', 'CEMETARY',
        'CHA HALLWAY/STAIRWELL/ELEVATOR', 'CHA PARKING LOT/GROUNDS', 'COLLEGE/UNIVERSITY GROUNDS', 'BRIDGE',
        'SCHOOL, PRIVATE, GROUNDS', 'FOREST PRESERVE', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'PARKING LOT', 'DRIVEWAY',
        'HALLWAY', 'YARD', 'CHA GROUNDS', 'RIVER BANK', 'STAIRWELL', 'CHA PARKING LOT') : 'PUBLIC_AREA',
        
        ('POLICE FACILITY/VEH PARKING LOT', 'GOVERNMENT BUILDING/PROPERTY', 'FEDERAL BUILDING', 'JAIL / LOCK-UP FACILITY',
        'FIRE STATION', 'GOVERNMENT BUILDING') : 'GOVERNMENT',
        
        ('OTHER', 'ABANDONED BUILDING', 'WAREHOUSE', 'ATM (AUTOMATIC TELLER MACHINE)', 'VACANT LOT/LAND',
        'CONSTRUCTION SITE', 'POOL ROOM', 'NEWSSTAND', 'HIGHWAY/EXPRESSWAY', 'COIN OPERATED MACHINE', 'HORSE STABLE',
        'FARM', 'GARAGE', 'WOODED AREA', 'GANGWAY', 'TRAILER', 'BASEMENT', 'CHA PLAY LOT') : 'OTHER'  
    }

    loc_mapping = {}
    for keys, values in loc_map.items():
        for key in keys:
            loc_mapping[key] = values
    return loc_mapping

def categorise(crime_data):
    crime_data.arrest = crime_data.arrest.astype(int)
    crime_data.domestic = crime_data.domestic.astype(int)

    crime_data.year = pd.Categorical(crime_data.year)
    crime_data.time = pd.Categorical(crime_data.time)
    crime_data.domestic = pd.Categorical(crime_data.domestic)
    crime_data.arrest = pd.Categorical(crime_data.arrest)
    crime_data.beat = pd.Categorical(crime_data.beat)
    crime_data.district = pd.Categorical(crime_data.district)
    crime_data.ward = pd.Categorical(crime_data.ward)
    crime_data.community_area = pd.Categorical(crime_data.community_area)

    return crime_data

def split_data(crime_data):

    crimes_data_prediction = crime_data.drop(['date'],axis=1)

    clm = ['block', 'iucr', 'primary_type', 'description', 'location_description',
        'fbi_code', 'updated_on', 'community_area_name', 'day_of_week', 'month',
        'primary_type_grouped', 'zone', 'season', 'loc_grouped']


    le = preprocessing.LabelEncoder()
    
    mapping_headings = {}

    for x in clm:
        crimes_data_prediction[x] = crimes_data_prediction[x].astype(str)
        crimes_data_prediction[x]=le.fit_transform(crimes_data_prediction[x])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_headings[x] = le_name_mapping

    crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)
    tr, te, ytr, yte = train_test_split(crimes_data_prediction.drop(['arrest_1'],axis=1),crimes_data_prediction['arrest_1'], test_size=0.3, random_state=42)
    return tr, te, ytr, yte, mapping_headings

def feature_imp(train, ytrain):
    mutual_info = mutual_info_classif(train, ytrain)
    #We select only dependent variable.Avoiding Independent variable
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = train.columns
    return mutual_info

def final_data(x_train, out_train, x_test):
    sel_five_cols = SelectKBest(mutual_info_classif, k=5)
    sel_five_cols.fit(x_train, out_train)
    columns = x_train.columns[sel_five_cols.get_support()]
    col=columns.values.tolist()
    return col

def rmse_vals(x_train, Y_train, x_test, Y_test):
    rmse = []
    for K in range(1,40):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K)

        model.fit(x_train, Y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        error = sqrt(mean_squared_error(Y_test,pred)) #calculate rmse
        rmse.append(error) #store rmse values
    return rmse

def accuracy_counter(x_train, Y_train, x_test, Y_test, knn):
    
    knn.fit(x_train, Y_train)

    y_pred = knn.predict(x_test)

    cm = confusion_matrix(Y_test, y_pred)
    ac = accuracy_score(Y_test,y_pred)
    return ac, cm

def predict(inp_dict):
    # for i, s in enumerate(li):
    #     dictt = mapp_list[i]
    #     li[i] = dictt[s]
    global mapp_list 
    for i, key in enumerate(inp_dict.keys()):
        dictt = mapp_list[i]
        inp_dict[key] = dictt[inp_dict[key]]
        
    # dictt = {'iucr': li[0], 'primary_type': li[1], 'description': li[2], 'fbi_code': li[3], 'primary_type_grouped': li[4]}
    f1 = pd.DataFrame(inp_dict, index=[0])
    out = knn.predict(f1)
    return out

def predict_model_param(model, inp_dict):
    # for i, s in enumerate(li):
    #     dictt = mapp_list[i]
    #     li[i] = dictt[s]
    global mapp_list 
    for i, key in enumerate(inp_dict.keys()):
        dictt = mapp_list[i]
        inp_dict[key] = dictt[inp_dict[key]]
        
    # dictt = {'iucr': li[0], 'primary_type': li[1], 'description': li[2], 'fbi_code': li[3], 'primary_type_grouped': li[4]}
    f1 = pd.DataFrame(inp_dict, index=[0])
    out = model.predict(f1)
    return out

crimes_data = load_data('crime.csv')

crimes_data.columns = crimes_data.columns.str.strip()
crimes_data.columns = crimes_data.columns.str.replace(',', '')
crimes_data.columns = crimes_data.columns.str.replace(' ', '_')
crimes_data.columns = crimes_data.columns.str.lower()

crimes_data[crimes_data.duplicated(keep=False)]

crimes_data.drop(['id','case_number','location'],axis=1,inplace=True)

crimes_data.dropna(subset=['latitude'],inplace=True)
crimes_data.reset_index(drop=True,inplace=True)

crimes_data.date = pd.to_datetime(crimes_data.date)
crimes_data['day_of_week'] = crimes_data.date.dt.day_name()
crimes_data['month'] = crimes_data.date.dt.month_name()
crimes_data['time'] = crimes_data.date.dt.hour

crimes_data['primary_type_grouped'] = crimes_data.primary_type.map(primary_type_map())

zone_mapping = {
    'N' : 'North',
    'S' : 'South',
    'E' : 'East',
    'W' : 'West'
}
crimes_data['zone'] = crimes_data.block.str.split(" ", n = 2, expand = True)[1].map(zone_mapping)

crimes_data['season'] = crimes_data.month.map(season_map())

crimes_data['loc_grouped'] = crimes_data.location_description.map(loc_map())

crimes_data = categorise(crimes_data)

X_train, X_test, y_train, y_test, mappings = split_data(crimes_data)
# pd.set_option('display.max_rows', None)

mutual_info = feature_imp(X_train, y_train)

column = final_data(X_train, y_train, X_test)
X_train=X_train[column]
X_test=X_test[column]

rmse_val = rmse_vals(X_train, y_train, X_test, y_test)

knn = KNeighborsClassifier(n_neighbors=20)
ac, cm = accuracy_counter(X_train, y_train, X_test, y_test, knn)

pkl_file = 'model.pkl'
pickle.dump(knn, open(pkl_file, 'wb'))

iucr_maps = mappings['iucr']
ptype_maps = mappings['primary_type']
desc_maps = mappings['description']
fbi_maps = mappings['fbi_code']
pgrp_maps = mappings['primary_type_grouped']
mapp_list = [iucr_maps, ptype_maps, desc_maps, fbi_maps, pgrp_maps]

# li = ['2027', 'NARCOTICS', 'POSS: CRACK', '18', 'OTHER_OFFENSE']
