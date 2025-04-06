import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/lkcka/MLops/refs/heads/main/Datasets/co2.csv', delimiter = ',')
    df.to_csv("co2.csv", index = False)
    return df

def preprocessing_data_frame(frame):
    df = pd.read_csv(frame)
    cat_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
    num_columns = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 
                   'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)']

    question_city = df[df['Fuel Consumption City (L/100 km)'] > 23]
    df = df.drop(question_city.index)

    question_engine = df[(df['Engine Size(L)'] < 1.4) & (df['Engine Size(L)'] > 7)]
    df = df.drop(question_engine.index)

    question_cylinders = df[df['Cylinders'].isin([3, 5, 10, 16])]
    df = df.drop(question_cylinders.index)

    question_comb = df[(df['Fuel Consumption Comb (L/100 km)'] < 4.5) & (df['Fuel Consumption Comb (L/100 km)'] > 17)]
    df = df.drop(question_comb.index)

    question_hwy = df[(df['Fuel Consumption Hwy (L/100 km)'] < 5) & (df['Fuel Consumption Hwy (L/100 km)'] > 16)]
    df = df.drop(question_hwy.index)

    question_co2 = df[(df['CO2 Emissions(g/km)'] < 120) & (df['CO2 Emissions(g/km)'] > 420)]
    df = df.drop(question_co2.index)

    
    df = df.reset_index(drop=True)
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv')
    return df

download_data()
preprocessing_data_frame("co2.csv")
