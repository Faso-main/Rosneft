import pandas as pd
import numpy as np
import os


# Configuration
CV_PATH=os.path.join('Data_validation','src','analysis_data.csv')

# Data_generation
np.random.seed(42)
n_wells = 1000

data = {
    'well_id': np.arange(1, n_wells+1),
    'depth': np.random.randint(1500, 4500, n_wells),
    'reservoir_pressure': np.random.normal(250, 30, n_wells).round(1),
    'current_pressure': np.random.normal(180, 40, n_wells).round(1),
    'oil_rate': np.random.lognormal(2.5, 0.3, n_wells).round(1),
    'water_cut': np.random.beta(2, 5, n_wells).round(3)*100,
    'GOR': np.random.weibull(1.5, n_wells)*50,
    'sand_content': np.random.choice([0, 1, 2, 3], n_wells, p=[0.6,0.3,0.08,0.02]),
    'well_age': np.random.randint(1, 25, n_wells),
    'last_workover': np.random.randint(0, 5, n_wells),
    'completion_type': np.random.choice(['Horizontal', 'Vertical', 'Multilateral'], n_wells),
    'permeability': np.random.lognormal(2.2, 0.5, n_wells).round(2),
    'skin_factor': np.random.normal(5, 3, n_wells).round(1),
    'work_type_recommended': np.random.choice(
        ['Hydraulic Fracturing', 'Acidizing', 'Water Shut-off', 'Sand Control', 'ESP Replacement', 'None'],
        n_wells,
        p=[0.25,0.2,0.15,0.1,0.15,0.15]
    )
}

# Создание DataFrame
df = pd.DataFrame(data)

# Добавление зависимостей между параметрами
df.loc[df['water_cut'] > 75, 'work_type_recommended'] = 'Water Shut-off'
df.loc[df['oil_rate'] < 50, 'work_type_recommended'] = 'Hydraulic Fracturing'
df.loc[df['sand_content'] > 1, 'work_type_recommended'] = 'Sand Control'
df.loc[df['current_pressure'] < 100, 'work_type_recommended'] = 'ESP Replacement'

# Сохранение в CSV
df.to_csv(CV_PATH, index=False)