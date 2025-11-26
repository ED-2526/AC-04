import pandas as pd

# Carregar dataset
df = pd.read_excel("dataset_e-comerce.xlsx")

errors = {}

# -------------------------------
# 1. ID: no pot estar repetit
# -------------------------------
ids_dup = df[df.duplicated("ID", keep=False)]
errors['ID_duplicats'] = ids_dup[['ID']]

# Valors nuls
errors['ID_nuls'] = df[df['ID'].isna()][['ID']]

# -------------------------------
# 2. Year_Birth: entre 1890 i 2000
# -------------------------------
mask_year = ~df['Year_Birth'].between(1890, 2000)
errors['Year_Birth_incorrectes'] = df.loc[mask_year, ['Year_Birth']]

# Valors nuls
errors['Year_Birth_nuls'] = df[df['Year_Birth'].isna()][['Year_Birth']]

# -------------------------------
# 3. Marital_Status: valors permesos
# -------------------------------
valors_marital = ['Single', 'Together', 'Married', 'Divorced','Widow']
mask_marital = ~df['Marital_Status'].isin(valors_marital)
errors['Marital_Status_incorrectes'] = df.loc[mask_marital, ['Marital_Status']]

# Valors nuls
errors['Marital_Status_nuls'] = df[df['Marital_Status'].isna()][['Marital_Status']]

# -------------------------------
# 4. Education: valors permesos
# -------------------------------
valors_education = ['Graduation', 'PhD', 'Master', 'Basic','2n Cycle']
mask_edu = ~df['Education'].isin(valors_education)
errors['Education_incorrectes'] = df.loc[mask_edu, ['Education']]

# Valors nuls
errors['Education_nuls'] = df[df['Education'].isna()][['Education']]

# -------------------------------
# 5. Income: no pot ser negatiu
# -------------------------------
mask_income = df['Income'] < 0
errors['Income_incorrectes'] = df.loc[mask_income, ['Income']]

# Valors nuls
errors['Income_nuls'] = df[df['Income'].isna()][['Income']]

# -------------------------------
# 6. Kidhome entre 0 i 5
# -------------------------------
mask_kidhome = ~df['Kidhome'].between(0, 5)
errors['Kidhome_incorrectes'] = df.loc[mask_kidhome, ['Kidhome']]

# Valors nuls
errors['Kidhome_nuls'] = df[df['Kidhome'].isna()][['Kidhome']]

# -------------------------------
# 7. Teenhome entre 0 i 5
# -------------------------------
mask_teenhome = ~df['Teenhome'].between(0, 5)
errors['Teenhome_incorrectes'] = df.loc[mask_teenhome, ['Teenhome']]

# Valors nuls
errors['Teenhome_nuls'] = df[df['Teenhome'].isna()][['Teenhome']]

# -------------------------------
# 8. Dt_Customer: format dd-mm-YYYY
# -------------------------------
def validar_data(x):
    try:
        pd.to_datetime(x, format="%d-%m-%Y")
        return True
    except:
        return False

mask_data = ~df['Dt_Customer'].apply(validar_data)
errors['Dt_Customer_incorrectes'] = df.loc[mask_data, ['Dt_Customer']]

# Valors nuls
errors['Dt_Customer_nuls'] = df[df['Dt_Customer'].isna()][['Dt_Customer']]

# -------------------------------
# MOSTRAR TOTS ELS ERRORS
# -------------------------------
for nom, err_df in errors.items():
    if len(err_df) > 0:
        print(f"\n--- {nom} ---")
        print(err_df)
    else:
        print(f"\n✔ Cap error trobat a: {nom}")
