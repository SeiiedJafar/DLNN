import pandas as pd

df = pd.read_csv('data.csv')
df.sample(5)

print("تعداد NaN ها برای هر ستون:")
print(df.isna().sum())

df.fillna(0, inplace=True)

# df.dropna(inplace=True)

df['Type of Breast Surgery'] = df['Type of Breast Surgery'].replace({'MASTECTOMY': 1, 'BREAST CONSERVING': 2})

df['Cancer Type Detailed'] = df['Cancer Type Detailed'].replace({'Breast Invasive Ductal Carcinoma': 1,
                                                                 'Breast Mixed Ductal and Lobular Carcinoma': 2,
                                                                 'Breast Invasive Lobular Carcinoma': 3,
                                                                 'Invasive Breast Carcinoma': 4,
                                                                 'Breast Invasive Mixed Mucinous Carcinoma': 5,
                                                                 'Breast Angiosarcoma': 6,
                                                                 'Breast': 7,
                                                                 'Metaplastic Breast Cancer': 8})

df['Cellularity'] = df['Cellularity'].replace({'Low': 1, 'Moderate': 2, 'High': 3})

df['Chemotherapy'] = df['Chemotherapy'].replace({'NO': 1, 'YES': 2})

df['Pam50 + Claudin-low subtype'] = df['Pam50 + Claudin-low subtype'].replace({'claudin-low': 1, 'LumA': 2, 'LumB': 3, 'Normal': 4, 'Her2': 5, 'Basal': 6, 'NC': 7})

df['ER status measured by IHC'] = df['ER status measured by IHC'].replace({'Negative': 1, 'Positve': 2})

df['ER Status'] = df['ER Status'].replace({'Negative': 1, 'Positive': 2})

df['HER2 status measured by SNP6'] = df['HER2 status measured by SNP6'].replace({'NEUTRAL': 1, 'LOSS': 2, 'GAIN': 3, 'UNDEF': 4})

df['HER2 Status'] = df['HER2 Status'].replace({'Negative': 1, 'Positive': 2})

df['Tumor Other Histologic Subtype'] = df['Tumor Other Histologic Subtype'].replace({'Ductal/NST': 1,
                                                                                     'Mixed': 2,
                                                                                     'Lobular': 3,
                                                                                     'Tubular/ cribriform': 4,
                                                                                     'Mucinous': 5,
                                                                                     'Medullary': 6,
                                                                                     'Metaplastic': 7,
                                                                                     'Other': 8})

df['Hormone Therapy'] = df['Hormone Therapy'].replace({'NO': 1, 'YES': 2})

df['Inferred Menopausal State'] = df['Inferred Menopausal State'].replace({'Pre': 1, 'Post': 2})

df['Integrative Cluster'] = df['Integrative Cluster'].replace({'4ER-': 4, '4ER+': 11})

df['Primary Tumor Laterality'] = df['Primary Tumor Laterality'].replace({'Left': 1, 'Right': 2})

df['Oncotree Code'] = df['Oncotree Code'].replace({'IDC': 1, 'MDLC': 2, 'ILC': 3, 'BRCA': 4, 'IMMC': 5, 'PBS': 6, 'BREAST': 7, 'MBC': 8})

df['Overall Survival Status'] = df['Overall Survival Status'].replace({'0:LIVING': 1, '1:DECEASED': 2})

df['PR Status'] = df['PR Status'].replace({'Negative': 1, 'Positive': 2})

df['Radio Therapy'] = df['Radio Therapy'].replace({'NO': 1, 'YES': 2})

df['Relapse Free Status'] = df['Relapse Free Status'].replace({'0:Not Recurred': 1, '1:Recurred': 2})

df['3-Gene classifier subtype'] = df['3-Gene classifier subtype'].replace({'ER-/HER2-': 1, 'ER+/HER2- High Prolif': 2, 'ER+/HER2- Low Prolif': 3, 'HER2+': 4})

df['Patient\'s Vital Status'] = df['Patient\'s Vital Status'].replace({'Living': 1, 'Died of Disease': 2, 'Died of Other Causes': 3})

columns = ['Type of Breast Surgery',
           'Cancer Type Detailed',
           'Cellularity',
           'Chemotherapy',
           'Pam50 + Claudin-low subtype',
           'Cohort',
           'ER status measured by IHC',
           'ER Status',
           'Neoplasm Histologic Grade',
           'HER2 status measured by SNP6',
           'HER2 Status',
           'Tumor Other Histologic Subtype',
           'Hormone Therapy',
           'Inferred Menopausal State',
           'Integrative Cluster',
           'Primary Tumor Laterality',
           'Lymph nodes examined positive',
           'Mutation Count',
           'Overall Survival Status',
           'PR Status',
           'Radio Therapy',
           'Relapse Free Status',
           '3-Gene classifier subtype',
           'Tumor Stage',
           'Patient\'s Vital Status']
for item in columns:
    df[item] = df[item].astype(int)

df.to_csv('cleaned_data1.csv', index=False)
