# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (15,5)
plt.style.use('bmh')

# DATA CLEANING
# In[2]:
df = pd.read_excel('reactivo_2.xls')
df.head()

# In[3]:
df = df.drop(['Row ID','Order ID', 'Customer ID', 'Country', 'Postal Code', 'Product ID', 'Product Name'], axis = 1)

# EDA
## Shipping & Dates:
# In[4]:
df['Shipping Days'] = df['Ship Date'] - df['Order Date']
df['Shipping Days'] = df['Shipping Days'].map(lambda x:x.days)

# In[5]:
fig, axs = plt.subplots(3,2, figsize = (15,18))
sns.histplot(ax = axs[0][0], data=df, x="Ship Mode")
axs[0][0].set_title('Ship Mode Histogram')

sns.histplot(ax = axs[0][1], data = df[df['Ship Mode'] == 'Standard Class'], x = 'Shipping Days')
sns.histplot(ax = axs[0][1], data = df[df['Ship Mode'] == 'Second Class'], x = 'Shipping Days', color = 'r')
sns.histplot(ax = axs[0][1], data = df[df['Ship Mode'] == 'First Class'], x = 'Shipping Days', color = 'g')
sns.histplot(ax = axs[0][1], data = df[df['Ship Mode'] == 'Same Day'], x = 'Shipping Days', color = 'm')
axs[0][1].set_title('Shipping days for Ship Mode')
axs[0][1].legend(['Standard Class','Second Class','First Class','Same Day'])

sns.histplot(ax = axs[1][0], data = df[df['Segment'] == 'Consumer'], x = 'Ship Mode')
sns.histplot(ax = axs[1][0], data = df[df['Segment'] == 'Corporate'], x = 'Ship Mode', color = 'r')
sns.histplot(ax = axs[1][0], data = df[df['Segment'] == 'Home Office'], x = 'Ship Mode', color = 'g')
axs[1][0].set_title('Segment for Ship Mode')
axs[1][0].legend(['Consumer','Corporate','Home Office'])

sns.histplot(ax = axs[1][1], data = df[df['Segment'] == 'Consumer'], x = 'Shipping Days')
sns.histplot(ax = axs[1][1], data = df[df['Segment'] == 'Corporate'], x = 'Shipping Days', color = 'r')
sns.histplot(ax = axs[1][1], data = df[df['Segment'] == 'Home Office'], x = 'Shipping Days', color = 'g')
axs[1][1].set_title('Segment for Shipping Days')
axs[1][1].legend(['Consumer','Corporate','Home Office'])

sns.histplot(ax = axs[2][0],data = df[df['Category'] == 'Office Supplies'], x = 'Ship Mode', color = 'r')
sns.histplot(ax = axs[2][0],data = df[df['Category'] == 'Furniture'], x = 'Ship Mode')
sns.histplot(ax = axs[2][0],data = df[df['Category'] == 'Technology'], x = 'Ship Mode', color = 'g')
axs[2][0].set_title('Category for Ship Mode')
axs[2][0].legend(['Office Supplies','Furniture','Technology'])

sns.histplot(ax = axs[2][1],data = df[df['Category'] == 'Office Supplies'], x = 'Shipping Days', color = 'r')
sns.histplot(ax = axs[2][1],data = df[df['Category'] == 'Furniture'], x = 'Shipping Days')
sns.histplot(ax = axs[2][1],data = df[df['Category'] == 'Technology'], x = 'Shipping Days', color = 'g')
axs[2][1].set_title('Category for Ship Days')
axs[2][1].legend(['Office Supplies','Furniture','Technology'])

fig.savefig('Histograma.png')
## Ventas:
# In[10]:
df['Order Count'] = 1
bydate = df.copy()
bydate = df.groupby(by = 'Order Date').sum()
del bydate['Shipping Days']

# In[11]:
v14 = dict(bydate[(bydate.index > '2014-01-01')&(bydate.index<'2015-01-01')].sum())
v15 = dict(bydate[(bydate.index > '2015-01-01')&(bydate.index<'2016-01-01')].sum())
v16 = dict(bydate[(bydate.index > '2016-01-01')&(bydate.index<'2017-01-01')].sum())
v17 = dict(bydate[(bydate.index > '2017-01-01')&(bydate.index<'2018-01-01')].sum())
resumen_años = pd.DataFrame([v14,v15,v16,v17], index = [2014,2015,2016,2017])

fig = plt.figure()
fig, axs = plt.subplots(5,1, figsize = (15,18))
sns.barplot(ax = axs[0], data = resumen_años, x = resumen_años.index, y = 'Sales')
axs[0].set_title('Ventas por Año')
sns.barplot(ax = axs[1], data = resumen_años, x = resumen_años.index, y = 'Quantity')
axs[1].set_title('Cantidad de artículos vendidos por Año')
sns.barplot(ax = axs[2], data = resumen_años, x = resumen_años.index, y = 'Discount')
axs[2].set_title('Descuentos por Año')
sns.barplot(ax = axs[3], data = resumen_años, x = resumen_años.index, y = 'Profit')
axs[3].set_title('Ganancias por año')
sns.barplot(ax = axs[4], data = resumen_años, x = resumen_años.index, y = 'Sales')
axs[4].set_title('Órdenes por año')
fig.savefig('Graficas de barras por año.png')

# In[12]:
fig = plt.figure()
fig, axs = plt.subplots(4,1, figsize = (15,18))
sns.lineplot(ax = axs[0], data = bydate, x = bydate.index, y = 'Sales')
axs[0].set_title('Ventas a través de los años')
axs[0].set_xlabel('')
sns.lineplot(ax = axs[1], data = bydate, x = bydate.index, y = 'Profit')
axs[1].set_title('Ganancias a través de los años')
axs[1].set_xlabel('')
sns.lineplot(ax = axs[2], data = bydate, x = bydate.index, y = 'Quantity')
axs[2].set_title('Cantidad de artículos vendidos a través de los años')
axs[2].set_xlabel('')
sns.lineplot(ax = axs[3], data = bydate, x = bydate.index, y = 'Order Count')
axs[3].set_title('Órdenes generadas a través de los años')
axs[3].set_xlabel('')
fig.savefig('Series de Tiempo.png')

## Ventas por estados:
# In[13]:
bystate = df.copy()
bystate = bystate.groupby(by = 'State').sum()
del bystate['Shipping Days']
print(bystate.sort_values(by = 'Profit').head())
print(bystate.sort_values(by = 'Profit').tail())

# In[14]:
prueba = df.copy()
prueba['Year'] = prueba['Order Date'].map(lambda x:x.year)
pt_prueba = pd.pivot_table(prueba, values = ['Sales','Profit','Discount','Quantity','Order Count'], index = 'State', columns = 'Year', aggfunc = 'sum')
pt_worst = pt_prueba[(pt_prueba.index == 'Texas')|(pt_prueba.index == 'Ohio')|(pt_prueba.index == 'Pennsylvania')]
pt_best = pt_prueba[(pt_prueba.index == 'California')|(pt_prueba.index == 'New York')|(pt_prueba.index == 'Washington')]

# MODELLING
# In[16]:
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# In[17]:
label = LabelEncoder()
df['Ship Mode'] = label.fit_transform(df['Ship Mode'])
df['Customer Name'] = label.fit_transform(df['Customer Name'])
df['Segment'] = label.fit_transform(df['Segment'])
df['City'] = label.fit_transform(df['City'])
df['State'] = label.fit_transform(df['State'])
df['Region'] = label.fit_transform(df['Region'])
df['Category'] = label.fit_transform(df['Category'])
df['Sub-Category'] = label.fit_transform(df['Sub-Category'])

# In[19]:
X = df.loc[:, ~df.columns.isin(['Order Date', 'Ship Date','Shipping Days','Order Count'])]
X_norm = (X-X.min())/(X.max()-X.min())
X_norm.describe()

# In[20]:
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 100)
    kmeans.fit(X_norm)
    wcss.append(kmeans.inertia_)


# In[21]:
fig = plt.figure(figsize = (10,5))
plt.plot(range(1,11),wcss)
plt.title('Codo de Jambú')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.savefig('Codo de Jambú.png')

# In[22]:
clustering = KMeans(n_clusters = 4, max_iter = 100)
clustering.fit(X_norm)
df['Cluster Label'] = clustering.labels_

# In[24]:
pca = PCA(n_components = 2)
pca_clients = pca.fit_transform(X_norm)
pca_clients_df = pd.DataFrame(data = pca_clients, columns = ['Componente_1','Componente_2'])
group_clients = pd.concat([pca_clients_df,df['Cluster Label']], axis = 1)

# In[25]:
fig = plt.figure()
color_theme = np.array(['blue','red','green','yellow'])
sns.scatterplot(x = group_clients.Componente_1, y = group_clients.Componente_2, c = color_theme[group_clients['Cluster Label']])
plt.title('Clustering')
plt.savefig('Agrupación.png')