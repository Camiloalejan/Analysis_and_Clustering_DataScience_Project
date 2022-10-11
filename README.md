# Analysis and Clustering - Data Science Project

## Abstract:
The following document shows the data of a (seemingly) office supply company. An EDA was made in order to analyse the features of the dataset by using visualizations such as: histograms, time series, bar charts and tables. The modelling part includes a clustering model, using the "Elbow Method" and decomposing the components with the PCA functions.

#### (A Spanish report is also included in the repository)

## EDA:
- Histograms analizing the shipping by: Ship mode, shipping days, segment, and category:
![Histogram](https://github.com/Camiloalejan/Analysis_and_Clustering_DataScience_Project/blob/main/images/Histograma.png)

- Bar Chart analizing: sales, number of items sold, discounts, profit, and amount of orders by year:
![Bar chart](https://github.com/Camiloalejan/Analysis_and_Clustering_DataScience_Project/blob/main/images/Graficas%20de%20barras%20por%20a%C3%B1o.png)

- Time series showing the behavior of sales, profit, number of items sold, and amount of orders:
![Time series](https://github.com/Camiloalejan/Analysis_and_Clustering_DataScience_Project/blob/main/images/Series%20de%20Tiempo.png)

#### NOTE: A pivot table was also made to show the 3 best and 3 worst states by profit per year.

## MODELLING:
- Use of the _Elbow Method_ to determine the optimal number of clusters to divide our data set.
![Jambú](https://github.com/Camiloalejan/Analysis_and_Clustering_DataScience_Project/blob/main/images/Codo%20de%20Jamb%C3%BA.png)

- Clustering of each registered customer:
![Clustering](https://github.com/Camiloalejan/Analysis_and_Clustering_DataScience_Project/blob/main/images/Agrupaci%C3%B3n.png)

#### NOTE: we needed the use of PCA function to decompose the features of each customer into just two principal components in order to graph the results of the clustering model.
