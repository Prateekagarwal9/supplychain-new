# Databricks notebook source
# MAGIC %md 
# MAGIC # Demand Forecasting Using ARIMA

# COMMAND ----------

# MAGIC %sh
# MAGIC curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
# MAGIC curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
# MAGIC apt-get update
# MAGIC ACCEPT_EULA=Y apt-get install msodbcsql17
# MAGIC apt-get -y install unixodbc-dev
# MAGIC sudo apt-get install python3-pip -y
# MAGIC pip3 install --upgrade pyodbc

# COMMAND ----------

#importing Libraries
import pandas as pd
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyodbc


conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=marketplace-server.database.windows.net;' + 'DATABASE=marketplace-db;' + 'uid=celebal;' + 'pwd=password@123')

data=pd.read_sql_query('''select *   FROM [dbo].[marketplace-db]''',conn)"""

# COMMAND ----------

# DBTITLE 1,Importing Libraries and Loading data
#importing Libraries
import pandas as pd
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *

#Read Data from dbfs
pandas_data = pd.read_csv("/dbfs/FileStore/tables/Historical_Product_Demand-607c8.csv", header = 'infer')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Cleaning and Preprocessing

# COMMAND ----------

# DBTITLE 0,Cleaning and Preprocessing
#Drop null values and Duplicates values
pandas_data.dropna(subset = ['Date'], inplace = True)
pandas_data.drop_duplicates(inplace = True)
#Create Spark Dataframe
data = spark.createDataFrame(pandas_data)

#udf for Removing brackets in orders column
def remove_bracket(string):
    if string.find('(') != -1:
        return int(string[string.find('(') + 1: string.find(')')])
    else:
        return int(string)
      
remove_bracket_udf = udf(remove_bracket)
data = data.withColumn('Order_Demand', remove_bracket_udf('Order_Demand'))


#udf for Date Conversion 
conv_date =  udf (lambda x: datetime.strptime(x, '%Y/%m/%d'), DateType())
data = data.withColumn('Order_Demand',data['Order_Demand'].cast(IntegerType()))
data_clean = data.withColumn('Date',  conv_date(col('Date') ))
display(data_clean)

# COMMAND ----------

# DBTITLE 1,Distinct values for Warehouse, Product Code, Product Category, 
#distinct Warehouse
distinct_warehouse = data_clean.select('Warehouse').distinct().count()
print('Total Distinct Warehouses - ',distinct_warehouse)

# Distinct Product Code
distinct_product_code = data_clean.select('Product_Code').distinct().count()
print('Total Distinct Product Codes - ',distinct_product_code)

# Distinct Product Category 
distinct_product_category = data_clean.select('Product_Category').distinct().count()
print('Total Distinct Product Categories - ',distinct_product_code)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Explanatory Data Analysis

# COMMAND ----------

# DBTITLE 1,Minimum number of product that capture the max number of orders (80%)
#aggregated Orders according to Product Code and Order Demand
data_clean_1 = data_clean.select('Product_Code','Order_Demand')\
                          .groupBy('Product_Code').agg(sum('Order_Demand').alias('Agg_Orders')).sort(desc('Agg_Orders'))

#total 670 Product captutre 80% of Orders
data_clean_1 = data_clean_1.limit(670)

#converted Date to Monthwise and aggregated orders accordingly
df = data_clean_1.join(data_clean, ["Product_Code"]).drop('Agg_Orders')
df_eighty_percent = df.withColumn("Date_Year", year(df.Date))\
  .withColumn("Date_Month", month(df.Date))\
  .withColumn("joined_column", concat(col("Date_Year"), lit('-'), col("Date_Month")))\
  .withColumn("Date_Monthwise", to_date(unix_timestamp("joined_column", "yyyy-MM").cast("timestamp")))\
  .drop("Date_Year")\
  .drop("Date_Month")\
  .drop("joined_column")\
  .drop("Date")

#Product Code and their Orders
display(df_eighty_percent)

# COMMAND ----------

# DBTITLE 1,Aggregated Orders Monthwise
df_orders_agg = df_eighty_percent.select('Product_Code','Date_Monthwise','Order_Demand')\
                      .groupBy('Date_Monthwise').agg(sum('Order_Demand').alias('Orders')).orderBy('Date_Monthwise')
display(df_orders_agg)

# COMMAND ----------

# DBTITLE 1,Mean Global Orders per Month
df_orders_mean = df_eighty_percent.select('Product_Code','Date_Monthwise','Order_Demand')\
                      .groupBy('Date_Monthwise').agg(mean('Order_Demand').alias('Orders')).orderBy('Date_Monthwise')
display(df_orders_mean)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### After taking very Informative data for our Prediction

# COMMAND ----------

# DBTITLE 1,Aggregated Orders Monthwise
df_process_agg = df_eighty_percent.select('Product_Code','Date_Monthwise','Order_Demand')\
                                  .groupBy('Date_Monthwise').agg(sum('Order_Demand').alias('Orders'))\
                                   .orderBy('Date_Monthwise').filter(col('Orders') > 8027352)
display(df_process_agg)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. Forecasting

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.1 Forecasting for Aggregated Orders of selected widget Monthwise

# COMMAND ----------

data_processed = data_clean.select('Product_Code','Order_Demand')\
                        .groupBy('Product_Code').agg(sum('Order_Demand').alias('Agg_Orders')).sort(desc('Agg_Orders'))

#took only 5 product for forecasting
data_limit = data_processed.limit(5)

df = data_limit.join(data_clean, ["Product_Code"]).drop('Agg_Orders')
df_cleaned = df.withColumn("Date_Year", year(df.Date))\
  .withColumn("Date_Month", month(df.Date))\
  .withColumn("joined_column", concat(col("Date_Year"), lit('-'), col("Date_Month")))\
  .withColumn("Date_Monthwise", to_date(unix_timestamp("joined_column", "yyyy-MM").cast("timestamp")))\
  .drop("Date_Year")\
  .drop("Date_Month")\
  .drop("joined_column")\
  .drop("Date")

df_final = df_cleaned.select('Product_Code','Warehouse','Date_monthwise','Order_Demand')\
                      .groupBy('Product_Code','Warehouse','Date_monthwise').agg(sum('Order_Demand').alias('Agg_Orders')).orderBy('Date_monthwise')
display(df_final)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Widgets for Product Code and Warehouse 

# COMMAND ----------

try:
  dbutils.widgets.removeAll()
except:
  print('no widgets')

# COMMAND ----------

# DBTITLE 1,Widget for Product_Code
product_code = df_final.select("Product_Code").distinct().rdd.flatMap(lambda x: x).collect()

#create widget
dbutils.widgets.dropdown("Product Code", product_code[0], product_code)


# COMMAND ----------

# Selecting the paricular widget
x = dbutils.widgets.get("Product Code")

#filtering according to the selected widget
specific_data = df_final.filter(col("Product_Code") == x)

#groupBy the data according to the product sales
product_specific_data = specific_data.groupBy('Warehouse','Date_monthwise','Agg_Orders').agg(sum("Agg_Orders").alias('Orders')).orderBy("Date_monthwise").drop('Agg_Orders')

# COMMAND ----------

display(product_specific_data)

# COMMAND ----------

# DBTITLE 1,widget for warehouse
warehouse = product_specific_data.select("Warehouse").distinct().rdd.flatMap(lambda x: x).collect()

dbutils.widgets.dropdown("Warehouse", warehouse[0], warehouse)

# COMMAND ----------

# Selecting the paricular widget
x = dbutils.widgets.get("Warehouse")

#filtering according to the selected widget
ware_specific_data = product_specific_data.filter(col("Warehouse") == x)

#groupBy the data according to the product sales
warehouse_specific_data = ware_specific_data.groupBy('Date_monthwise','Orders').agg(sum("Orders").alias('Agg_Orders')).orderBy("Date_monthwise").drop('Orders')

# COMMAND ----------

display(warehouse_specific_data)

# COMMAND ----------

# DBTITLE 1,Splitting train-test
training = warehouse_specific_data.filter(col('Date_monthwise') <= '2016-09-01')
test = warehouse_specific_data.filter(col('Date_monthwise') >= '2016-10-01')

# COMMAND ----------

# DBTITLE 1,ARIMA Model
import sys
from statsmodels.tsa.arima_model import ARIMA

#Changing the datatypes
warehouse_specific_data_DF = training.toPandas()
warehouse_specific_data_DF["Date_monthwise"] = pd.to_datetime(warehouse_specific_data_DF["Date_monthwise"])
warehouse_specific_data_DF["Agg_Orders"] = pd.to_numeric(warehouse_specific_data_DF["Agg_Orders"])

#Creating the pandas series to feed for training the ARIMA model
orders = warehouse_specific_data_DF["Date_monthwise"].tolist()
dates = warehouse_specific_data_DF["Agg_Orders"].tolist()

warehouse_specific_data_DF = warehouse_specific_data_DF.sort_values(["Date_monthwise"], ascending = True)

model_data = pd.Series(data = warehouse_specific_data_DF['Agg_Orders'].values.astype('float32'), index=warehouse_specific_data_DF["Date_monthwise"])

# COMMAND ----------

# DBTITLE 1,Training the ARIMA Model 
#fit model
model = ARIMA(model_data, order=(3,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# COMMAND ----------

# DBTITLE 1,Forecasting with ARIMA model
predictions = model_fit.predict(start = "2016-10-01",end =  "2016-12-01", dynamic= True).to_dict()

#converting prediction to DF
val = predictions.values()
val = [float(x) for x in val]
mon = list(predictions.keys())
prediction_df = pd.DataFrame({'date': mon, 'Predicted': val})
prediction_sales_df = spark.createDataFrame(prediction_df)

# COMMAND ----------

# DBTITLE 1,Forecasted Demand for next Three Months for selected Widget
display(prediction_sales_df.sort("date", ascending = True))

# COMMAND ----------

# DBTITLE 1,write to sql db
"""pred_df = prediction_sales_df.sort("date", ascending = True)
pred_df = pred_df.select('date','Predicted')

#connection properties for sql server
connectionProperties = {
  "user" : "align",
  "password" : "celebal@123"
}
#write data to sql server
pred_df.write.jdbc(url="jdbc:sqlserver://aligntechpoc.database.windows.net:1433;database=Sap_Hana;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;", table="arima_forecast", properties=connectionProperties)"""

# COMMAND ----------

# DBTITLE 1,RMSE and R Squared Value
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


testt = test.select('Agg_Orders').collect()
predictionss = prediction_sales_df.select('Predicted').collect()
rms_arima = sqrt(mean_squared_error(testt, predictionss))
print('RMSE Value:',rms_arima)

r_2_arima =  r2_score(testt, predictionss)
#print('R Squared Value:',r_2_arima)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.2 Forecasting for Aggregated Orders Monthwise on Scale (0-100)

# COMMAND ----------

df_process_monthwise = df_eighty_percent.select('Product_Code','Date_Monthwise','Order_Demand')\
                      .groupBy('Date_Monthwise').agg(sum('Order_Demand').alias('Orders')).orderBy('Date_Monthwise').filter(col('Orders') > 8027352)

#converted Orders in 0-100 scale
range_df = df_process_monthwise.withColumn('Range_Orders', (col('Orders')/4685480373)).drop('Orders')
range_df = range_df.withColumn('Range_Orders',col('Range_Orders')*100)
range_df = range_df.withColumn('Range_Orders',round(col('Range_Orders'),2))

#splitting train-test
training_2 = range_df.filter(col('Date_monthwise') <= '2016-09-01')
test_2 = range_df.filter(col('Date_monthwise') >= '2016-10-01')

#Changing the datatypes
warehouse_specific_data_DF_2 = training_2.toPandas()
warehouse_specific_data_DF_2["Date_Monthwise"] = pd.to_datetime(warehouse_specific_data_DF_2["Date_Monthwise"])
warehouse_specific_data_DF_2["Range_Orders"] = pd.to_numeric(warehouse_specific_data_DF_2["Range_Orders"])

#Creating the pandas series to feed for training the ARIMA model
orders = warehouse_specific_data_DF_2["Date_Monthwise"].tolist()
dates = warehouse_specific_data_DF_2["Range_Orders"].tolist()

warehouse_specific_data_DF_2 = warehouse_specific_data_DF_2.sort_values(["Date_Monthwise"], ascending = True)

model_data_2 = pd.Series(data = warehouse_specific_data_DF_2['Range_Orders'].values.astype('float32'), index=warehouse_specific_data_DF_2["Date_Monthwise"])

# COMMAND ----------

# DBTITLE 1,ARIMA Model
#fit model
model_2 = ARIMA(model_data_2, order=(3,0,0))
model_fit_2 = model_2.fit(disp=0)
print(model_fit_2.summary())

# COMMAND ----------

# DBTITLE 1,Forecasting with ARIMA model
predictions = model_fit_2.predict(start = "2016-10-01",end =  "2016-12-01", dynamic= True).to_dict()

#converted prediction to DF
val = predictions.values()
val = [float(x) for x in val]
mon = list(predictions.keys())
prediction_df_2 = pd.DataFrame({'date': mon, 'Predicted': val})
prediction_sales_df_2 = spark.createDataFrame(prediction_df_2)

# COMMAND ----------

# DBTITLE 1,RMSE Value and R Squared Value
testt_m = test_2.select('Range_Orders').collect()
predictionss_m = prediction_sales_df_2.select('Predicted').collect()
rms_m = sqrt(mean_squared_error(testt_m, predictionss_m))
print('RMSE Value :',rms_m)

r_2_arima_m =  r2_score(testt, predictionss)
#print('R Squared Value',r_2_arima_m)

# COMMAND ----------

display(warehouse_specific_data_DF_2)

# COMMAND ----------

