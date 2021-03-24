use StockPrediction_DS
--adding data in source tables

CREATE TABLE ISEQOverallHistorical (
ISEQ_ID int NOT NULL,
Date_ISEQ date, 
Price_ISEQ int,
Open_ISEQ int,
High_ISEQ int,
Low_ISEQ int,
Volume_ISEQ int,
ChangePercent_ISEQ decimal (3,2),
CONSTRAINT pk_ISEQ_ID PRIMARY KEY (ISEQ_ID),
);



CREATE TABLE StockDetails (
Company_name varchar(100) NOT NULL,
InstrumentName nvarchar(100), 
MarketCap bigint,
ISEQWeightingPercent int,
DescriptionStockDetails nvarchar(800),
Website nvarchar(500),
DepartmentDetails nvarchar(100),
CONSTRAINT pk_Company_name PRIMARY KEY (Company_name),
);

CREATE TABLE StockHistorical (
Historical_ID int NOT NULL,
Company_name varchar(100) NOT NULL,
Date_Stock date, 
Open_Stock int,
High_Stock int,
Low_Stock int,
Close_Stock int,
AdjClose_Stock int,
Volume_Stock bigint,
CONSTRAINT pk_Historical_ID PRIMARY KEY (Historical_ID),
);


CREATE TABLE IrishTimesAllStockNews (
News_ID int NOT NULL,
Company_name_News varchar(100),
Date_News date, 
Headline_News nvarchar(500),
Description_News nvarchar(500),
CONSTRAINT pk_News_ID PRIMARY KEY (News_ID)
);


CREATE TABLE IrishTimesPolitical (
Political_ID int NOT NULL,
Name_political varchar(50),
Date_Political date, 
Headline_Political nvarchar(500),
Description_Political nvarchar(500),
CONSTRAINT pk_Political_ID PRIMARY KEY (Political_ID),
);


CREATE TABLE IrishTimesEconomical (
Economical_ID int NOT NULL,
Name_Economical varchar(50),
Date_Economical date, 
Headline_Economical nvarchar(500),
Description_Economical nvarchar(500),
CONSTRAINT pk_Economical_ID PRIMARY KEY (Economical_ID),
);


CREATE TABLE IrishTimesSocial (
Social_ID int NOT NULL,
Name_Social varchar(50),
Date_Social date, 
Headline_Social nvarchar(500),
Description_Social nvarchar(500),
CONSTRAINT pk_Social_ID PRIMARY KEY (Social_ID),
);


CREATE TABLE IrishTimesTech (
Tech_ID int NOT NULL,
Name_Tech varchar(50),
Date_Tech date, 
Headline_Tech nvarchar(500),
Description_Tech nvarchar(500),
CONSTRAINT pk_Tech_ID PRIMARY KEY (Tech_ID),
);


CREATE TABLE IrishTimesEnvior (
Envior_ID int NOT NULL,
Envior_Tech varchar(50),
Date_Envior date, 
Headline_Envior nvarchar(500),
Description_Envior nvarchar(500),
CONSTRAINT pk_Envior_ID PRIMARY KEY (Envior_ID),
);


CREATE TABLE IrishTimesLegal (
Legal_ID int NOT NULL,
Legal_Tech varchar(50),
Date_Legal date, 
Headline_Legal nvarchar(500),
Description_Legal nvarchar(500),
CONSTRAINT pk_Legal_ID PRIMARY KEY (Legal_ID),
);




use stock_analysis_dw
---creating data base dimentions


create table pestal_dim(
pestal_ID int NOT NULL,
name_pestal nvarchar(50),
date date, 
Headline_pestal nvarchar(500),
Description_pestal nvarchar(500),
CONSTRAINT pk_pestal_ID PRIMARY KEY (pestal_ID),
);


select * from ISEQOverallHistorical_dim 
CREATE TABLE ISEQOverallHistorical_dim (
ISEQ_ID int NOT NULL,
date date, 
Price_ISEQ int,
Open_ISEQ int,
High_ISEQ int,
Low_ISEQ int,
Volume_ISEQ int,
ChangePercent_ISEQ decimal (3,2),
CONSTRAINT pk_ISEQ_ID PRIMARY KEY (ISEQ_ID),
);


CREATE TABLE StockDetails_dim (
Company_name varchar(100) NOT NULL,
InstrumentName nvarchar(100), 
MarketCap bigint,
ISEQWeightingPercent int,
DescriptionStockDetails nvarchar(800),
Website nvarchar(500),
DepartmentDetails nvarchar(100),
CONSTRAINT pk_Company_name PRIMARY KEY (Company_name),
);


CREATE TABLE StockHistorical_dim (
Historical_ID int NOT NULL,
Company_name varchar(100) NOT NULL,
date date, 
Open_Stock int,
High_Stock int,
Low_Stock int,
Close_Stock int,
AdjClose_Stock int,
Volume_Stock bigint,
CONSTRAINT pk_Historical_ID PRIMARY KEY (Historical_ID),
);


CREATE TABLE IrishTimesAllStockNews_dim (
News_ID int NOT NULL,
Company_name varchar(100),
date date, 
Headline_News nvarchar(500),
Description_News nvarchar(500),
CONSTRAINT pk_News_ID PRIMARY KEY (News_ID)
);


select * from IrishTimesAllStockNews_dim
CREATE TABLE datedim (
Fulldate date NOT NULL,
date int,
Month int, 
Year int,
Day nvarchar(100),
Quater nvarchar(100),
CONSTRAINT pk_Fulldate PRIMARY KEY (Fulldate)
);


CREATE TABLE stock_fact (
Historical_ID int,
Company_name varchar(100) , 
News_ID int,
Fulldate date,
Price_diff_high_low_ISEQ int,
);


CREATE TABLE Overall_ISEQ_fact(
pestal_ID int ,
ISEQ_ID int ,
Date date,
High_low_diff int,
Price_diff int
);


---Fact table creation queries

SELECT IrishTimesAllStockNews_dim.date,StockHistorical_dim.Company_name,StockHistorical_dim.Historical_ID,IrishTimesAllStockNews_dim.News_ID,
StockHistorical_dim.High_Stock - (StockHistorical_dim.Low_Stock) as high_low
FROM   IrishTimesAllStockNews_dim INNER JOIN
StockHistorical_dim ON IrishTimesAllStockNews_dim.date = StockHistorical_dim.date



SELECT ISEQOverallHistorical_dim.date,pestal_dim.pestal_ID,ISEQOverallHistorical_dim.ChangePercent_ISEQ,
ISEQOverallHistorical_dim.ISEQ_ID,ISEQOverallHistorical_dim.High_ISEQ - (ISEQOverallHistorical_dim.Low_ISEQ) as high_low,
ISEQOverallHistorical_dim.Price_ISEQ - coalesce(lag(ISEQOverallHistorical_dim.Price_ISEQ)
over (order by ISEQOverallHistorical_dim.ISEQ_ID), 1) as Price_diff
From ISEQOverallHistorical_dim INNER JOIN pestal_dim ON ISEQOverallHistorical_dim.date=pestal_dim.date


---SSRS Report Queries



select Overall_ISEQ_fact.High_low_diff,Overall_ISEQ_fact.date,pestal_dim.Headline_pestal,pestal_dim.Description_pestal from Overall_ISEQ_fact
INNER JOIN pestal_dim on Overall_ISEQ_fact.pestal_ID=pestal_dim.pestal_ID where Overall_ISEQ_fact.High_low_diff>49 


select pestal_dim.name_pestal,stock_fact.Fulldate,count(stock_fact.Price_diff_high_low_ISEQ) as 'Number of Articles'
from stock_fact inner join pestal_dim on pestal_dim.date=stock_fact.Fulldate where
stock_fact.Price_diff_high_low_ISEQ>4.9 group by stock_fact.Fulldate,pestal_dim.name_pestal


select p.date,p.name_pestal,count(p.name_pestal) as cnt from pestal_dim p group by date,name_pestal


select name_pestal,count(*) as Frequency from pestal_dim group by name_pestal

