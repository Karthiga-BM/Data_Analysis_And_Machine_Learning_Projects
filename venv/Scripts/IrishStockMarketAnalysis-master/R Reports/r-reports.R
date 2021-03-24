#Installing Package
install.packages("RODBC")
#loading the package
require(RODBC)
#establishing connection to database
conn = odbcDriverConnect("Driver={SQL Server};

                         server=DESKTOP-JAG58LJ;

                         database=stock_analysis_dw;

                         trusted_connection=true")
#storing all the neccesary columns to respective variable
pestal_ID=sqlQuery(conn,"select pestal_ID from pestal_dim" )
name_pestal=sqlQuery(conn,"select name_pestal from pestal_dim" )
date=sqlQuery(conn,"select date from pestal_dim" )
Headline_pestal=sqlQuery(conn,"select Headline_pestal from pestal_dim" )
Description_pestal=sqlQuery(conn,"select Headline_pestal from pestal_dim" )
Price_ISEQ=sqlQuery(conn,"select Price_ISEQ from ISEQOverallHistorical_dim" )
date_ISEQ=sqlQuery(conn,"select date from ISEQOverallHistorical_dim" )
ISEQ=sqlQuery(conn,"select * from ISEQOverallHistorical_dim" )
abbey=sqlQuery(conn,"Select Close_Stock from StockHistorical_dim where Company_name='ABBEY'")
aib=sqlQuery(conn,"Select Close_Stock from StockHistorical_dim where Company_name='AIB'")
stockHistorical=sqlQuery(conn,"Select * from StockHistorical_dim")
fact1=sqlQuery(conn,"Select * from stock_fact")
fact2=sqlQuery(conn,"Select * from Overall_ISEQ_fact")
write.csv(fact2,"C:/Users/bhara/OneDrive/Desktop/fact1.csv", row.names = FALSE)
summary=summary(name_pestal)
#plot1
table=table(summary)
pie=pie(table,main="Factors in Pestal data")
#plot2
a <- ggplot(ISEQ, aes(x = ISEQ$Price_ISEQ))
a + geom_area(stat = "bin")
# creating Default scatter plot of ISEQQ price over last 15 years
p <- ggplot(ISEQ, aes(ISEQ$Price_ISEQ, ISEQ$date))
p + geom_point()
abbey$Close_Stock
#creating Line plot of all the availabe shares with respect to their prices for last 115 days
ds=data.frame(abbey$Close_Stock[1:115],aib$Close_Stock)
a=a <- ggplot(ds, aes(abbey$Close_Stock[1:115],aib$Close_Stock))
a+geom_line()
ggplot(stockHistorical, aes(x=stockHistorical$date[100], y=stockHistorical$Close_Stock[100], group=stockHistorical$Company_name[100])) +
  geom_line(aes(linetype=stockHistorical$Company_name[100], color = stockHistorical$Company_name))+
  geom_point(aes(shape=stockHistorical$Company_name[100], color = stockHistorical$Company_name))
