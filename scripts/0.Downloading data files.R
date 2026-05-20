#install.packages("jsonlite") #If not installed yet
library(jsonlite)

data <- fromJSON("master_2138521_01.json") #Change the number, depending on the file

write.table(data[[1]][[1]], "Kijkgeschiedenis.txt", row.names = FALSE)
write.table(data[[3]][[3]], "Likelijst.txt", row.names = FALSE)
