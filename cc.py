import pandas as pd
df1 = pd.read_csv('FAO.csv', encoding = "ISO-8859-1")
df2 = pd.read_csv('APIv2.csv')

df1 = df1['Area Abbreviation']
df2 = df2['Country Code']
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
df1.to_csv('file1.csv',sep='\t',index=False)
df2.to_csv('file2.csv',sep='\t',index=False)

with open('file1.csv', 'r') as t1, open('file2.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

with open('update2.csv', 'w') as outFile:
   for line in filetwo:
      if line not in fileone:
         outFile.write(line)


