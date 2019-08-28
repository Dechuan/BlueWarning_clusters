import pandas as pd
import re
def findID(line):
    if re.search('带库相关',line):
        return 'IEF238D'
    if re.search('SWIFTNet',line):
        return 'SWIFTNet'
    if re.search('Netcool',line):
        return 'Netcool'
    else:
        b=re.search('\.\d{2}\s*(CP\w+)?\s*([SJ]\d{7})?\s+(\S+)',line)
        if b!=None:
            return b.group(3)
        else:
            return None
dataFile = 'data/month12.csv'
data = pd.read_csv(dataFile,encoding="gb2312")
idList, summaryList = data['id'], data['summary']
messageID=[]
for i in range(len(summaryList)):
    key=findID(summaryList[i])
    messageID.append(key)
ModelDict = {'id': idList,'summary':summaryList,'messageID': messageID}
result=pd.DataFrame(ModelDict, columns=['id', 'summary','messageID'])
result.to_csv("result/testresult.csv",index=False)
