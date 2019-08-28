import pandas as pd
import re
def FindID(content,area):
    rstr = r"[\+\_@\$]"
    if re.search('netcool监控服务器',str(area)):
        return 'netcool'
    if re.search('端到端',str(area)):
        return 'E2E'
    if re.search('SWIFT',str(area),re.IGNORECASE):
        return 'AIX SWIFT'
    if re.search('SAA',str(area)):
        return 'AIX SWIFT'
    if re.search('SAG',str(area)):
        return 'AIX SWIFT'
    if re.search('带库相关',content):
        return 'IEF238D'
    if re.search('Netcool',content):
        return 'netcool'
    else:
        b = re.search('\.\d{2}\s*(CP\w+)?\s*([SJ]\d{7})?\s+\*?\d*\s*(\S+)', content)
        if b:
            return re.sub(rstr, '', b.group(3))
        else:
            return 'Not Found!'
def FindKey(data):
    idList, summaryList,sysList = data['id'], data['summary'],data['sysname']
    messageID = []
    KeyDict={}
    for i in range(len(summaryList)):
        key = FindID(summaryList[i],sysList[i])
        messageID.append(key)
    KeyDict = {'id': idList,'messageID':messageID,'summary':summaryList}
    out_df= pd.DataFrame(KeyDict, columns=['id', 'messageID','summary'])
    return out_df
dataFile = 'data/all.csv'
data = pd.read_csv(dataFile,encoding="gbk")
result=FindKey(data)
result.to_csv("result/testresult.csv",index=False)