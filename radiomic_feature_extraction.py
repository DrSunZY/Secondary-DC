from csv import excel
from email.mime import image
from msilib.schema import Feature
import radiomics
import numpy as np
import pandas as pd
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor()
data_A = pd.DataFrame()
data_A
list_one = np.arange(1,115) # 根 据病例数多少来设置
list_one
#设置循环连续读取影像数据
for m in list_one:
	imgname = 'C:/Users/swift/Desktop/ADC/'+str(m)+'.nrrd'
	maskname = 'C:/Users/swift/Desktop/ADC/'+str(m)+'label.nrrd'
	featureVector = extractor.execute(imgname,maskname)
	df_add = pd.DataFrame([featureVector])
	data_A = pd.concat([data_A, df_add])
print("fuck")
data_A
#转移到excel
data_A. to_excel('data_ADC.xlsx')
