import pandas as pd
import numpy as np
import datetime as dt

# Read dataframes
    # I included the argument dtype on the training dataset to avoid a DtypeWarning on columns (11, 12, 31)
train = pd.read_csv('readonly/train.csv', encoding='cp1252', 
                           dtype = {'zip_code': object, 'non_us_str_code': object, 'grafitti_status': object})
test = pd.read_csv('readonly/test.csv', encoding='cp1252')
address = pd.read_csv('readonly/addresses.csv')
latlong = pd.read_csv('readonly/latlons.csv')
    
# Join address and latlong in one dataframe.  Then, join address to both train and test data
address = address.set_index('address').join(latlong.set_index('address'), how = 'left')
train = train.set_index('ticket_id').join(address.set_index('ticket_id'), how = 'left')
test = test.set_index('ticket_id').join(address.set_index('ticket_id'), how = 'left')

#print(train.head())
#print(test.head())

#print(pd.to_datetime(train['hearing_date']).dt.datetime())
#print(pd.to_datetime(train['hearing_date']).dt.values.astype(np.int64))

#print(pd.to_datetime(train['hearing_date'], errors = 'coerce'))

print(pd.to_datetime(train['hearing_date'], errors = 'coerce').dt.values.astype(np.int64))
