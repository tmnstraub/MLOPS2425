def age_(data):
    
    data['bin_age'] = 0  
    data.loc[(data['age'] <= 35) & (data['age'] >= 18),'bin_age'] = 1
    data.loc[(data['age'] <= 60) & (data['age'] >= 36),'bin_age'] = 2
    data.loc[data['age'] >=61,'bin_age'] = 3
    
    return data

def campaign_(data):
    
    
    data.loc[data['campaign'] == 1,'campaign'] = 1
    data.loc[(data['campaign'] >= 2) & (data['campaign'] <= 3),'campaign'] = 2
    data.loc[data['campaign'] >= 4,'campaign'] = 3
    
    return data

def duration_(data):
    
    data['t_min'] = 0
    data['t_e_min'] = 0
    data['e_min']=0
    data.loc[data['duration'] <= 5,'t_min'] = 1
    data.loc[(data['duration'] > 5) & (data['duration'] <= 10),'t_e_min'] = 1
    data.loc[data['duration'] > 10,'e_min'] = 1
    
    return data

def pdays_(data):
    data['pdays_not_contacted'] = 0
    data['months_passed'] = 0
    data.loc[data['pdays'] == -1 ,'pdays_not_contacted'] = 1
    data['months_passed'] = data['pdays']/30
    data.loc[(data['months_passed'] >= 0) & (data['months_passed'] <=2) ,'months_passed'] = 1
    data.loc[(data['months_passed'] > 2) & (data['months_passed'] <=6),'months_passed'] = 2
    data.loc[data['months_passed'] > 6 ,'months_passed'] = 3
    
    return data


def balance_(data):
    data['Neg_Balance'] = 0
    data['No_Balance'] = 0
    data['Pos_Balance'] = 0
    data.loc[~data['balance']<0,'Neg_Balance'] = 1
    data.loc[data['balance'] < 1,'bin_Balance'] = 0
    data.loc[(data['balance'] >= 1) & (data['balance'] < 100),'bin_Balance'] = 1
    data.loc[(data['balance'] >= 100) & (data['balance'] < 500),'bin_Balance'] = 2
    data.loc[(data['balance'] >= 500) & (data['balance'] < 2000),'bin_Balance'] = 3
    data.loc[(data['balance'] >= 2000) & (data['balance'] < 5000),'bin_Balance'] = 4
    data.loc[data['balance'] >= 5000,'bin_Balance'] = 5
    
    return data


