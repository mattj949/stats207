import pandas as pd

# Define a helper function to process the data
def process_data(df, label, start_date, end_date):
    
    # Convert the Date column from a string to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # We only want dates prior to this cutoff
    df = df.loc[(df['Date'] <= end_date)]
    
    # We only want dates after this cutoff
    df = df.loc[(df['Date'] >= start_date)]
    
    # Reset Index
    df.reset_index(inplace=True, drop=True)
    
    # Carries forward old prices, so we aren't using future information
    df.fillna(method = 'ffill', inplace=True) 
    
    # Drop the 'Ticker' column
    df = df.drop(columns = ['Ticker'], axis=1)
    
    # Rename the 'Close' column with the passed label
    df = df.rename(columns={'Close': label})
    #df = df.drop(columns = 'Close')
    
    return df

def slice_dataset(data, start_date, end_date):

    # Get the indices of the start date and the end date
    start_index = int(np.where(data[:,0] == pd.to_datetime(start_date))[0])
    end_index = int(np.where(data[:,0] == pd.to_datetime(end_date))[0])
    
    # Get all of the input data (X)
    X_data = data.astype('float32')   
    
    # Get all of the input data (X) for the desired date range
    X_data = X_data[start_index:end_index+1]

    # Get all of the output data (Y) for the desired date range
    y_data = data[start_index:end_index+1,1].astype('float32')
    
    return X_data, y_data
