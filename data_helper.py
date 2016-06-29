import pandas as pd
import numpy as np
import logging
import logging.handlers
import datetime

holiday = [True] * 3 + [False] * 28

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def time_to_slice(row,*args):
    str_time = row['Time']
    tmp = map(lambda x: int(x), str_time.split(':'))
    if args:
        return (tmp[0]*12 + tmp[1]/5 + 1)*0.5
    else:
        return tmp[0] * 6 + tmp[1] / 10 + 1


def is_holiday(date_str):
    date_split = date_str.split('-')
    idx = int(date_split[2]) - 1
    return holiday[idx]


def get_order_label(row):
    str_time = str(row['driver_id'])
    if str_time == 'nan':
        return 1.0
    else:
        return 0.0


def get_row_sum(row):
    return np.sum(row[['tlevel_1', 'tlevel_2', 'tlevel_3', 'tlevel_4']])

def process_labels(row):
    time_slice = float(row['label_time_slice'])
    labels = row['label']
    labelset = {}
    for l in labels:
        labelset[l[0]] = l[1]
    #(row['f1'],row['f2'],row['f3'],row['label']) = (0.0 if not (time_slice-3+i) in labelset else labelset[time_slice-3+i] for i in range(4))
    res = [0.0 if not (time_slice-3+i) in labelset else labelset[time_slice-3+i] for i in range(4)]
    return pd.Series({'f1':res[0], 'f2':res[1], 'f3':res[2], 'gap':res[3]})

def get_cluster_df(data_dir):
    filename = data_dir + 'cluster_map/cluster_map'
    cluster_names = ['district_hash', 'district_id']
    cluster_type = {'district_hash': pd.np.string_, 'district_id': pd.np.string_}
    cluster_df = pd.read_table(filename, dtype=cluster_type, names=cluster_names)
    return cluster_df


def get_weather_df(data_dir, data_date):
    filename = data_dir + 'weather_data/weather_data_' + data_date
    if data_dir.find('test') >= 0:
        filename += '_test'

    weather_column_types = {'Date': pd.np.string_,
                            'Time': pd.np.string_,
                            'Weather': pd.np.int64,
                            'Temperature': pd.np.float64,
                            'PM2.5': pd.np.float64}
    weather_column_names = ['Date', 'Time', 'Weather', 'Temperature', 'PM2.5']
    weather_data_frame = pd.read_table(filename,
                                       delim_whitespace=True,
                                       names=weather_column_names,
                                       dtype=weather_column_types)
    # weather_data_frame['holiday'] = weather_data_frame['Date'].apply(is_holiday)
    weather_data_frame['Date'] = pd.to_datetime(weather_data_frame['Date'])
    weather_data_frame['Weekday'] = weather_data_frame['Date'].dt.dayofweek
    weather_data_frame['Weekday'] = weather_data_frame['Weekday']

    weather_data_frame['time_slice'] = weather_data_frame.apply(time_to_slice,args=('weather',), axis=1)
    drop_column_names = ['Date', 'Time']
    weather_data_frame = weather_data_frame.drop(drop_column_names, axis=1)
    logger.info('Weather DataFrame Ready.')
    return weather_data_frame


def get_traffic_df(data_dir, data_date):
    cluster_df = get_cluster_df(data_dir)

    filename = data_dir + 'traffic_data/traffic_data_' + data_date
    if data_dir.find('test') >= 0:
        filename += '_test'

    traffic_column_names = ['district_hash', 't1', 't2', 't3', 't4', 'Date', 'Time']
    traffic_data_frame = pd.read_table(filename,
                                       delim_whitespace=True,
                                       names=traffic_column_names)
    traffic_data_frame = pd.merge(traffic_data_frame, cluster_df,
                                  left_on='district_hash', right_on='district_hash', how='inner')
    traffic_data_frame = traffic_data_frame.rename(columns={'district_id': 'district'})

    traffic_data_frame['time_slice'] = traffic_data_frame.apply(time_to_slice, axis=1)

    traffic_data_frame['tlevel_1'] = traffic_data_frame.apply(lambda row: row['t1'].split(':')[1], axis=1).astype(float)
    traffic_data_frame['tlevel_2'] = traffic_data_frame.apply(lambda row: row['t2'].split(':')[1], axis=1).astype(float)
    traffic_data_frame['tlevel_3'] = traffic_data_frame.apply(lambda row: row['t3'].split(':')[1], axis=1).astype(float)
    traffic_data_frame['tlevel_4'] = traffic_data_frame.apply(lambda row: row['t4'].split(':')[1], axis=1).astype(float)

    drop_column_names = ['Date', 'Time', 't1', 't2', 't3', 't4', 'district_hash']
    traffic_data_frame = traffic_data_frame.drop(drop_column_names, axis=1)
    logger.info('Traffic DataFrame Ready.')
    return traffic_data_frame


def get_order_data(data_dir, data_date):
    cluster_df = get_cluster_df(data_dir)

    filename = data_dir + 'order_data/order_data_' + data_date
    if data_dir.find('test') >= 0:
        filename += '_test'

    order_column_names = ['order_id',
                          'driver_id',
                          'passenger_id',
                          'start_district_hash',
                          'dest_district_hash',
                          'Price',
                          'Date',
                          'Time']
    order_data_frame = pd.read_table(filename,
                                     delim_whitespace=True,
                                     names=order_column_names)

    order_data_frame['time_slice'] = order_data_frame.apply(time_to_slice, axis=1)
    order_data_frame = pd.merge(order_data_frame, cluster_df,
                                left_on='start_district_hash', right_on='district_hash', how='inner')
    order_data_frame = order_data_frame.rename(columns={'district_id': 'start_district'})
    order_data_frame['label'] = order_data_frame.apply(get_order_label, axis=1)
    order_data_frame['indicator'] = order_data_frame.apply(lambda row: 1, axis=1)

    drop_column_names = [order_column_names[i] for i in [0, 1, 2, 3, 4, 6, 7]]
    drop_column_names = drop_column_names + ['district_hash_x', 'district_hash_y']
    logger.info('Order DataFrame Ready.')
    return order_data_frame


def get_df_time_slices(**kwargs):
    if kwargs['dataframe'] is not None:
        time_slice_list = kwargs['dataframe']['time_slice'].tolist()
        return np.array(time_slice_list)
    if kwargs['df_type'] is not None:
        df_access_dict = {'weather': get_weather_df,
                          'traffic': get_traffic_df,
                          'order': get_order_data}
        data_dir = kwargs['data_dir']
        data_date = kwargs['data_date']
        data_frame = df_access_dict[kwargs['df_type']](data_dir, data_date)
        time_slice_list = data_frame['time_slice'].unique()
        return np.array(time_slice_list)


def get_closest_time_slice(row, *args):
    if isinstance(row, pd.Series):
        x = row['time_slice']
        district = row[args[0]]
        time_dict = args[1]
        time_slice = np.array(time_dict[district])
    else:
        x = row
        time_slice = args[1]
    idx = np.argmin(np.abs(time_slice - x))
    return int(time_slice[idx])


def get_upmost_time_slice(row, *args):
    if isinstance(row, pd.Series):
        x = row['test_time']
        district = row[args[0]]
        time_dict = args[1]
        time_slice = np.array(time_dict[district])
    else:
        x = row
        time_slice = args[1]
    tmp = time_slice[time_slice >= x]
    if len(tmp) > 0:
        return np.min(tmp)
    else:
        return np.max(time_slice)


def merge_df_helper(data_dir, day, test_time_slices):
    traffic_df = get_traffic_df(data_dir, day)
    weather_df = get_weather_df(data_dir, day)
    order_df = get_order_data(data_dir, day)

    #grouped = order_df.groupby(['time_slice', 'start_district', 'dest_district']) #nodst
    grouped = order_df.groupby(['time_slice', 'start_district'])
    tmp_df = grouped.agg({'Price': 'sum', 'label': 'sum', 'indicator': 'sum'})
    tmp_df['start_district'] = tmp_df.index.get_level_values('start_district').astype(str)
    #tmp_df['dest_district'] = tmp_df.index.get_level_values('dest_district').astype(str) #nodst
    tmp_df['time_slice'] = tmp_df.index.get_level_values('time_slice')
    tmp_df = tmp_df[tmp_df['time_slice'] <= max(test_time_slices)]
    tmp_df.index = range(len(tmp_df))
    logger.info('Reindexing Order DataFrame')

    tmp_df['test_time'] = tmp_df['time_slice'].apply(get_upmost_time_slice, args=('test_time', test_time_slices))

    # Merge Weather DataFrame by picking the nearest time slice
    weather_time_slices = get_df_time_slices(dataframe=weather_df)
    tmp_df['weather_time_slice'] = tmp_df['test_time'].apply(get_upmost_time_slice, args=('weather', weather_time_slices))
    tmp_df = pd.merge(tmp_df, weather_df,
                      left_on='weather_time_slice', right_on='time_slice', how='inner')
    tmp_df = tmp_df.rename(columns={'time_slice_x': 'time_slice'}).drop('time_slice_y', axis=1)
    logger.info('Merge Weather DataFrame Complete.')
    # Merge Traffic DataFrame by picking the nearest time slice

    district_time_df = traffic_df[['time_slice', 'district']].groupby('district').agg(lambda x: list(x))
    district_time_dict = district_time_df.T.to_dict('records')[0]
    tmp_df['src_time_slice'] = tmp_df.apply(get_closest_time_slice, args=('start_district', district_time_dict), axis=1)

    tmp_df1 = pd.merge(tmp_df, traffic_df,
                      left_on=['start_district', 'src_time_slice'],
                      right_on=['district', 'time_slice'],
                      how='inner')
    tmp_df2 = tmp_df1.rename(columns={'time_slice_x': 'time_slice',
                                      'tlevel_1': 'src_t1',
                                      'tlevel_2': 'src_t2',
                                      'tlevel_3': 'src_t3',
                                      'tlevel_4': 'src_t4'}).drop('time_slice_y', axis=1)
    logger.info('Merge src_district with Traffic DataFrame Complete.')

    df = tmp_df2
    df['time_slice'] = df['time_slice'].apply(np.ceil)
    #df = df.drop(['src_time_slice', 'dst_time_slice', 'weather_time_slice', 'district_x', 'district_y'], axis=1) #nodst
    df = df.drop(['src_time_slice', 'weather_time_slice'], axis=1)

    df['label'] = df.apply(lambda row: (row['time_slice'], row['label']), axis=1)
    df = df.drop(['time_slice'], axis=1)
    #grouped = df.groupby(['test_time', 'start_district', 'dest_district']) #nodst
    grouped = df.groupby(['test_time', 'start_district'])
    '''final_df = grouped.agg({'Price': 'sum',
                      'indicator': 'sum',
                      'Weather': 'max',
                      'Temperature': 'max',
                      'PM2.5': 'max',
                      'Weekday': 'max',
                      'src_t1': 'mean',
                      'src_t2': 'mean',
                      'src_t3': 'mean',
                      'src_t4': 'mean',
                      'dst_t1': 'mean',
                      'dst_t2': 'mean',
                      'dst_t3': 'mean',
                      'dst_t4': 'mean'})'''# nodst
    final_df = grouped.agg({'Price': 'sum',
                      'indicator': 'sum',
                      'Weather': 'max',
                      'Temperature': 'max',
                      'PM2.5': 'max',
                      'Weekday': 'max',
                      'src_t1': 'mean',
                      'src_t2': 'mean',
                      'src_t3': 'mean',
                      'src_t4': 'mean'})
    final_df['Price'] = final_df['Price'].div(final_df['indicator'], axis=0)
    final_df['label_time_slice'] = final_df.index.get_level_values('test_time')
    final_df['src'] = final_df.index.get_level_values('start_district')
    final_df['label'] = grouped['label'].apply(list)

    final_df = pd.concat([final_df,final_df.apply(process_labels, axis=1)],axis = 1)
    final_df.index = range(len(final_df))
    logger.info('Time_Src_Dst DataFrame Complete.')
    return final_df


def merge_df(data_dir, days):
    if isinstance(days, list):
        results = []
        for day in days:
            day_result = []
            logger.info('=====================================')
            logger.info('Now processing ' + day)
            for itr in (np.arange(4, 144, 4),np.arange(5, 144, 4),np.arange(6, 144, 4),np.arange(7, 144, 4)):
                logger.info('=====================================')
                logger.debug('Processing tst  time slices:\n' + str(itr))
                test_time_slices = itr
                result = merge_df_helper(data_dir, day, test_time_slices)
                day_result.append(result)
            day_df = pd.concat(day_result)
            logger.info('saving training table for '+day)
            day_df.to_csv('train_'+day+'.csv')
            logger.info('saving complete.')
            results.append(day_df)
        res_df = pd.concat(results)
    else:
        results = []
        logger.info('=====================================')
        logger.info('Now processing ' + days)
        for itr in (np.arange(4, 144, 4),np.arange(5, 144, 4),np.arange(6, 144, 4),np.arange(7, 144, 4)):
            logger.info('=====================================')
            logger.info('Now processing ' + days)
            logger.debug('Processing test time slices:\n' + str(itr))
            test_time_slices = itr
            result = merge_df_helper(data_dir, days, test_time_slices)
            results.append(result)
        res_df = pd.concat(results)
        logger.info('saving training table for '+days)
        res_df.to_csv('train_'+days+'.csv')
        logger.info('saving complete.')
    res_df = res_df.drop(['label'],axis=1)
    return res_df


def test_df(data_dir, days):
    if isinstance(days, list):
        results = []
        for day in days:
            day_result = []
            logger.info('=====================================')
            logger.info('Now processing ' + day)
            for itr in [(np.arange(46, 144, 12))]:
                logger.info('=====================================')
                logger.debug('Processing tst  time slices:\n' + str(itr))
                test_time_slices = itr
                result = merge_df_helper(data_dir, day, test_time_slices)
                day_result.append(result)
            day_df = pd.concat(day_result)
            logger.info('saving testing table: '+'test_'+day+'.csv')
            day_df.to_csv('test_'+day+'.csv')
            logger.info('saving complete.')
            results.append(day_df)
        res_df = pd.concat(results)
    else:
        results = []
        logger.info('=====================================')
        logger.info('Now processing ' + days)
        for itr in [(np.arange(46, 144, 12))]:
            logger.info('=====================================')
            logger.info('Now processing ' + days)
            logger.debug('Processing test time slices:\n' + str(itr))
            test_time_slices = itr
            result = merge_df_helper(data_dir, days, test_time_slices)
            results.append(result)
        res_df = pd.concat(results)
        logger.info('saving testing table: '+'test_'+day+'.csv')
        res_df.to_csv('test_'+days+'.csv')
        logger.info('saving complete.')
    res_df = res_df.drop(['label'],axis=1)
    return res_df

def valid_df(data_dir, days):
    results = []
    for day in days:
        day_result = []
        logger.info('=====================================')
        logger.info('Now processing ' + day)
        for itr in [np.arange(46, 144, 12)]:
            logger.info('=====================================')
            logger.debug('Processing tst time slices:\n' + str(itr))
            test_time_slices = itr
            result = merge_df_helper(data_dir, day, test_time_slices)
            day_result.append(result)
        day_df = pd.concat(day_result)
        logger.info('saving testing table: '+'test_'+day+'.csv')
        day_df.to_csv('test_'+day+'.csv')
        logger.info('saving complete.')
        results.append(day_df)
    res_df = pd.concat(results)
    res_df = res_df.drop(['label'],axis=1)
    res_df.to_csv('valid_with_dummies.csv')
    return res_df

if __name__ == '__main__':
    test_data_dir = 'data/test_set_2/'
    test_days = ['2016-01-' + str(i) for i in [23, 25, 27, 29, 31]]
    test = test_df(test_data_dir, test_days)
