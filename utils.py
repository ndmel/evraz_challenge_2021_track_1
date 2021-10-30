import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def metric(answers, user_csv):
    """
    General metric for the challenge
    """
    delta_c = np.abs(np.array(answers['C']) - np.array(user_csv['C']))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers['TST']) - np.array(user_csv['TST']))
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(answers['C'])

    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def plot_feature_importance(importance, names, model_type):
    """
    Plot for feature importance
    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def _feature_stats_ras(df, produv_df):
    """
    Статистики по полям RAS / RAS_MUL_POL / RAS_REL_POL - sum, median, mean, min, max, std
    """
    return df.merge(produv_df.groupby(['NPLV'], as_index=False).agg(NPLV_SUM_RAS=('RAS', 'sum'),
                                                                    NPLV_MEDIAN_RAS=('RAS', 'median'),
                                                                    NPLV_MEAN_RAS=('RAS', 'mean'),
                                                                    NPLV_MIN_RAS=('RAS', 'min'),
                                                                    NPLV_MAX_RAS=('RAS', 'max'),
                                                                    NPLV_STD_RAS=('RAS', 'std')),
                    on=['NPLV'], how='left', validate='1:1') \
        .merge(produv_df.groupby(['NPLV'], as_index=False).agg(NPLV_SUM_RAS_MUL_POL=('RAS_MUL_POL', 'sum'),
                                                               NPLV_MEDIAN_RAS_MUL_POL=('RAS_MUL_POL', 'median'),
                                                               NPLV_MEAN_RAS_MUL_POL=('RAS_MUL_POL', 'mean'),
                                                               NPLV_MIN_RAS_MUL_POL=('RAS_MUL_POL', 'min'),
                                                               NPLV_MAX_RAS_MUL_POL=('RAS_MUL_POL', 'max'),
                                                               NPLV_STD_RAS_MUL_POL=('RAS_MUL_POL', 'std')),
               on=['NPLV'], how='left', validate='1:1') \
        .merge(produv_df.groupby(['NPLV'], as_index=False).agg(NPLV_SUM_RAS_REL_POL=('RAS_REL_POL', 'sum'),
                                                               NPLV_MEDIAN_RAS_REL_POL=('RAS_REL_POL', 'median'),
                                                               NPLV_MEAN_RAS_REL_POL=('RAS_REL_POL', 'mean'),
                                                               NPLV_MIN_RAS_REL_POL=('RAS_REL_POL', 'min'),
                                                               NPLV_MAX_RAS_REL_POL=('RAS_REL_POL', 'max'),
                                                               NPLV_STD_RAS_REL_POL=('RAS_REL_POL', 'std')),
               on=['NPLV'], how='left', validate='1:1')


def _feature_stats_chugun(df, chugun_df):
    '''
    Базовые статистики по чугуну
    '''
    return df.merge(chugun_df.drop('DATA_ZAMERA', 1),
                    on=['NPLV'], how='left', validate='1:1')


def _feature_stats_chronom(df, chronom_df):
    '''
    Статистики по хронометражу
    '''

    chronom_res = []
    for nplv, group in chronom_df.groupby(['NPLV']):

        # Была ли додувка в плавке
        has_doduvka_t = None
        has_doduvka_c = None
        if 'Додувка на t' in group.NOP.unique():
            has_doduvka_t = 1
        if 'Додувка на C' in group.NOP.unique():
            has_doduvka_c = 1

        # Времена
        plavka_nach = group[group.NOP == 'Завалка лома'].VR_NACH.min()
        produvka_nach = group[group.NOP == 'Продувка'].VR_NACH.min()

        # Потраченный О2 до начала продувки
        all_o2 = group[group.VR_NACH < produvka_nach]['O2'].sum()
        all_operations_count = group[group.VR_NACH < produvka_nach].shape[0]

        # Время с начала плавки, время с начала работы над плавкой
        time_to_produv = (produvka_nach - plavka_nach).total_seconds()
        time_to_produv_from_start = (produvka_nach - group.VR_NACH.min()).total_seconds()

        # Запоминаем
        chronom_res.append([nplv, has_doduvka_t, has_doduvka_c, all_o2, time_to_produv, time_to_produv_from_start,
                            all_operations_count])

    chronom_res_df = pd.DataFrame(chronom_res,
                                  columns=['NPLV', 'HAS_DODUVKA_T', 'HAS_DODUVKA_C', 'PLAVKI_O2',
                                           'TIME_FROM_PLAVKI_TO_PRODUV', 'TIME_FROM_START_TO_PRODUV',
                                           'OPERATIONS_COUNT_TO_PRODUV'])

    return df.merge(chronom_res_df, on=['NPLV'], how='left', validate='1:1')


def _feature_operations_chronom(df, chronom_df):
    """
    Какие операции совершались до продувки
    """

    # Добавим начало продувки
    chronom_with_time_df = chronom_df.merge(
        chronom_df[chronom_df.NOP == 'Продувка'][['NPLV', 'VR_NACH']] \
            .rename(columns={'VR_NACH': 'VR_NACH_PRODUV'}),
        on=['NPLV'], how='left', validate='m:1')

    # Какие операции и как часто совершались до начала продувки
    chronom_stats_df = chronom_with_time_df[
        chronom_with_time_df.VR_NACH <= chronom_with_time_df.VR_NACH_PRODUV].groupby(['NPLV', 'NOP_le'],
                                                                                     as_index=False).size() \
        .pivot_table(index='NPLV', columns='NOP_le', values='size') \
        .add_suffix('_CHRONO_NOP').reset_index()
    chronom_stats_df.columns.name = None

    return df.merge(chronom_stats_df.fillna(0), on=['NPLV'], how='left', validate='1:1')


def _feature_time_since_chugun(df, produv_df, chugun_df):
    """
    Как много времени прошло с замера чугуна до начала продувки (секунды)
    """
    stats_df = produv_df.groupby(['NPLV'], as_index=False)['SEC'].min().merge(
        chugun_df[['NPLV', 'DATA_ZAMERA']], on='NPLV', how='left', validate='1:1')
    stats_df['CHUGUN_SINCE_ZAME_SECONDS'] = np.abs(stats_df['SEC'] - stats_df['DATA_ZAMERA']).dt.total_seconds()

    return df.merge(stats_df[['NPLV', 'CHUGUN_SINCE_ZAME_SECONDS']],
                    on=['NPLV'], how='left', validate='1:1')


def _feature_stats_pov(df, produv_df):
    """
    Статистики по полю POL: sum, median, mean, min, max, std
    """
    return df.merge(produv_df.groupby(['NPLV'], as_index=False).agg(NPLV_SUM_POL=('POL', 'sum'),
                                                                    NPLV_MEDIAN_POL=('POL', 'median'),
                                                                    NPLV_MEAN_POL=('POL', 'mean'),
                                                                    NPLV_MIN_POL=('POL', 'min'),
                                                                    NPLV_MAX_POL=('POL', 'max'),
                                                                    NPLV_STD_POL=('POL', 'std')),
                    on=['NPLV'], how='left', validate='1:1')


def _feature_first_last_ras(df, produv_df):
    """
    Первый / последний расход кислорода за плавку
    """

    df = df.merge(produv_df.groupby(['NPLV'], as_index=False).head(1)[['NPLV', 'RAS']] \
                  .rename(columns={'RAS': 'NPLV_FIRST_RAS'}),
                  on=['NPLV'], how='left', validate='1:1')
    df = df.merge(produv_df.groupby(['NPLV'], as_index=False).tail(1)[['NPLV', 'RAS']] \
                  .rename(columns={'RAS': 'NPLV_LAST_RAS'}),
                  on=['NPLV'], how='left', validate='1:1')
    return df


def _feature_produv_seconds(df, produv_df):
    """
    В сумме секунд продувки
    """

    return df.merge(
        (produv_df.groupby(['NPLV']).SEC.max() - produv_df.groupby(['NPLV']).SEC.min()).dt.total_seconds() \
            .reset_index().rename(columns={'SEC': 'NPLV_total_seconds'}),
        on=['NPLV'], how='left', validate='1:1')


def _feature_produv_time(df, produv_df):
    """
    Временные фичи - день недели начала продувки, час начала продувки
    """

    return df.merge(
        produv_df.assign(PRODUV_DAYOFWEEK=produv_df.SEC.dt.dayofweek) \
            .groupby(['NPLV']).head(1)[['NPLV', 'PRODUV_DAYOFWEEK']],
        on=['NPLV'], how='left', validate='1:1').merge(
        produv_df.assign(PRODUV_START_HOUR=produv_df.SEC.dt.hour) \
            .groupby(['NPLV']).head(1)[['NPLV', 'PRODUV_START_HOUR']],
        on=['NPLV'], how='left', validate='1:1')


def _feature_produv_time_since_last_seconds(df, produv_df):
    """
    Сколько секунд прошло с прошлой продувки
    """

    return df.merge(
        (produv_df.groupby(['NPLV']).SEC.min() - produv_df.groupby(['NPLV']).SEC.min().shift()) \
            .dt.total_seconds().reset_index().rename(columns={'SEC': 'NPLV_TIME_SINCE_LAST'}),
        on=['NPLV'], how='left', validate='1:1')


def _feature_stats_lom(df, lom_df):
    """
    Какой лом с каким весом добавлялся в смесь + общее кол-во лома
    """

    lom_df_pivot = lom_df.pivot_table(index='NPLV', columns=['VDL'], values='VES').reset_index()
    lom_df_pivot.columns.name = None
    lom_df_pivot['VDL_SUM'] = lom_df_pivot.loc[:, 'VDL_':].sum(axis=1)

    for lom_col in ['VDL_4', 'VDL_8', 'VDL_13', 'VDL_23', 'VDL_61', 'VDL_20', 'VDL_48', 'VDL_49', 'VDL_63', 'VDL_3']:
        if lom_col not in lom_df_pivot.columns:
            lom_df_pivot[lom_col] = None
            lom_df_pivot[lom_col + '_REL'] = None
        else:
            lom_df_pivot[lom_col + '_REL'] = lom_df_pivot[lom_col]  /  lom_df_pivot['VDL_SUM']

    return df.merge(lom_df_pivot.fillna(0), on=['NPLV'], how='left', validate='1:1')


def _feature_last_temp_gas(df, gas_df):
    """
    Фичи газа - последние показатели, средние показатели, суммарные показатели (до конца продувки)
    """
    return df.merge(
        gas_df.groupby('NPLV').tail(1)[['NPLV', 'T', 'T_rel', 'V', 'O2', 'N2', 'H2', 'CO2', 'CO', 'AR']] \
            .set_index('NPLV').add_suffix('_last_gas').reset_index(),
        on=['NPLV'], how='left', validate='1:1').merge(
        gas_df.groupby('NPLV')[
            ['V', 'T', 'T_rel', 'O2', 'N2', 'H2', 'CO2', 'CO', 'AR', 'T фурмы 1', 'T фурмы 2', 'O2_pressure']].mean() \
            .add_suffix('_mean_gas').reset_index(),
        on=['NPLV'], how='left', validate='1:1').merge(
        gas_df.groupby('NPLV')[['V', ]].sum() \
            .add_suffix('_sum_gas').reset_index(),
        on=['NPLV'], how='left', validate='1:1').merge(
        gas_df[gas_df['Time'] >= gas_df['VR_NACH']]\
            .groupby(['NPLV'])[['O2_V', 'N2_V', 'H2_V', 'CO2_V', 'CO_V', 'AR_V']].sum()\
            .add_suffix('_SUM_PRODUVKA').reset_index(),
        on=['NPLV'], how='left', validate='1:1')


def _feature_gas_temp_additional(df, gas_df, chugun_df, chronom_df, produv_df):
    """
    Относительная температура газа к температуре чугуна во время замера 
    Как изменилась температура газа с момента замера чугуна
    """
    
    # Температура чугуна к газу во время замера чугуна
    chugun_df_with_gas = chugun_df.assign(DATA_ZAMERA_MIN = chugun_df.DATA_ZAMERA.dt.round('min')).merge(
        gas_df.assign(DATA_ZAMERA_MIN = gas_df.Time.dt.round('min'))\
            [['NPLV', 'DATA_ZAMERA_MIN', 'T']].rename(columns = {'T':'T_GAS_ZAMER'}),
        on = ['NPLV', 'DATA_ZAMERA_MIN'], how = 'left'
    ).groupby(['NPLV', 'DATA_ZAMERA_MIN', 'T'], as_index = False)[['T_GAS_ZAMER']].mean()
    chugun_df_with_gas['T_CHUGUN_REL_GAS'] = chugun_df_with_gas['T'] / chugun_df_with_gas['T_GAS_ZAMER']
    df = df.merge(chugun_df_with_gas[['NPLV', 'T_GAS_ZAMER', 'T_CHUGUN_REL_GAS']], on=['NPLV'], how='left', validate='1:1')
    
    # Как изменилась температура газа с момента замера чугуна
    nplv_start_with_temp_gas_df = chronom_df.assign(VR_NACH_MIN = chronom_df.VR_NACH.dt.round('min'))\
            [chronom_df.NOP == 'Продувка'].merge(
        gas_df.assign(VR_NACH_MIN = gas_df.Time.dt.round('min'))\
            [['NPLV', 'VR_NACH_MIN', 'T']].rename(columns = {'T':'T_GAS_ZAMER_NACH_PRODUV'}),
        on = ['NPLV', 'VR_NACH_MIN'], how = 'left'
    ).groupby(['NPLV', 'VR_NACH_MIN'], as_index = False)[['T_GAS_ZAMER_NACH_PRODUV']].mean()
    nplv_start_with_temp_gas_df= nplv_start_with_temp_gas_df.merge(
        chugun_df_with_gas[['NPLV', 'T_GAS_ZAMER']], on = ['NPLV'], how = 'left', validate = '1:1'
    )
    nplv_start_with_temp_gas_df['T_GAS_DELTA_SINCE_ZAMER'] = nplv_start_with_temp_gas_df['T_GAS_ZAMER'] \
        - nplv_start_with_temp_gas_df['T_GAS_ZAMER_NACH_PRODUV']
    df = df.merge(nplv_start_with_temp_gas_df[['NPLV', 'T_GAS_DELTA_SINCE_ZAMER']], on=['NPLV'], how='left', validate='1:1')

    return df


def _feature_stats_sip(df, sip_df):
    """
    Общие фичи по добавлению материалов в смесь: сколько добавили конкретного материала
    """

    sip_df_pivot = sip_df.groupby(['NPLV', 'VDSYP'], as_index=False)['VSSYP'].sum() \
        .pivot_table(index='NPLV', columns=['VDSYP'], values='VSSYP').reset_index()
    sip_df_pivot.columns.name = None
    sip_df_pivot['VDSYP_SUM'] = sip_df_pivot.loc[:, 'VDSYP_':].sum(axis=1)

    # add if missing column
    for col in ['VDSYP_104', 'VDSYP_11', 'VDSYP_119', 'VDSYP_171', 'VDSYP_344', 'VDSYP_346', 'VDSYP_397', 'VDSYP_408',
                'VDSYP_442']:
        if col not in sip_df_pivot.columns:
            sip_df_pivot[col] = None
            sip_df_pivot[col + '_REL'] = None
        else:
            sip_df_pivot[col + '_REL'] = sip_df_pivot[col]  /  sip_df_pivot['VDSYP_SUM']

    return df.merge(sip_df_pivot.fillna(0), on=['NPLV'], how='left', validate='1:1')


def _feature_stats_plavki(df, plavki_df):
    """
    Фичи плавки - направление развилки, типы фурмы
    """
    plavki_df_local = plavki_df.copy()

    # Направление разливки
    def get_napr_zad_code(x):
        if x == 'МНЛС':
            return 1
        if x == 'МНЛЗ':
            return 2
        if x == 'Изл':
            return 3

    plavki_df_local['plavka_NAPR_ZAD'] = plavki_df_local['plavka_NAPR_ZAD'].map(lambda x: get_napr_zad_code(x))

    # фурма тип
    plavki_df_local['plavka_TIPE_FUR'] = plavki_df_local['plavka_TIPE_FUR'].map(
        lambda x: 1 if x == 'цилиндрическая' else 2)

    # фурма тип
    def get_tipe_gol(x):
        if '5 сопловая' in x:
            return 1
        if '4-сопл х54' in x:
            return 2
        return 0

    plavki_df_local['plavka_TIPE_GOL'] = plavki_df_local['plavka_TIPE_GOL'].map(lambda x: get_tipe_gol(x))

    # Запоминаем
    keep_col = ['NPLV', 'plavka_NAPR_ZAD', 'plavka_STFUT', 'plavka_TIPE_FUR', 'plavka_ST_FURM', 'plavka_TIPE_GOL',
                'plavka_ST_GOL', 'plavka_NMZ']
    return df.merge(plavki_df_local[keep_col], on=['NPLV'], how='left', validate='1:1')


# ре-индексируем и пересчитаем по-секудно
def min_to_sec(df):
    df_vals = []
    for nplv, group in df.groupby(['NPLV']):
        group = group.copy()
        group.set_index('SEC', inplace=True)
        group = group.resample('1S').asfreq().reset_index()
        group = group[df.columns].copy()
        df_vals.extend(group.interpolate().values)
    return pd.DataFrame(df_vals, columns=df.columns)

def prepare_features(df, produv_df, lom_df, plavki_df, sip_df, chugun_df, gas_df, chronom_df, used_features):
    """
    Общая функция для сбора фичей для датасета
    """

    # basic produv stats
    df = _feature_stats_ras(df, produv_df)
    df = _feature_stats_pov(df, produv_df)
    df = _feature_produv_seconds(df, produv_df)

    # basic gas stats
    df = _feature_last_temp_gas(df, gas_df)

    # basic lom stats
    df = _feature_stats_lom(df, lom_df)

    # basic plavki stats
    df = _feature_stats_plavki(df, plavki_df)

    # basic sips info
    df = _feature_stats_sip(df, sip_df)

    # basic chugun info
    df = _feature_stats_chugun(df, chugun_df)

    # basic chronom stats
    df = _feature_stats_chronom(df, chronom_df)

    # additional features
    if 'NPLV_FIRST_RAS' in used_features:
        df = _feature_first_last_ras(df, produv_df)
    if 'NPLV_TIME_SINCE_LAST' in used_features:
        df = _feature_produv_time_since_last_seconds(df, produv_df)
    if 'NPLV_SUM_RAS_VIA_MEDIAN' in used_features:
        df['NPLV_SUM_RAS_VIA_MEDIAN'] = df['NPLV_total_seconds'] * df['NPLV_MEDIAN_RAS']
    if 'CHUGUN_SINCE_ZAME_SECONDS' in used_features:
        df = _feature_time_since_chugun(df, produv_df, chugun_df)
    if 'PRODUV_DAYOFWEEK' in used_features:
        df = _feature_produv_time(df, produv_df)
    if '0_CHRONO_NOP' in used_features:
        df = _feature_operations_chronom(df, chronom_df)
    if 'VES_CHUGUN_REL_LOM' in used_features:
        df['VES_CHUGUN_REL_LOM'] = df['VES'] / df['VDL_SUM']
    if 'VES_CHUGUN_SUM_LOM' in used_features:
        df['VES_CHUGUN_SUM_LOM'] = df['VES'] + df['VDL_SUM']
    if 'VES_CHUGUN_REL_SIP' in used_features:
        df['VES_CHUGUN_REL_SIP'] = df['VES'] / df['VDSYP_SUM']
    if 'VES_CHUGUN_SUM_SIP' in used_features:
        df['VES_CHUGUN_SUM_SIP'] = df['VES'] + df['VDSYP_SUM']
    if 'VES_CHUGUN_SUM_SIP_LOM' in used_features:
        df['VES_CHUGUN_SUM_SIP_LOM'] = df['VES'] + df['VDSYP_SUM'] + df['VDL_SUM']
    if 'T_CHUGUN_REL_GAS' in used_features:
        df = _feature_gas_temp_additional(df, gas_df, chugun_df, chronom_df, produv_df)
    if 'MAIN_FURMA' in used_features:
        df['MAIN_FURMA'] = (df['T фурмы 1_mean_gas'] > df['T фурмы 2_mean_gas']).astype('int')
        same_furma_indx = df[df['T фурмы 1_mean_gas'] == df['T фурмы 2_mean_gas']].index
        df.loc[same_furma_indx, 'MAIN_FURMA'] = -1
        

        
    return df
