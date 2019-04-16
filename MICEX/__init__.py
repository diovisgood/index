import os
_Base_Dir = os.path.basename(os.path.dirname(__file__)) + '/'

Source = [ ('moex/' + x) for x in [
    'LKOH', 'SBER', 'GAZP', 'GMKN', 'NVTK', 'ROSN', 'TATN', 'YNDX', 'MGNT', 'ALRS', 'SNGS',
    'MTSS', 'SNGSP', 'FIVE', 'CHMF', 'NLMK', 'POLY', 'IRAO', 'VTBR', 'MOEX', 'PLZL', 'SBERP',
    'USDRUB_TOD', 'Brent',
    'MICEX'
]]

Default_Timezone = 'Europe/Moscow'

Work_Date = ['2018-12-01', '2019-03-31']
Work_Time = ['10:01', '18:40']

Dataset_File = _Base_Dir + 'dataset.csv.gz'

Input_Size = len(Source) - 1
