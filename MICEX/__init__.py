import os
_Base_Dir = os.path.basename(os.path.dirname(__file__)) + '/'

Source = [ (_Base_Dir + 'data/' + x) for x in [
    'LKOH', 'SBER', 'GAZP', 'GMKN', 'NVTK', 'ROSN', 'TATN', 'YNDX', 'MGNT', 'ALRS', 'SNGS',
    'MTSS', 'SNGSP', 'FIVE', 'CHMF', 'NLMK', 'POLY', 'IRAO', 'VTBR', 'MOEX', 'PLZL', 'SBERP',
    'USDRUB_TOD', 'Brent',
    'MICEX'
]]

Default_Timezone = 'Europe/Moscow'

Work_Date = ['2018-12-01', '2019-03-31']
Work_Time = ['10:01', '18:40']

Database_File = _Base_Dir + 'database.csv.gz'

Input_Size = len(Source) - 1
Target_Offset = 2
Output_Size = 1