"""调试脚本 - 检查Tushare返回的股票数据"""

import tushare as ts
import pandas as pd

TOKEN = '2d3ab38a292d8548cf2bb8b1eaccc06268c3945c3eb556d647e8ccdf'
ts.set_token(TOKEN)
pro = ts.pro_api()

print("正在获取股票基本信息...")
stock_list = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,symbol,name,area,industry,list_date,market'
)

print(f"\n总共获取到: {len(stock_list)} 只股票")
print(f"\n数据列: {stock_list.columns.tolist()}")
print(f"\n前10条数据:")
print(stock_list.head(10))

print(f"\n\n检查ts_code格式:")
print(stock_list['ts_code'].value_counts().head(20))

print(f"\n\n按ts_code前3位分组:")
stock_list['prefix'] = stock_list['ts_code'].str[:3]
prefix_counts = stock_list.groupby('prefix').size().sort_values(ascending=False)
print(prefix_counts.head(20))

print(f"\n\n检查market字段:")
print(stock_list['market'].value_counts())

print(f"\n\n尝试筛选主板股票:")
# 方法1: 按market字段筛选
if 'market' in stock_list.columns:
    mainboard = stock_list[stock_list['market'].isin(['主板', 'MainBoard', '主板A'])]
    print(f"按market筛选主板: {len(mainboard)} 只")

# 方法2: 按代码前缀筛选
sh_mainboard = stock_list[stock_list['ts_code'].str.startswith('60')]
sz_mainboard = stock_list[stock_list['ts_code'].str.startswith(('000', '001'))]
print(f"上交所主板(60): {len(sh_mainboard)} 只")
print(f"深交所主板(000/001): {len(sz_mainboard)} 只")
print(f"合计主板: {len(sh_mainboard) + len(sz_mainboard)} 只")

print(f"\n\n上交所主板样本:")
print(sh_mainboard.head(10))

print(f"\n\n深交所主板样本:")
print(sz_mainboard.head(10))
