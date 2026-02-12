"""
A股沪深主板股票推荐策略
专注于10cm涨跌幅的主板股票，区分开盘和收盘推荐策略
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============== 基础类 ==============
class StockStrategyBase:
    """策略基类"""

    def __init__(self, token):
        """初始化策略"""
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.today = datetime.now().strftime('%Y%m%d')

    def get_mainboard_stocks(self):
        """获取沪深主板股票列表 (10cm涨跌幅)"""
        # 上海主板: 600xxx, 601xxx, 603xxx, 605xxx
        # 深圳主板: 000xxx, 001xxx
        stock_list = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date,market'
        )

        # 过滤主板股票
        stock_list = stock_list[
            (stock_list['ts_code'].str.match(r'^60\d{4}$')) |  # 上交所主板
            (stock_list['ts_code'].str.match(r'^00[01]\d{4}$'))  # 深交所主板
        ]

        # 过滤掉ST、退市股票
        stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停')]

        return stock_list

    def get_daily_data(self, ts_code, days=120):
        """获取日线数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return None
        df = df.sort_values('trade_date')
        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        if df is None or len(df) < 20:
            return None

        df = df.copy()

        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()

        # RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD指标
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']

        # 布林带
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        # 成交量变化
        df['volume_ma5'] = df['vol'].rolling(window=5).mean()
        df['volume_ma10'] = df['vol'].rolling(window=10).mean()
        df['volume_ratio'] = df['vol'] / df['volume_ma5']

        # KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        # 涨停板判断 (10%)
        df['limit_up'] = df['pct_chg'] >= 9.5
        df['limit_down'] = df['pct_chg'] <= -9.5

        # 近期连续涨停天数
        df['consecutive_up'] = 0
        for i in range(len(df)):
            if df.iloc[i]['limit_up']:
                if i > 0 and df.iloc[i-1]['consecutive_up'] > 0:
                    df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
                else:
                    df.iloc[i, df.columns.get_loc('consecutive_up')] = 1

        return df


# ============== 开盘前策略 (9:25前) ==============
class OpeningStrategy(StockStrategyBase):
    """开盘前推荐策略 (9:25前)"""

    def __init__(self, token):
        super().__init__(token)

    def opening_score(self, df):
        """开盘前评分 (0-100)"""
        if df is None or len(df) < 20:
            return 0, []

        score = 0
        reasons = []
        latest = df.iloc[-1]
        yesterday = df.iloc[-2] if len(df) >= 2 else None

        # 1. 昨日涨停连板 (25分)
        if latest['consecutive_up'] >= 3:
            score += 25
            reasons.append(f"{int(latest['consecutive_up'])}连板强势")
        elif latest['consecutive_up'] >= 2:
            score += 20
            reasons.append("2连板")
        elif latest['limit_up']:
            score += 15
            reasons.append("昨日涨停")

        # 2. 跳空高开形态 (20分) - 需要实时数据，此处用昨日收盘价判断
        if yesterday is not None and latest['close'] > latest['open']:
            gap_ratio = (latest['open'] - yesterday['close']) / yesterday['close'] * 100
            if gap_ratio > 3:
                score += 20
                reasons.append("跳空高开潜力")
            elif gap_ratio > 1:
                score += 12
                reasons.append("小幅跳空")

        # 3. 均线多头排列 (15分)
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 15
            reasons.append("均线多头排列")
        elif latest['ma5'] > latest['ma20']:
            score += 8
            reasons.append("短期趋势向上")

        # 4. 成交量异常放大 (15分)
        if latest['volume_ratio'] > 2:
            score += 15
            reasons.append("昨日巨量换手")
        elif latest['volume_ratio'] > 1.5:
            score += 10
            reasons.append("昨日放量")

        # 5. RSI金叉信号 (10分)
        if yesterday is not None and 30 < latest['rsi'] < 70:
            if yesterday['rsi'] < latest['rsi']:
                score += 10
                reasons.append("RSI向上")
            else:
                score += 5
                reasons.append("RSI合理区间")

        # 6. KDJ低位金叉 (15分)
        if yesterday is not None:
            if (latest['k'] > latest['d'] and yesterday['k'] <= yesterday['d'] and
                latest['k'] < 40):
                score += 15
                reasons.append("KDJ低位金叉")
            elif latest['j'] > 100:
                score += 8
                reasons.append("KDJ超强势")

        return score, reasons

    def get_limit_up_stocks(self, date=None):
        """获取涨停股票列表"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        df = self.pro.limit_list_d(trade_date=date, limit_type='U')
        if df.empty:
            # 如果没有当天数据，获取最近一个交易日
            df = self.pro.limit_list_d(limit_type='U')
            if not df.empty:
                df = df[df['trade_date'] == df['trade_date'].max()]
        return df

    def analyze_stock_opening(self, stock_info):
        """开盘前分析单只股票"""
        ts_code = stock_info['ts_code']
        stock_name = stock_info['name']
        industry = stock_info.get('industry', '')

        # 获取日线数据
        daily_data = self.get_daily_data(ts_code)
        if daily_data is None or len(daily_data) < 30:
            return None

        # 计算技术指标
        df = self.calculate_indicators(daily_data)
        if df is None:
            return None

        latest = df.iloc[-1]

        # 开盘前评分
        score, reasons = self.opening_score(df)

        # 判断推荐等级
        if score >= 75:
            level = "强烈关注"
            emoji = "★★★★★"
        elif score >= 60:
            level = "关注"
            emoji = "★★★★"
        elif score >= 45:
            level = "谨慎观察"
            emoji = "★★★"
        elif score >= 30:
            level = "观望"
            emoji = "★★"
        else:
            level = "不推荐"
            emoji = "★"

        return {
            'ts_code': ts_code,
            'name': stock_name,
            'industry': industry,
            'close_price': latest['close'],
            'pct_chg': latest['pct_chg'],
            'score': round(score, 1),
            'level': level,
            'emoji': emoji,
            'reasons': reasons,
            'consecutive_up': int(latest['consecutive_up']),
            'volume_ratio': round(latest['volume_ratio'], 2) if pd.notna(latest['volume_ratio']) else None,
            'ma5': round(latest['ma5'], 2),
            'ma20': round(latest['ma20'], 2),
            'rsi': round(latest['rsi'], 1) if pd.notna(latest['rsi']) else None,
            'k': round(latest['k'], 1) if pd.notna(latest['k']) else None,
            'd': round(latest['d'], 1) if pd.notna(latest['d']) else None,
        }

    def scan_opening(self, top_n=20):
        """开盘前扫描"""
        print(f"\n{'='*70}")
        print(f"【开盘前推荐策略】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # 获取主板股票列表
        print("正在获取沪深主板股票列表...")
        stock_list = self.get_mainboard_stocks()
        print(f"共获取 {len(stock_list)} 只沪深主板股票\n")

        # 优先获取涨停股票
        print("正在获取昨日涨停股票...")
        limit_up_stocks = self.get_limit_up_stocks()
        limit_codes = limit_up_stocks['ts_code'].tolist() if not limit_up_stocks.empty else []
        print(f"昨日涨停: {len(limit_codes)} 只\n")

        results = []
        total = len(stock_list)

        # 先分析涨停股票
        if limit_codes:
            limit_stocks = stock_list[stock_list['ts_code'].isin(limit_codes)]
            for idx, stock in limit_stocks.iterrows():
                print(f"\r分析涨停股票: {idx+1}/{len(limit_stocks)} - {stock['name']}", end='')
                try:
                    result = self.analyze_stock_opening(stock)
                    if result and result['score'] > 30:
                        results.append(result)
                except Exception as e:
                    continue

        # 再分析其他股票
        for idx, stock in stock_list.iterrows():
            if stock['ts_code'] in limit_codes:
                continue
            print(f"\r分析进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) - {stock['name']}", end='')

            try:
                result = self.analyze_stock_opening(stock)
                # 只保留分数较高的
                if result and result['score'] > 45:
                    results.append(result)
            except Exception as e:
                continue

        print("\n")

        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)

        # 输出结果
        self._print_results(results, top_n, "开盘前")

        return results

    def _print_results(self, results, top_n, strategy_name):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f"TOP {top_n} 【{strategy_name}】推荐股票")
        print(f"{'='*70}\n")

        for i, stock in enumerate(results[:top_n], 1):
            print(f"{i}. {stock['emoji']} {stock['name']} ({stock['ts_code']})")
            print(f"   行业: {stock['industry']}")
            print(f"   昨收: {stock['close_price']:.2f}  涨跌幅: {stock['pct_chg']:+.2f}%")
            print(f"   评分: {stock['score']:.1f}  等级: {stock['level']}")
            if stock['consecutive_up'] > 0:
                print(f"   连板: {stock['consecutive_up']}板")
            if stock['volume_ratio']:
                print(f"   量比: {stock['volume_ratio']:.2f}")
            print(f"   理由: {', '.join(stock['reasons'])}")
            print()

        # 统计
        strong = len([r for r in results if r['level'] == '强烈关注'])
        follow = len([r for r in results if r['level'] == '关注'])
        watch = len([r for r in results if r['level'] == '谨慎观察'])

        print(f"{'='*70}")
        print(f"统计: 强烈关注{strong} | 关注{follow} | 谨慎观察{watch} | 总计{len(results)}")
        print(f"{'='*70}\n")

        # 保存CSV
        df = pd.DataFrame(results)
        csv_file = f"opening_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存: {csv_file}\n")


# ============== 收盘前策略 (14:56前) ==============
class ClosingStrategy(StockStrategyBase):
    """收盘前推荐策略 (14:56前)"""

    def __init__(self, token):
        super().__init__(token)

    def closing_score(self, df):
        """收盘前评分 (0-100)"""
        if df is None or len(df) < 20:
            return 0, []

        score = 0
        reasons = []
        latest = df.iloc[-1]

        # 1. 尾盘拉升形态 (25分) - 收盘价高于开盘价且接近最高价
        if latest['close'] > latest['open']:
            gain = (latest['close'] - latest['open']) / latest['open'] * 100
            close_to_high = (latest['high'] - latest['close']) / latest['close'] * 100

            if gain > 5 and close_to_high < 1:
                score += 25
                reasons.append("强势尾盘拉升")
            elif gain > 3 and close_to_high < 2:
                score += 18
                reasons.append("尾盘拉升")
            elif gain > 0:
                score += 10
                reasons.append("收阳线")

        # 2. 技术形态突破 (25分)
        # 突破20日均线
        if latest['close'] > latest['ma20'] and df.iloc[-2]['close'] <= df.iloc[-2]['ma20']:
            score += 15
            reasons.append("突破20日线")

        # 突破布林带上轨
        if latest['close'] > latest['bb_upper']:
            score += 10
            reasons.append("突破布林上轨")

        # 3. 成交量配合 (20分)
        if latest['volume_ratio'] > 2:
            score += 20
            reasons.append("巨量配合")
        elif latest['volume_ratio'] > 1.5:
            score += 15
            reasons.append("放量")
        elif latest['volume_ratio'] > 1.2:
            score += 10
            reasons.append("温和放量")

        # 4. MACD信号 (15分)
        if latest['macd'] > latest['signal'] and latest['hist'] > df.iloc[-2]['hist']:
            score += 15
            reasons.append("MACD加速向上")
        elif latest['macd'] > latest['signal']:
            score += 10
            reasons.append("MACD金叉")

        # 5. RSI合理区间 (10分)
        if 50 < latest['rsi'] < 70:
            score += 10
            reasons.append("RSI强势区")
        elif 40 < latest['rsi'] <= 50:
            score += 6
            reasons.append("RSI上升区")
        elif 70 <= latest['rsi'] < 80:
            score += 5
            reasons.append("RSI接近超买")

        # 6. K线形态 (5分)
        body = abs(latest['close'] - latest['open'])
        upper_shadow = latest['high'] - max(latest['close'], latest['open'])
        lower_shadow = min(latest['close'], latest['open']) - latest['low']

        # 大阳线
        if body > lower_shadow * 2 and upper_shadow < body * 0.3:
            score += 5
            reasons.append("大阳线形态")
        # 突破形态
        elif latest['close'] == latest['high']:
            score += 3
            reasons.append("光头阳线")

        return score, reasons

    def analyze_stock_closing(self, stock_info):
        """收盘前分析单只股票"""
        ts_code = stock_info['ts_code']
        stock_name = stock_info['name']
        industry = stock_info.get('industry', '')

        # 获取日线数据
        daily_data = self.get_daily_data(ts_code)
        if daily_data is None or len(daily_data) < 30:
            return None

        # 计算技术指标
        df = self.calculate_indicators(daily_data)
        if df is None:
            return None

        latest = df.iloc[-1]

        # 收盘前评分
        score, reasons = self.closing_score(df)

        # 判断推荐等级
        if score >= 75:
            level = "强烈推荐"
            emoji = "★★★★★"
        elif score >= 60:
            level = "推荐"
            emoji = "★★★★"
        elif score >= 45:
            level = "谨慎关注"
            emoji = "★★★"
        elif score >= 30:
            level = "观望"
            emoji = "★★"
        else:
            level = "不推荐"
            emoji = "★"

        return {
            'ts_code': ts_code,
            'name': stock_name,
            'industry': industry,
            'close_price': latest['close'],
            'pct_chg': latest['pct_chg'],
            'score': round(score, 1),
            'level': level,
            'emoji': emoji,
            'reasons': reasons,
            'volume_ratio': round(latest['volume_ratio'], 2) if pd.notna(latest['volume_ratio']) else None,
            'ma5': round(latest['ma5'], 2),
            'ma20': round(latest['ma20'], 2),
            'ma60': round(latest['ma60'], 2),
            'rsi': round(latest['rsi'], 1) if pd.notna(latest['rsi']) else None,
            'macd': round(latest['macd'], 4) if pd.notna(latest['macd']) else None,
            'bb_upper': round(latest['bb_upper'], 2),
        }

    def scan_closing(self, top_n=20):
        """收盘前扫描"""
        print(f"\n{'='*70}")
        print(f"【收盘前推荐策略】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # 获取主板股票列表
        print("正在获取沪深主板股票列表...")
        stock_list = self.get_mainboard_stocks()
        print(f"共获取 {len(stock_list)} 只沪深主板股票\n")

        results = []
        total = len(stock_list)

        for idx, stock in stock_list.iterrows():
            print(f"\r分析进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) - {stock['name']}", end='')

            try:
                result = self.analyze_stock_closing(stock)
                if result and result['score'] > 40:
                    results.append(result)
            except Exception as e:
                continue

        print("\n")

        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)

        # 输出结果
        self._print_results(results, top_n, "收盘前")

        return results

    def _print_results(self, results, top_n, strategy_name):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f"TOP {top_n} 【{strategy_name}】推荐股票")
        print(f"{'='*70}\n")

        for i, stock in enumerate(results[:top_n], 1):
            print(f"{i}. {stock['emoji']} {stock['name']} ({stock['ts_code']})")
            print(f"   行业: {stock['industry']}")
            print(f"   收盘价: {stock['close_price']:.2f}  涨跌幅: {stock['pct_chg']:+.2f}%")
            print(f"   综合评分: {stock['score']:.1f}  等级: {stock['level']}")
            print(f"   MA5: {stock['ma5']:.2f} | MA20: {stock['ma20']:.2f} | MA60: {stock['ma60']:.2f}")
            if stock['rsi']:
                print(f"   RSI: {stock['rsi']:.1f}")
            if stock['volume_ratio']:
                print(f"   量比: {stock['volume_ratio']:.2f}")
            print(f"   理由: {', '.join(stock['reasons'])}")
            print()

        # 统计
        strong = len([r for r in results if r['level'] == '强烈推荐'])
        buy = len([r for r in results if r['level'] == '推荐'])
        hold = len([r for r in results if r['level'] == '谨慎关注'])

        print(f"{'='*70}")
        print(f"统计: 强烈推荐{strong} | 推荐{buy} | 谨慎关注{hold} | 总计{len(results)}")
        print(f"{'='*70}\n")

        # 保存CSV
        df = pd.DataFrame(results)
        csv_file = f"closing_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存: {csv_file}\n")


# ============== 主函数 ==============
def main():
    """主函数"""
    TOKEN = '2d3ab38a292d8548cf2bb8b1eaccc06268c3945c3eb556d647e8ccdf'

    # 根据当前时间选择策略
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    print(f"\n当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n请选择推荐策略:")
    print(f"  1. 开盘前推荐 (适合9:25前运行)")
    print(f"  2. 收盘前推荐 (适合14:56前运行)")
    print(f"  3. 同时运行两种策略")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == '1':
        print("\n执行开盘前推荐策略...")
        strategy = OpeningStrategy(TOKEN)
        results = strategy.scan_opening(top_n=20)

    elif choice == '2':
        print("\n执行收盘前推荐策略...")
        strategy = ClosingStrategy(TOKEN)
        results = strategy.scan_closing(top_n=20)

    elif choice == '3':
        print("\n执行开盘前推荐策略...")
        opening = OpeningStrategy(TOKEN)
        opening_results = opening.scan_opening(top_n=15)

        print("\n" + "="*70)
        print("\n执行收盘前推荐策略...")
        closing = ClosingStrategy(TOKEN)
        closing_results = closing.scan_closing(top_n=15)

    else:
        print("\n无效选项!")


if __name__ == '__main__':
    main()
