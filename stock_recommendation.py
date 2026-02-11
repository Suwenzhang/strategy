"""
A股股票推荐策略
基于Tushare数据，结合技术分析和基本面筛选
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockRecommendationStrategy:
    def __init__(self, token):
        """初始化策略"""
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.today = datetime.now().strftime('%Y%m%d')

    def get_stock_list(self):
        """获取A股股票列表"""
        # 获取股票基本信息
        stock_list = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        # 过滤掉ST股票
        stock_list = stock_list[~stock_list['name'].str.contains('ST|退')]
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
        df['volume_ratio'] = df['vol'] / df['volume_ma5']

        # 最新数据
        latest = df.iloc[-1]
        return df, latest

    def technical_score(self, latest):
        """技术面评分 (0-100)"""
        score = 0
        reasons = []

        # 1. 均线多头排列 (20分)
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 20
            reasons.append("均线多头排列")
        elif latest['ma5'] > latest['ma20']:
            score += 10
            reasons.append("短期均线向上")

        # 2. 价格在20日均线上方 (10分)
        if latest['close'] > latest['ma20']:
            score += 10
            reasons.append("价格站上20日线")

        # 3. RSI超卖反弹 (15分)
        if 30 <= latest['rsi'] <= 45:
            score += 15
            reasons.append("RSI处于超卖反弹区")
        elif 45 < latest['rsi'] < 70:
            score += 10
            reasons.append("RSI处于合理区间")

        # 4. MACD金叉 (20分)
        if latest['macd'] > latest['signal'] and latest['hist'] > 0:
            score += 20
            reasons.append("MACD金叉")

        # 5. 成交量放大 (15分)
        if latest['volume_ratio'] > 1.5:
            score += 15
            reasons.append("成交量显著放大")
        elif latest['volume_ratio'] > 1.2:
            score += 8
            reasons.append("成交量温和放大")

        # 6. 布林带位置 (10分)
        if latest['close'] > latest['bb_mid'] and latest['close'] < latest['bb_upper']:
            score += 10
            reasons.append("价格位于布林带中轨上方")
        elif latest['close'] < latest['bb_lower']:
            score += 5
            reasons.append("价格触及布林带下轨")

        # 7. 近期涨幅 (10分)
        if latest['pct_chg'] > 0 and latest['pct_chg'] < 7:
            score += 10
            reasons.append("温和上涨")
        elif latest['pct_chg'] < 0:
            score += 5
            reasons.append("回调买入机会")

        return score, reasons

    def get_financial_data(self, ts_code):
        """获取基本面数据"""
        try:
            # 获取最新财务指标
            df = self.pro.fina_indicator(ts_code=ts_code, period='20241231')
            if df.empty:
                df = self.pro.fina_indicator(ts_code=ts_code)
            if df.empty:
                return None
            return df.iloc[0]
        except:
            return None

    def fundamental_score(self, financial_data):
        """基本面评分 (0-100)"""
        if financial_data is None:
            return 0, ["无财务数据"]

        score = 0
        reasons = []

        # 1. ROE (25分)
        if financial_data.get('roe', 0) > 20:
            score += 25
            reasons.append(f"ROE优秀: {financial_data.get('roe', 0):.2f}%")
        elif financial_data.get('roe', 0) > 15:
            score += 18
            reasons.append(f"ROE良好: {financial_data.get('roe', 0):.2f}%")
        elif financial_data.get('roe', 0) > 10:
            score += 12
            reasons.append(f"ROE尚可: {financial_data.get('roe', 0):.2f}%")

        # 2. 毛利率 (20分)
        if financial_data.get('gross_profit_margin', 0) > 40:
            score += 20
            reasons.append(f"毛利率高: {financial_data.get('gross_profit_margin', 0):.2f}%")
        elif financial_data.get('gross_profit_margin', 0) > 25:
            score += 15
            reasons.append(f"毛利率较好: {financial_data.get('gross_profit_margin', 0):.2f}%")

        # 3. 营收增长 (20分)
        if financial_data.get('or_yoy', 0) > 30:
            score += 20
            reasons.append(f"营收高增长: {financial_data.get('or_yoy', 0):.2f}%")
        elif financial_data.get('or_yoy', 0) > 15:
            score += 15
            reasons.append(f"营收增长良好: {financial_data.get('or_yoy', 0):.2f}%")
        elif financial_data.get('or_yoy', 0) > 0:
            score += 10
            reasons.append(f"营收正增长: {financial_data.get('or_yoy', 0):.2f}%")

        # 4. 净利润增长 (20分)
        if financial_data.get('profit_yoy', 0) > 30:
            score += 20
            reasons.append(f"净利润高增长: {financial_data.get('profit_yoy', 0):.2f}%")
        elif financial_data.get('profit_yoy', 0) > 15:
            score += 15
            reasons.append(f"净利润增长良好: {financial_data.get('profit_yoy', 0):.2f}%")
        elif financial_data.get('profit_yoy', 0) > 0:
            score += 10
            reasons.append(f"净利润正增长: {financial_data.get('profit_yoy', 0):.2f}%")

        # 5. 负债率 (15分)
        debt_ratio = financial_data.get('debt_to_assets', 0)
        if 0 < debt_ratio < 40:
            score += 15
            reasons.append(f"负债率低: {debt_ratio:.2f}%")
        elif 40 <= debt_ratio < 60:
            score += 10
            reasons.append(f"负债率合理: {debt_ratio:.2f}%")
        elif debt_ratio < 80:
            score += 5
            reasons.append(f"负债率偏高: {debt_ratio:.2f}%")

        return score, reasons

    def get_realtime_quote(self, ts_code):
        """获取实时行情"""
        try:
            df = self.pro.daily(ts_code=ts_code, limit='1')
            if df.empty:
                return None
            return df.iloc[0]
        except:
            return None

    def analyze_stock(self, stock_info):
        """综合分析单只股票"""
        ts_code = stock_info['ts_code']
        stock_name = stock_info['name']
        industry = stock_info.get('industry', '')

        # 获取日线数据
        daily_data = self.get_daily_data(ts_code)
        if daily_data is None or len(daily_data) < 60:
            return None

        # 计算技术指标
        result = self.calculate_indicators(daily_data)
        if result is None:
            return None
        df, latest = result

        # 技术面评分
        tech_score, tech_reasons = self.technical_score(latest)

        # 基本面评分
        financial_data = self.get_financial_data(ts_code)
        fund_score, fund_reasons = self.fundamental_score(financial_data)

        # 综合评分 (技术面60% + 基本面40%)
        total_score = tech_score * 0.6 + fund_score * 0.4

        # 判断推荐等级
        if total_score >= 75:
            level = "强烈推荐"
            emoji = "★★★★★"
        elif total_score >= 60:
            level = "推荐"
            emoji = "★★★★"
        elif total_score >= 45:
            level = "谨慎关注"
            emoji = "★★★"
        elif total_score >= 30:
            level = "观望"
            emoji = "★★"
        else:
            level = "不推荐"
            emoji = "★"

        return {
            'ts_code': ts_code,
            'name': stock_name,
            'industry': industry,
            'price': latest['close'],
            'pct_chg': latest['pct_chg'],
            'tech_score': round(tech_score, 1),
            'fund_score': round(fund_score, 1),
            'total_score': round(total_score, 1),
            'level': level,
            'emoji': emoji,
            'tech_reasons': tech_reasons,
            'fund_reasons': fund_reasons,
            'ma5': round(latest['ma5'], 2),
            'ma20': round(latest['ma20'], 2),
            'rsi': round(latest['rsi'], 1) if pd.notna(latest['rsi']) else None,
            'macd': round(latest['macd'], 4) if pd.notna(latest['macd']) else None,
            'volume_ratio': round(latest['volume_ratio'], 2) if pd.notna(latest['volume_ratio']) else None
        }

    def scan_market(self, stock_list=None, top_n=20):
        """扫描市场并推荐股票"""
        print(f"\n{'='*60}")
        print(f"A股股票推荐策略 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        if stock_list is None:
            print("正在获取股票列表...")
            stock_list = self.get_stock_list()
            print(f"共获取 {len(stock_list)} 只股票\n")

        results = []
        total = len(stock_list)

        for idx, stock in stock_list.iterrows():
            print(f"\r分析进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) - {stock['name']}", end='')

            try:
                result = self.analyze_stock(stock)
                if result and result['total_score'] > 40:
                    results.append(result)
            except Exception as e:
                continue

        print("\n")

        # 按综合评分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)

        # 输出推荐结果
        print(f"\n{'='*60}")
        print(f"TOP {top_n} 推荐股票")
        print(f"{'='*60}\n")

        for i, stock in enumerate(results[:top_n], 1):
            print(f"{i}. {stock['emoji']} {stock['name']} ({stock['ts_code']})")
            print(f"   行业: {stock['industry']}")
            print(f"   现价: {stock['price']:.2f}  涨跌幅: {stock['pct_chg']:+.2f}%")
            print(f"   综合评分: {stock['total_score']:.1f} | 技术面: {stock['tech_score']:.1f} | 基本面: {stock['fund_score']:.1f}")
            print(f"   MA5: {stock['ma5']:.2f} | MA20: {stock['ma20']:.2f}")
            if stock['rsi']:
                print(f"   RSI: {stock['rsi']:.1f}")
            if stock['volume_ratio']:
                print(f"   量比: {stock['volume_ratio']:.2f}")
            print(f"   推荐等级: {stock['level']}")
            print(f"   技术面理由: {', '.join(stock['tech_reasons'])}")
            if stock['fund_reasons']:
                print(f"   基本面理由: {', '.join(stock['fund_reasons'])}")
            print()

        # 统计信息
        strong_buy = len([r for r in results if r['level'] == '强烈推荐'])
        buy = len([r for r in results if r['level'] == '推荐'])
        hold = len([r for r in results if r['level'] == '谨慎关注'])

        print(f"{'='*60}")
        print(f"统计信息:")
        print(f"  强烈推荐: {strong_buy} 只")
        print(f"  推荐: {buy} 只")
        print(f"  谨慎关注: {hold} 只")
        print(f"  总计符合条件: {len(results)} 只")
        print(f"{'='*60}\n")

        # 保存结果到CSV
        df_results = pd.DataFrame(results)
        csv_file = f"stock_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {csv_file}\n")

        return results


def main():
    """主函数"""
    # Tushare API Token
    TOKEN = '2d3ab38a292d8548cf2bb8b1eaccc06268c3945c3eb556d647e8ccdf'

    # 创建策略实例
    strategy = StockRecommendationStrategy(TOKEN)

    # 选项1: 分析全市场（需要较长时间）
    results = strategy.scan_market(top_n=15)

    # 选项2: 分析指定股票列表
    # target_stocks = ['000001.SZ', '600000.SH', '600519.SH', '300750.SZ']
    # stock_list = strategy.pro.stock_basic(ts_code=target_stocks,
    #                                       fields='ts_code,symbol,name,area,industry,list_date')
    # results = strategy.scan_market(stock_list=stock_list, top_n=10)


if __name__ == '__main__':
    main()
