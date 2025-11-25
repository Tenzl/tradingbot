import time
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import talib as ta
except Exception as e:
    raise RuntimeError("TA-Lib required: install ta-lib before running.")

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
# Webhook URL để gửi tín hiệu (Thay YOUR-FUNCTION-NAME bằng tên thực tế)
import os
# WEBHOOK_URL = os.getenv("WEBHOOK_URL")
# if not WEBHOOK_URL:
#     raise RuntimeError("WEBHOOK_URL is not set in environment variables")

WEBHOOK_URL = "https://trade-ingest-thk.azurewebsites.net/api/webhook/3359389a-7483-4ae9-ac8f-991d647b3f8a"
SECRET_KEY = "vTiBpjPuLARyN6sa0edRem2mrs3wYDPfmxVwqLZyYpc"
# SECRET_KEY = os.getenv("SECRET_KEY")
# if not SECRET_KEY:
#     raise RuntimeError("SECRET_KEY is not set in environment variables")
# Test health check: https://trading-bot-webhook.azurewebsites.net/api/health

# Đường dẫn model
MODEL_PATH = "xgb_realtime_bot_model.json"

# Thiết lập giao dịch
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
LIMIT = 1000  # Binance API giới hạn tối đa 1000 nến/request

# Ngưỡng vào lệnh (Điều chỉnh theo kết quả training)
LONG_THRESHOLD_PROB = 0.53   # Vào LONG khi xác suất TĂNG > 53%
SHORT_THRESHOLD_PROB = 0.47  # Vào SHORT khi xác suất TĂNG < 47% (tức Giảm > 53%)

# Config tính toán chỉ báo (PHẢI GIỐNG VỚI TRAINING)
CFG = dict(
    z_windows=[60, 240, 1440],
    rsi=14, mfi=14, wr=14, cci=14, atr=14, adx=14,
    macd=(12, 26, 9), stoch=(14, 3, 3), sar=(0.02, 0.2),
    sma_periods=[1440],
    ema_periods=[1440],
    roc_periods=[30],
)

# List 49 Features + Close (ĐÚNG THỨ TỰ NHƯ LÚC TRAIN)
SELECTED_FEATURES = [
    'alpha_macdhist_mtf_grad', 'alpha_mfi_mtf_grad', 'Close_to_SMA1440',
    'alpha_obv_mtf_grad', 'MFI_14_z240', 'STOCH_K_z60', 'SAR_z1440',
    'Bollinger_Middle_z1440', 'alpha_rsi_mtf_grad', 'alpha_aggr_mtf_grad',
    'Bollinger_Middle_z60', 'CloseToVWAP_z1440', 'ADX_14_z1440',
    'alpha_tradeint_mtf_grad', 'ADX_14_z240', 'alpha_rsi_mfi_div_z60',
    'alpha_aggr_z60_filt_ema1440', 'CCI_14_z60', 'UpDownVolumeRatio_60_z60',
    'STOCH_D_z240', 'MFI_14_z60', 'STOCH_D_z1440', 'alpha_stoch_wr_div_z60',
    'alpha_cci_mtf_grad', 'alpha_rsi_obv_div_z240', 'SAR_z240',
    'CloseToVWAP_z60', 'alpha_upshadow_x_aggr_z60', 'alpha_flow_z60_filt_ema1440',
    'alpha_rsi_cci_div_z60', 'alpha_rsi_mfi_div_z240', 'ATR_14_z60',
    'MACD_hist_z60', 'Williams_R_14_z240', 'QuotePerTrade_z240',
    'UpDownVolumeRatio_60_z1440', 'SAR_z60', 'alpha_buypress_mtf_grad',
    'alpha_updown_mtf_grad', 'alpha_cci_z60_filt_sma1440', 'alpha_roc30_mtf_grad',
    'BuyPressure_z240', 'alpha_atr_mtf_grad', 'Williams_R_14_z60',
    'Range_to_Close_z1440', 'CloseToVWAP_z240', 'OBV_log_z1440',
    'STOCH_K_z1440', 'MACD_z60', 'Close'  # Model được train với cột Close
]

# ==========================================
# 2. HÀM GỬI WEBHOOK
# ==========================================
def send_signal_to_web(action, symbol, price, confidence):
    """Gửi tín hiệu về Website qua HTTP POST"""
    payload = {
        "secret_key": SECRET_KEY,  # Bảo mật
        "symbol": symbol,
        "action": action,  # "LONG" hoặc "SHORT"
        "price": float(price),  # Convert numpy/pandas float to Python float
        "confidence": float(confidence),  # Convert numpy/pandas float to Python float
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ Webhook sent success: {response.text}")
        else:
            print(f"⚠️ Webhook failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Webhook error: {e}")

# ==========================================
# 3. HÀM TÍNH TOÁN 49 FEATURES
# ==========================================
def zscore(series: pd.Series, w: int) -> pd.Series:
    """Tính Z-Score với cửa sổ rolling"""
    m = series.rolling(w, min_periods=w).mean()
    sd = series.rolling(w, min_periods=w).std(ddof=0)
    return (series - m) / sd.replace(0, np.nan)

def calculate_features(df):
    """
    Tính toán y hệt 49 features như trong file training.
    Input: DataFrame từ Binance Kline
    Output: DataFrame với đầy đủ features
    """
    df = df.copy()
    
    # Đổi tên cột về lowercase nếu cần
    rename_map = {
        'Open time': 'open_time', 'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume', 'Close time': 'close_time',
        'Quote asset volume': 'quote_volume', 'Number of trades': 'num_trades',
        'Taker buy base asset volume': 'taker_buy_base',
        'Taker buy quote asset volume': 'taker_buy_quote'
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    
    # Tạo cột Close (viết hoa) cho model - GIỐNG NHƯ KHI TRAIN
    df['Close'] = df['close']
    
    # ============ BASE INDICATORS ============
    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    vol = df['volume'].astype(float).values
    
    # Core TA-Lib indicators
    df[f'RSI_{CFG["rsi"]}'] = ta.RSI(close, timeperiod=CFG['rsi'])
    df[f'MFI_{CFG["mfi"]}'] = ta.MFI(high, low, close, vol, timeperiod=CFG['mfi'])
    df[f'Williams_R_{CFG["wr"]}'] = ta.WILLR(high, low, close, timeperiod=CFG['wr'])
    df[f'CCI_{CFG["cci"]}'] = ta.CCI(high, low, close, timeperiod=CFG['cci'])
    
    macd, macd_signal, macd_hist = ta.MACD(close, 
                                            fastperiod=CFG['macd'][0], 
                                            slowperiod=CFG['macd'][1], 
                                            signalperiod=CFG['macd'][2])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    
    u, m, l = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    df['Bollinger_Middle'] = m
    
    for p in CFG['sma_periods']:
        df[f'SMA_{p}'] = ta.SMA(close, timeperiod=p)
    for p in CFG['ema_periods']:
        df[f'EMA_{p}'] = ta.EMA(close, timeperiod=p)
    
    df[f'ATR_{CFG["atr"]}'] = ta.ATR(high, low, close, timeperiod=CFG['atr'])
    df[f'ADX_{CFG["adx"]}'] = ta.ADX(high, low, close, timeperiod=CFG['adx'])
    df['OBV'] = ta.OBV(close, vol)
    df['OBV_log'] = np.log1p(df['OBV'].abs()) * np.sign(df['OBV'])
    
    # STOCH
    k_period, d_period, s_period = CFG['stoch']
    slowk, slowd = ta.STOCH(high, low, close, 
                             fastk_period=k_period, 
                             slowk_period=d_period, 
                             slowk_matype=0,
                             slowd_period=s_period, 
                             slowd_matype=0)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    
    # SAR
    af, maxaf = CFG['sar']
    df['SAR'] = ta.SAR(high, low, acceleration=af, maximum=maxaf)
    
    # ROC 30
    df['ROC_30'] = ta.ROCP(close, timeperiod=CFG['roc_periods'][0])
    
    # ============ ENGINEERED FEATURES ============
    # VWAP
    df['VWAP'] = df['quote_volume'] / df['volume']
    
    # Buy/Sell Volume
    df['buy_vol'] = df['taker_buy_base']
    df['sell_vol'] = df['volume'] - df['buy_vol']
    df['buy_quote'] = df['taker_buy_quote']
    df['sell_quote'] = df['quote_volume'] - df['buy_quote']
    
    # Aggressor Imbalance & Buy Pressure
    df['AggressorImbalance'] = 2 * (df['buy_vol'] / df['volume']) - 1
    df['BuyPressure'] = df['buy_quote'] / df['sell_quote']
    
    # Candle Structure
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['UpperShadowRatio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / rng
    df['LowerShadowRatio'] = (df[['open', 'close']].min(axis=1) - df['low']) / rng
    df['BodyRatio'] = (df['close'] - df['open']).abs() / rng
    df['Range_to_Close'] = (df['high'] - df['low']) / df['close']
    
    # VWAP Proximity
    df['CloseToVWAP'] = (df['close'] - df['VWAP']) / df['VWAP']
    
    # Quote per Trade & Trade Intensity
    df['QuotePerTrade'] = df['quote_volume'] / df['num_trades']
    df['TradeIntensity'] = df['num_trades'] / df['volume']
    
    # Return & Flow Momentum
    df['Return'] = df['close'] / df['open'] - 1
    df['buy_vol_roc5'] = df['buy_vol'].pct_change(5)
    df['sell_vol_roc5'] = df['sell_vol'].pct_change(5)
    df['FlowMomentum'] = df['buy_vol_roc5'] - df['sell_vol_roc5']
    
    # Up/Down Volume Ratio
    up = np.where(df['Return'] > 0, df['volume'], 0.0)
    down = np.where(df['Return'] < 0, df['volume'], 0.0)
    df['UpDownVolumeRatio_60'] = (pd.Series(up).rolling(60).sum() / 
                                   pd.Series(down).rolling(60).sum()).values
    
    # Relative distances
    for p in CFG['sma_periods']:
        df[f'Close_to_SMA{p}'] = (df['close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']
    for p in CFG['ema_periods']:
        df[f'Close_to_EMA{p}'] = (df['close'] - df[f'EMA_{p}']) / df[f'EMA_{p}']
    
    # ============ Z-SCORES ============
    Z_BASES = [
        f'RSI_{CFG["rsi"]}', f'MFI_{CFG["mfi"]}', f'Williams_R_{CFG["wr"]}', 
        f'CCI_{CFG["cci"]}', f'ATR_{CFG["atr"]}', f'ADX_{CFG["adx"]}', 
        'MACD_hist', 'MACD', 'OBV_log', 'STOCH_K', 'STOCH_D', 'SAR', 
        'Bollinger_Middle', 'QuotePerTrade', 'BuyPressure', 'UpDownVolumeRatio_60',
        'Range_to_Close', 'CloseToVWAP', 'AggressorImbalance', 'FlowMomentum',
        'BodyRatio', 'TradeIntensity', 'ROC_30'
    ]
    
    for w in CFG['z_windows']:
        for col in Z_BASES:
            if col in df.columns:
                df[f'{col}_z{w}'] = zscore(df[col].astype(float), w)
    
    # ============ ALPHA FEATURES ============
    # MTF Gradients
    f = lambda base: (df[f'{base}_z60'] - df[f'{base}_z1440'])
    df['alpha_rsi_mtf_grad'] = f(f'RSI_{CFG["rsi"]}')
    df['alpha_mfi_mtf_grad'] = f(f'MFI_{CFG["mfi"]}')
    df['alpha_cci_mtf_grad'] = f(f'CCI_{CFG["cci"]}')
    df['alpha_obv_mtf_grad'] = f('OBV_log')
    df['alpha_macdhist_mtf_grad'] = f('MACD_hist')
    df['alpha_aggr_mtf_grad'] = f('AggressorImbalance')
    df['alpha_buypress_mtf_grad'] = f('BuyPressure')
    df['alpha_updown_mtf_grad'] = f('UpDownVolumeRatio_60')
    df['alpha_atr_mtf_grad'] = f(f'ATR_{CFG["atr"]}')
    df['alpha_tradeint_mtf_grad'] = f('TradeIntensity')
    df['alpha_roc30_mtf_grad'] = f('ROC_30')
    
    # Divergences
    df['alpha_rsi_mfi_div_z60'] = df[f'RSI_{CFG["rsi"]}_z60'] - df[f'MFI_{CFG["mfi"]}_z60']
    df['alpha_rsi_mfi_div_z240'] = df[f'RSI_{CFG["rsi"]}_z240'] - df[f'MFI_{CFG["mfi"]}_z240']
    df['alpha_rsi_obv_div_z240'] = df[f'RSI_{CFG["rsi"]}_z240'] - df['OBV_log_z240']
    df['alpha_rsi_cci_div_z60'] = df[f'RSI_{CFG["rsi"]}_z60'] - df[f'CCI_{CFG["cci"]}_z60']
    df['alpha_stoch_wr_div_z60'] = df['STOCH_K_z60'] - df[f'Williams_R_{CFG["wr"]}_z60']
    
    # Regime Filtering
    df['alpha_aggr_z60_filt_ema1440'] = df['AggressorImbalance_z60'] * df['Close_to_EMA1440']
    df['alpha_flow_z60_filt_ema1440'] = df['FlowMomentum_z60'] * df['Close_to_EMA1440']
    df['alpha_cci_z60_filt_sma1440'] = df[f'CCI_{CFG["cci"]}_z60'] * df['Close_to_SMA1440']
    
    # Candle x Aggressor
    df['alpha_upshadow_x_aggr_z60'] = df['UpperShadowRatio'] * df['AggressorImbalance_z60']
    
    # Xử lý NaN/Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

# ==========================================
# 4. CLASS BOT GIAO DỊCH
# ==========================================
class TradingBot:
    def __init__(self):
        # Load Model XGBoost
        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)
        print(f"🤖 Model loaded: {MODEL_PATH}")
        print(f"🔗 Using Binance REST API")

    def fetch_kline_data(self):
        """Lấy dữ liệu Kline từ Binance REST API - 2 lần để đủ 1500+ nến"""
        try:
            # Binance Kline API trả về: [Open time, Open, High, Low, Close, Volume, 
            #                             Close time, Quote volume, Trades, 
            #                             Taker buy base, Taker buy quote, Ignore]
            
            symbol_formatted = SYMBOL.replace('/', '')  # BTCUSDT
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'}
            
            url = "https://api.binance.com/api/v3/klines"
            
            # Lần 1: Lấy 1000 nến mới nhất
            params1 = {
                'symbol': symbol_formatted,
                'interval': interval_map.get(TIMEFRAME, '1m'),
                'limit': 1000
            }
            
            response1 = requests.get(url, params=params1, timeout=10)
            klines1 = response1.json()
            
            # Lần 2: Lấy 1000 nến trước đó (endTime = start của batch 1)
            first_time = int(klines1[0][0])  # Open time của nến đầu tiên
            params2 = {
                'symbol': symbol_formatted,
                'interval': interval_map.get(TIMEFRAME, '1m'),
                'limit': 1000,
                'endTime': first_time - 1  # Lấy data trước timestamp này
            }
            
            response2 = requests.get(url, params=params2, timeout=10)
            klines2 = response2.json()
            
            # Gộp 2 batch (batch cũ + batch mới)
            klines = klines2 + klines1
            
            print(f"✅ Fetched {len(klines)} candles from Binance")
            
            # Parse đầy đủ dữ liệu
            full_df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'num_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert kiểu dữ liệu
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                        'taker_buy_base', 'taker_buy_quote']:
                full_df[col] = full_df[col].astype(float)
            
            full_df['num_trades'] = full_df['num_trades'].astype(int)
            full_df['timestamp'] = pd.to_datetime(full_df['open_time'], unit='ms')
            
            return full_df
            
        except Exception as e:
            print(f"❌ Error fetching Kline data: {e}")
            return None

    def run(self):
        """Vòng lặp chính của bot"""
        print(f"🚀 Bot started for {SYMBOL} (LONG + SHORT Strategy)")
        print(f"📊 LONG Threshold: {LONG_THRESHOLD_PROB:.1%} | SHORT Threshold: {SHORT_THRESHOLD_PROB:.1%}")
        
        while True:
            # --- A. Canh thời gian (đợi nến đóng) ---
            now = datetime.now()
            sleep_sec = 61 - now.second
            print(f"\n💤 Waiting {sleep_sec}s for candle close...")
            time.sleep(sleep_sec)
            
            try:
                # --- B. Lấy dữ liệu từ Binance ---
                print(f"📥 Fetching Kline data from Binance...")
                df = self.fetch_kline_data()
                
                if df is None or len(df) < 1440:
                    print(f"⚠️ Not enough data (got {len(df) if df is not None else 0} rows), need at least 1440")
                    continue
                
                current_price = df.iloc[-1]['close']
                current_time = df.iloc[-1]['timestamp']
                
                print(f"⏰ Time: {current_time} | Price: ${current_price:,.2f}")
                
                # --- C. Tính toán 49 Features ---
                print(f"🔧 Calculating 49 features...")
                full_df = calculate_features(df)
                
                # Lấy dòng cuối cùng (nến vừa đóng)
                last_row = full_df.iloc[[-1]][SELECTED_FEATURES]
                
                # Kiểm tra NaN
                if last_row.isnull().any().any():
                    print(f"⚠️ Features contain NaN, skipping prediction")
                    send_signal_to_web("ERROR", SYMBOL, current_price, 0.0)
                    continue
                
                # --- D. Dự đoán bằng XGBoost ---
                dinput = xgb.DMatrix(last_row)
                prob_up = self.model.predict(dinput)[0]  # Xác suất TĂNG
                prob_down = 1 - prob_up
                
                print(f"🔮 Prediction | UP: {prob_up:.4f} ({prob_up*100:.2f}%) | DOWN: {prob_down:.4f} ({prob_down*100:.2f}%)")
                
                # --- E. Logic Vào Lệnh (LONG + SHORT) ---
                
                # 🟢 LONG: Khi xác suất TĂNG > ngưỡng
                if prob_up > LONG_THRESHOLD_PROB:
                    confidence = prob_up * 100
                    print(f"🟢 LONG SIGNAL! (Confidence: {confidence:.2f}%)")
                    send_signal_to_web("LONG", SYMBOL, current_price, confidence)
                
                # 🔴 SHORT: Khi xác suất TĂNG < ngưỡng (tức Giảm cao)
                elif prob_up < SHORT_THRESHOLD_PROB:
                    confidence = prob_down * 100
                    print(f"🔴 SHORT SIGNAL! (Confidence: {confidence:.2f}%)")
                    send_signal_to_web("SHORT", SYMBOL, current_price, confidence)
                
                else:
                    print(f"⚪ No signal (Probability in neutral zone)")
                    send_signal_to_web("NEUTRAL", SYMBOL, current_price, float(prob_up * 100))
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                try:
                    send_signal_to_web("ERROR", SYMBOL, 0.0, 0.0)
                except:
                    pass
                time.sleep(5)

# ==========================================
# 5. CHẠY BOT
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 REALTIME TRADING BOT - 49 FEATURES - LONG/SHORT")
    print("=" * 60)
    
    bot = TradingBot()
    bot.run()
