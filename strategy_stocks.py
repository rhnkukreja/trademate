import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import linregress
import talib
from enum import Enum

class PatternType(Enum):
    BREAKOUT = "breakout"
    TRIANGLE = "triangle" 
    FLAG = "flag"
    SUPPORT_RESISTANCE = "support_resistance"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    WEDGE = "wedge"

@dataclass
class PatternSignal:
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: float
    stop_loss: float
    breakout_level: float
    volume_confirmation: bool
    strength_score: float
    timeframe_detected: str

class ChartPatternRecognizer:
    """
    Comprehensive Chart Pattern Recognition System
    Detects various trading patterns and scores their reliability
    """
    
    def __init__(self, lookback_period: int = 50, min_pattern_length: int = 10):
        self.lookback_period = lookback_period
        self.min_pattern_length = min_pattern_length
        
    def analyze_stock(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Main function to analyze stock data and return pattern signals
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            
        Returns:
            List of PatternSignal objects with detected patterns
        """
        signals = []
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Pattern detection methods
        signals.extend(self._detect_breakout_patterns(df))
        signals.extend(self._detect_triangle_patterns(df))
        signals.extend(self._detect_flag_patterns(df))
        signals.extend(self._detect_support_resistance_break(df))
        signals.extend(self._detect_double_bottom(df))
        
        # Filter and rank signals by confidence
        signals = [s for s in signals if s.confidence >= 0.6]
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators needed for pattern recognition"""
        df = df.copy()
        
        # Moving Averages (your existing logic)
        df['sma_20'] = talib.SMA(df['close'], 20)
        df['sma_50'] = talib.SMA(df['close'], 50)
        df['sma_100'] = talib.SMA(df['close'], 100)
        df['sma_200'] = talib.SMA(df['close'], 200)
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        # Momentum
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        return df
        
    def _detect_breakout_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect breakout patterns above moving averages with volume confirmation
        This aligns with your existing strategy
        """
        signals = []
        
        if len(df) < self.min_pattern_length:
            return signals
            
        recent_data = df.tail(self.lookback_period)
        latest = recent_data.iloc[-1]
        prev = recent_data.iloc[-2]
        
        # Check for MA breakouts (your existing logic enhanced)
        mas_to_check = ['sma_20', 'sma_50', 'sma_100', 'sma_200']
        breakout_mas = []
        
        for ma in mas_to_check:
            if (latest['close'] > latest[ma] and 
                prev['close'] <= prev[ma] and
                latest[ma] is not np.nan):
                breakout_mas.append(ma)
        
        if len(breakout_mas) >= 2:  # Breaking multiple MAs
            # Volume confirmation (your 2X logic)
            volume_confirmed = latest['volume_ratio'] >= 2.0
            
            # Additional pattern strength checks
            strength_score = self._calculate_breakout_strength(recent_data, latest)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_breakout_confidence(
                recent_data, latest, len(breakout_mas), volume_confirmed, strength_score
            )
            
            if confidence >= 0.6:
                # Calculate targets (your 10% target logic)
                entry_price = latest['close']
                target_price = entry_price * 1.10  # 10% target
                stop_loss = entry_price * 0.95     # 5% stop loss
                
                signal = PatternSignal(
                    pattern_type=PatternType.BREAKOUT,
                    confidence=confidence,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    breakout_level=latest['close'],
                    volume_confirmation=volume_confirmed,
                    strength_score=strength_score,
                    timeframe_detected="15m"
                )
                signals.append(signal)
                
        return signals
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect ascending/descending/symmetrical triangles"""
        signals = []
        
        if len(df) < 20:
            return signals
            
        recent_data = df.tail(30)
        
        # Find peaks and troughs
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Find significant peaks and troughs
        peak_indices = find_peaks(highs, distance=3, prominence=np.std(highs) * 0.5)[0]
        trough_indices = find_peaks(-lows, distance=3, prominence=np.std(lows) * 0.5)[0]
        
        if len(peak_indices) >= 3 and len(trough_indices) >= 3:
            # Get recent peaks and troughs
            recent_peaks = [(i, highs[i]) for i in peak_indices[-3:]]
            recent_troughs = [(i, lows[i]) for i in trough_indices[-3:]]
            
            # Check for triangle patterns
            triangle_type = self._identify_triangle_type(recent_peaks, recent_troughs)
            
            if triangle_type:
                latest = recent_data.iloc[-1]
                convergence_point = self._calculate_triangle_convergence(recent_peaks, recent_troughs)
                
                # Check for breakout
                if self._is_triangle_breakout(recent_data, convergence_point, triangle_type):
                    confidence = self._calculate_triangle_confidence(recent_data, triangle_type)
                    
                    if confidence >= 0.65:
                        entry_price = latest['close']
                        height = max([p[1] for p in recent_peaks]) - min([t[1] for t in recent_troughs])
                        target_price = entry_price + (height * 0.8)  # 80% of triangle height
                        stop_loss = entry_price - (height * 0.3)
                        
                        signal = PatternSignal(
                            pattern_type=PatternType.TRIANGLE,
                            confidence=confidence,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            breakout_level=convergence_point,
                            volume_confirmation=latest['volume_ratio'] > 1.5,
                            strength_score=confidence,
                            timeframe_detected="15m"
                        )
                        signals.append(signal)
        
        return signals
    
    def _detect_flag_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect bullish flag patterns (consolidation after strong move)"""
        signals = []
        
        if len(df) < 25:
            return signals
            
        recent_data = df.tail(25)
        
        # Look for strong prior move (flagpole)
        flagpole_start = recent_data.iloc[0]['close']
        flag_start = recent_data.iloc[15]['close']  # Start of consolidation
        
        # Check if there was a strong upward move (flagpole)
        if flag_start > flagpole_start * 1.05:  # 5% move up
            
            # Check for consolidation (flag) - sideways price action
            flag_data = recent_data.tail(10)
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            flag_range = (flag_high - flag_low) / flag_low
            
            # Flag should be narrow range (< 3% range)
            if flag_range < 0.03:
                latest = recent_data.iloc[-1]
                
                # Check for breakout above flag
                if latest['close'] > flag_high:
                    volume_confirmed = latest['volume_ratio'] > 1.5
                    confidence = 0.7 if volume_confirmed else 0.6
                    
                    entry_price = latest['close']
                    flagpole_height = flag_start - flagpole_start
                    target_price = entry_price + flagpole_height  # Flagpole projection
                    stop_loss = flag_low
                    
                    signal = PatternSignal(
                        pattern_type=PatternType.FLAG,
                        confidence=confidence,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        breakout_level=flag_high,
                        volume_confirmation=volume_confirmed,
                        strength_score=confidence,
                        timeframe_detected="15m"
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_support_resistance_break(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect breaks above key support/resistance levels"""
        signals = []
        
        if len(df) < 30:
            return signals
            
        recent_data = df.tail(30)
        
        # Identify key support/resistance levels
        resistance_levels = self._find_resistance_levels(recent_data)
        
        latest = recent_data.iloc[-1]
        prev = recent_data.iloc[-2]
        
        for resistance_level in resistance_levels:
            # Check for breakout above resistance
            if (latest['close'] > resistance_level and 
                prev['close'] <= resistance_level and
                latest['volume_ratio'] > 1.3):
                
                # Calculate how significant this resistance was
                touches = self._count_resistance_touches(recent_data, resistance_level)
                confidence = min(0.6 + (touches * 0.1), 0.9)
                
                entry_price = latest['close']
                target_price = entry_price * 1.08  # 8% target for resistance breaks
                stop_loss = resistance_level * 0.98  # Just below resistance
                
                signal = PatternSignal(
                    pattern_type=PatternType.SUPPORT_RESISTANCE,
                    confidence=confidence,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    breakout_level=resistance_level,
                    volume_confirmation=latest['volume_ratio'] > 1.3,
                    strength_score=confidence,
                    timeframe_detected="15m"
                )
                signals.append(signal)
        
        return signals
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect double bottom reversal patterns"""
        signals = []
        
        if len(df) < 40:
            return signals
            
        recent_data = df.tail(40)
        lows = recent_data['low'].values
        
        # Find significant troughs
        trough_indices = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.3)[0]
        
        if len(trough_indices) >= 2:
            # Check last two troughs for double bottom
            last_trough = trough_indices[-1]
            prev_trough = trough_indices[-2]
            
            last_low = lows[last_trough]
            prev_low = lows[prev_trough]
            
            # Double bottom criteria
            if (abs(last_low - prev_low) / prev_low < 0.02 and  # Similar lows (within 2%)
                last_trough - prev_trough >= 10):  # Adequate separation
                
                # Find the peak between the two bottoms (neckline)
                between_data = recent_data.iloc[prev_trough:last_trough]
                neckline = between_data['high'].max()
                
                latest = recent_data.iloc[-1]
                
                # Check for neckline breakout
                if latest['close'] > neckline:
                    volume_confirmed = latest['volume_ratio'] > 1.4
                    confidence = 0.75 if volume_confirmed else 0.65
                    
                    entry_price = latest['close']
                    pattern_height = neckline - min(last_low, prev_low)
                    target_price = entry_price + pattern_height
                    stop_loss = max(last_low, prev_low)
                    
                    signal = PatternSignal(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=confidence,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        breakout_level=neckline,
                        volume_confirmation=volume_confirmed,
                        strength_score=confidence,
                        timeframe_detected="15m"
                    )
                    signals.append(signal)
        
        return signals
    
    # Helper methods for calculations
    def _calculate_breakout_strength(self, df: pd.DataFrame, latest_candle: pd.Series) -> float:
        """Calculate strength of breakout based on multiple factors"""
        strength = 0.0
        
        # Volume strength (0-0.3)
        if latest_candle['volume_ratio'] >= 3.0:
            strength += 0.3
        elif latest_candle['volume_ratio'] >= 2.0:
            strength += 0.2
        elif latest_candle['volume_ratio'] >= 1.5:
            strength += 0.1
            
        # Candle strength (0-0.2)
        candle_body = abs(latest_candle['close'] - latest_candle['open'])
        candle_range = latest_candle['high'] - latest_candle['low']
        if candle_range > 0:
            body_ratio = candle_body / candle_range
            strength += body_ratio * 0.2
            
        # Momentum (0-0.2)
        recent_closes = df['close'].tail(5)
        if len(recent_closes) == 5:
            momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
            strength += min(momentum * 4, 0.2)  # Scale momentum
            
        # RSI position (0-0.1)
        if not np.isnan(latest_candle['rsi']):
            if 50 <= latest_candle['rsi'] <= 70:  # Good momentum zone
                strength += 0.1
                
        # MACD confirmation (0-0.2)
        if (not np.isnan(latest_candle['macd']) and 
            not np.isnan(latest_candle['macd_signal']) and
            latest_candle['macd'] > latest_candle['macd_signal']):
            strength += 0.2
            
        return min(strength, 1.0)
    
    def _calculate_breakout_confidence(self, df: pd.DataFrame, latest: pd.Series, 
                                     ma_breaks: int, volume_confirmed: bool, 
                                     strength_score: float) -> float:
        """Calculate overall confidence in breakout pattern"""
        confidence = 0.3  # Base confidence
        
        # Multiple MA breaks (0-0.3)
        confidence += min(ma_breaks * 0.075, 0.3)
        
        # Volume confirmation (0-0.2)
        if volume_confirmed:
            confidence += 0.2
            
        # Strength score (0-0.3)
        confidence += strength_score * 0.3
        
        # Market context - check if broader market is supportive (0-0.2)
        # This would integrate with your Nifty/BankNifty bullish condition
        # For now, assume you pass this as a parameter or check externally
        confidence += 0.2  # Placeholder for market context
        
        return min(confidence, 1.0)
    
    def _identify_triangle_type(self, peaks: List[Tuple], troughs: List[Tuple]) -> Optional[str]:
        """Identify if peaks and troughs form a triangle pattern"""
        if len(peaks) < 2 or len(troughs) < 2:
            return None
            
        # Calculate trend lines
        peak_trend = linregress([p[0] for p in peaks], [p[1] for p in peaks])
        trough_trend = linregress([t[0] for t in troughs], [t[1] for t in troughs])
        
        # Determine triangle type based on trend line slopes
        peak_slope = peak_trend.slope
        trough_slope = trough_trend.slope
        
        if abs(peak_slope) < 0.1 and trough_slope > 0.1:
            return "ascending"
        elif peak_slope < -0.1 and abs(trough_slope) < 0.1:
            return "descending"
        elif peak_slope < -0.05 and trough_slope > 0.05:
            return "symmetrical"
            
        return None
    
    def _calculate_triangle_convergence(self, peaks: List[Tuple], troughs: List[Tuple]) -> float:
        """Calculate where triangle trend lines would converge"""
        peak_trend = linregress([p[0] for p in peaks], [p[1] for p in peaks])
        trough_trend = linregress([t[0] for t in troughs], [t[1] for t in troughs])
        
        # Find intersection point (simplified)
        latest_index = max([p[0] for p in peaks] + [t[0] for t in troughs])
        peak_projection = peak_trend.slope * latest_index + peak_trend.intercept
        trough_projection = trough_trend.slope * latest_index + trough_trend.intercept
        
        return (peak_projection + trough_projection) / 2
    
    def _is_triangle_breakout(self, df: pd.DataFrame, convergence_point: float, triangle_type: str) -> bool:
        """Check if price has broken out of triangle pattern"""
        latest = df.iloc[-1]
        
        if triangle_type in ["ascending", "symmetrical"]:
            return latest['close'] > convergence_point * 1.01  # 1% above convergence
        elif triangle_type == "descending":
            return latest['close'] < convergence_point * 0.99  # 1% below convergence
            
        return False
    
    def _calculate_triangle_confidence(self, df: pd.DataFrame, triangle_type: str) -> float:
        """Calculate confidence in triangle pattern"""
        confidence = 0.5  # Base confidence
        
        latest = df.iloc[-1]
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            confidence += 0.15
            
        # Triangle type reliability
        if triangle_type == "ascending":
            confidence += 0.1  # Generally more reliable
            
        # Price action quality
        recent_volatility = df['close'].tail(10).std()
        avg_volatility = df['close'].std()
        
        if recent_volatility < avg_volatility * 0.8:  # Lower volatility = better consolidation
            confidence += 0.1
            
        return min(confidence, 0.95)
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Find significant resistance levels in the data"""
        highs = df['high'].values
        resistance_levels = []
        
        # Find peaks
        peak_indices = find_peaks(highs, distance=3, prominence=np.std(highs) * 0.3)[0]
        
        for idx in peak_indices:
            level = highs[idx]
            # Only include if price has approached this level multiple times
            if self._count_resistance_touches(df, level) >= 2:
                resistance_levels.append(level)
                
        return resistance_levels
    
    def _count_resistance_touches(self, df: pd.DataFrame, level: float, tolerance: float = 0.005) -> int:
        """Count how many times price has touched a resistance level"""
        touches = 0
        for _, row in df.iterrows():
            if abs(row['high'] - level) / level <= tolerance:
                touches += 1
        return touches

# Usage Example for your development team
def example_usage():
    """
    Example of how to use the Chart Pattern Recognition System
    """
    
    # Sample data structure (replace with your actual data)
    # This should come from your data provider (Zerodha, Alpha Vantage, etc.)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15T'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Initialize the pattern recognizer
    recognizer = ChartPatternRecognizer(lookback_period=50, min_pattern_length=10)
    
    try:
        # Analyze the stock data
        signals = recognizer.analyze_stock(sample_data)
        
        print(f"Found {len(signals)} pattern signals:")
        
        for i, signal in enumerate(signals):
            print(f"\nSignal {i+1}:")
            print(f"  Pattern: {signal.pattern_type.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Entry: ₹{signal.entry_price:.2f}")
            print(f"  Target: ₹{signal.target_price:.2f}")
            print(f"  Stop Loss: ₹{signal.stop_loss:.2f}")
            print(f"  Volume Confirmed: {signal.volume_confirmation}")
            print(f"  Strength Score: {signal.strength_score:.2f}")
            
    except Exception as e:
        print(f"Error in pattern recognition: {e}")

# Integration with your existing strategy
class EnhancedTradingStrategy:
    """
    Integration of pattern recognition with your existing MA breakout strategy
    """
    
    def __init__(self):
        self.pattern_recognizer = ChartPatternRecognizer()
        
    def analyze_stock_for_trading(self, stock_data: pd.DataFrame, 
                                 nifty_bullish: bool, 
                                 min_confidence: float = 0.7) -> Dict:
        """
        Enhanced analysis combining your MA logic with pattern recognition
        """
        
        # Your existing MA breakout logic
        ma_signals = self._check_ma_breakout(stock_data)
        
        # Pattern recognition signals
        pattern_signals = self.pattern_recognizer.analyze_stock(stock_data)
        
        # Filter high-confidence patterns
        strong_patterns = [s for s in pattern_signals if s.confidence >= min_confidence]
        
        # Combine signals
        if ma_signals and strong_patterns and nifty_bullish:
            # Take the highest confidence pattern signal
            best_pattern = max(strong_patterns, key=lambda x: x.confidence)
            
            return {
                'action': 'BUY',
                'pattern_type': best_pattern.pattern_type.value,
                'confidence': best_pattern.confidence,
                'entry_price': best_pattern.entry_price,
                'target_price': best_pattern.target_price,
                'stop_loss': best_pattern.stop_loss,
                'volume_confirmed': best_pattern.volume_confirmation,
                'ma_breakout': True,
                'market_supportive': nifty_bullish
            }
        
        return {'action': 'WAIT', 'reason': 'Insufficient signal strength'}
    
    def _check_ma_breakout(self, df: pd.DataFrame) -> bool:
        """Your existing MA breakout logic"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate MAs
        sma_20 = talib.SMA(df['close'], 20).iloc[-1]
        sma_50 = talib.SMA(df['close'], 50).iloc[-1]
        
        # Check for breakout
        return (latest['close'] > sma_20 and 
                latest['close'] > sma_50 and
                prev['close'] <= sma_20)

if __name__ == "__main__":
    example_usage()
