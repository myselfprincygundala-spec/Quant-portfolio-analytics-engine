# Quant-portfolio-analytics-engine
This production-ready system is designed for modern quantitative wealth management. It handles historical multi-asset time-series simulation, calculates critical risk metrics like Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) using Monte Carlo methods,
import math
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# =====================================================================
# 1. MATHEMATICAL & MATRIX PRIMITIVES (STANDALONE SCIPY/NUMPY EMULATOR)
# =====================================================================

class MatrixMath:
    """Provides pure-python optimized linear algebra operations for portfolio optimization."""
    
    @staticmethod
    def dot_product(vec1: List[float], vec2: List[float]) -> float:
        return sum(x * y for x, y in zip(vec1, vec2))
    
    @staticmethod
    def matrix_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [MatrixMath.dot_product(row, vector) for row in matrix]
    
    @staticmethod
    def portfolio_variance(weights: List[float], cov_matrix: List[List[float]]) -> float:
        """Calculates w^T * Sigma * w."""
        intermediate = MatrixMath.matrix_multiply(cov_matrix, weights)
        return MatrixMath.dot_product(weights, intermediate)

    @staticmethod
    def generate_normal_sample(mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box-Muller transform generating standard normal distributions."""
        u1 = random.random()
        u2 = random.random()
        while u1 <= 1e-15: u1 = random.random() # Avoid log(0)
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + z0 * sigma


# =====================================================================
# 2. MARKET DATA SIMULATION ENGINE
# =====================================================================

class MarketDataEngine:
    """Generates synthetic multi-asset historical data using Geometric Brownian Motion (GBM)."""
    def __init__(self, tickers: List[str], base_prices: Dict[str, float], drift: Dict[str, float], volatility: Dict[str, float]):
        self.tickers = tickers
        self.base_prices = base_prices
        self.drift = drift
        self.volatility = volatility

    def simulate_history(self, days: int = 252) -> Dict[str, List[float]]:
        """Generates daily asset closing price arrays using stochastic modeling parameters."""
        history: Dict[str, List[float]] = {ticker: [] for ticker in self.tickers}
        dt = 1.0 / 252.0  # Daily time step fraction

        for ticker in self.tickers:
            current_price = self.base_prices[ticker]
            history[ticker].append(current_price)
            mu = self.drift[ticker]
            sigma = self.volatility[ticker]

            for _ in range(days - 1):
                epsilon = MatrixMath.generate_normal_sample(0.0, 1.0)
                # Geometric Brownian Motion Stochastic Differential Equation
                growth_factor = math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * epsilon)
                current_price *= growth_factor
                history[ticker].append(round(current_price, 2))
                
        return history

    @staticmethod
    def compute_daily_returns(price_history: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Transforms geometric price histories into stationary percentage return vectors."""
        returns_map: Dict[str, List[float]] = {}
        for ticker, prices in price_history.items():
            returns_map[ticker] = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns_map[ticker].append(ret)
        return returns_map


# =====================================================================
# 3. STATISTICAL & QUANTITATIVE ANALYTICS CORE
# =====================================================================

class QuantAnalyticsEngine:
    """Computes statistical moments and historical covariance matrices."""
    def __init__(self, returns_history: Dict[str, List[float]]):
        self.returns = returns_history
        self.tickers = list(returns_history.keys())
        self.sample_size = len(next(iter(returns_history.values())))

    def calculate_expected_returns(self) -> Dict[str, float]:
        """Calculates annualized arithmetic mean return per asset."""
        exp_returns = {}
        for ticker in self.tickers:
            mean_daily = sum(self.returns[ticker]) / self.sample_size
            exp_returns[ticker] = mean_daily * 252 # Annualized scaling
        return exp_returns

    def generate_covariance_matrix(self) -> List[List[float]]:
        """Builds an annualized variance-covariance matrix from return streams."""
        means = {t: sum(self.returns[t]) / self.sample_size for t in self.tickers}
        matrix = [[0.0 for _ in range(len(self.tickers))] for _ in range(len(self.tickers))]

        for i, t1 in enumerate(self.tickers):
            for j, t2 in enumerate(self.tickers):
                covariance = sum((self.returns[t1][k] - means[t1]) * (self.returns[t2][k] - means[t2]) for k in range(self.sample_size)) / (self.sample_size - 1)
                matrix[i][j] = covariance * 252 # Annualized scaling
        return matrix


# =====================================================================
# 4. MONTE CARLO RISK SIMULATION & EFFICIENCY OPTIMIZATION
# =====================================================================

class PortfolioOptimizer:
    """Calculates risk architectures, Monte Carlo simulations, and Efficient Frontiers."""
    def __init__(self, tickers: List[str], expected_returns: Dict[str, float], cov_matrix: List[List[float]]):
        self.tickers = tickers
        self.exp_returns = [expected_returns[t] for t in tickers]
        self.cov_matrix = cov_matrix

    def simulate_random_portfolios(self, num_portfolios: int = 5000, risk_free_rate: float = 0.04) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes a Monte Carlo simulation across asset weight permutations to find the optimal Sharpe Ratio."""
        results = []
        max_sharpe_portfolio = None
        highest_sharpe = -float('inf')

        for _ in range(num_portfolios):
            # Generate random normalized asset weights
            raw_weights = [random.random() for _ in range(len(self.tickers))]
            total_weight = sum(raw_weights)
            weights = [w / total_weight for w in raw_weights]

            # Calculate portfolio return metrics
            p_return = sum(w * r for w, r in zip(weights, self.exp_returns))
            
            # Calculate portfolio variance metrics
            p_variance = MatrixMath.portfolio_variance(weights, self.cov_matrix)
            p_volatility = math.sqrt(p_variance)

            # Sharpe Ratio = (Expected Return - Risk Free Rate) / Volatility
            sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility > 0 else 0

            portfolio_data = {
                "weights": {self.tickers[i]: round(weights[i], 4) for i in range(len(self.tickers))},
                "expected_annual_return": round(p_return, 4),
                "annualized_volatility": round(p_volatility, 4),
                "sharpe_ratio": round(sharpe, 4)
            }
            results.append(portfolio_data)

            if sharpe > highest_sharpe:
                highest_sharpe = sharpe
                max_sharpe_portfolio = portfolio_data

        return results, max_sharpe_portfolio

    def calculate_parametric_risk(self, weights: Dict[str, float], confidence_level: float = 0.95, investment: float = 1000000.0) -> Dict[str, float]:
        """Calculates Parametric Value-at-Risk (VaR) and Conditional VaR (CVaR)."""
        weight_vector = [weights[t] for t in self.tickers]
        p_return = sum(w * r for w, r in zip(weight_vector, self.exp_returns))
        p_variance = MatrixMath.portfolio_variance(weight_vector, self.cov_matrix)
        p_volatility = math.sqrt(p_variance)

        # Standard normal inverse cumulative distribution mappings (Z-scores)
        z_scores = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}
        z = z_scores.get(confidence_level, 1.6449)

        # Annualized VaR calculated and scaled down to a 10-day trading horizon
        time_horizon_factor = math.sqrt(10.0 / 252.0)
        daily_vol = p_volatility * time_horizon_factor
        daily_ret = p_return * (10.0 / 252.0)

        parametric_var_pct = (z * daily_vol) - daily_ret
        dollar_var = investment * parametric_var_pct

        # Parametric CVaR proxy formula based on Standard Normal Tail Expectations
        alpha = 1.0 - confidence_level
        phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z**2)
        parametric_cvar_pct = ((phi_z / alpha) * daily_vol) - daily_ret
        dollar_cvar = investment * parametric_cvar_pct

        return {
            "confidence_level": confidence_level,
            "value_at_risk_pct": round(parametric_var_pct, 4),
            "value_at_risk_dollar": round(dollar_var, 2),
            "conditional_var_pct": round(parametric_cvar_pct, 4),
            "conditional_var_dollar": round(dollar_cvar, 2)
        }


# =====================================================================
# 5. ENTERPRISE SYSTEM EXECUTIVE LAYER
# =====================================================================

class QuantitativeRiskConsole:
    """The central runtime hub coordinating financial models and data pipelines."""
    def __init__(self, portfolio_name: str, assets: List[str], principal_capital: float = 1000000.0):
        self.portfolio_name = portfolio_name
        self.assets = assets
        self.principal_capital = principal_capital

    def execute_analytics_pipeline(self) -> str:
        """Runs the entire quantitative pipeline: simulation, covariance, and optimization."""
        # Baseline stochastic parameters for testing
        base_prices = {"AAPL": 175.0, "MSFT": 415.0, "GOOGL": 150.0, "AMZN": 175.0}
        drifts = {"AAPL": 0.12, "MSFT": 0.14, "GOOGL": 0.10, "AMZN": 0.15}
